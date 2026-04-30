from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import subprocess
import threading
import traceback
import time
from pathlib import Path

try:
    from tkinter import filedialog, messagebox, scrolledtext
    import tkinter as tk
except ModuleNotFoundError:
    filedialog = None
    messagebox = None
    scrolledtext = None
    tk = None

from app import DEBUG_DIR, RESULTS_DIR, call_nvidia_api, check_ffmpeg, extract_json_text, read_b64
from segment_describer import (
    SEGMENT_SECONDS,
    format_time,
    get_video_duration,
    make_segment_proxy_video,
)


APP_DIR = Path(__file__).resolve().parent
RUNNING_CLIP_DIR = APP_DIR / "running_clips_original"
MIN_RUNNING_CONFIDENCE = 0.45
NVIDIA_RETRIES = 3
DEFAULT_KEYWORDS = "奔跑、跑步、快跑、冲刺、追逐、逃跑、运动奔跑"


def safe_stem(path: str | Path) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in Path(path).stem)


def safe_text_slug(text: str, default: str = "target") -> str:
    slug = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug[:60] or default


def normalize_keywords(keywords: str | None) -> str:
    return (keywords or DEFAULT_KEYWORDS).strip() or DEFAULT_KEYWORDS


def get_video_output_dir(video_path: str | Path, keywords: str | None = None) -> Path:
    input_path = Path(video_path).expanduser().resolve()
    keyword_text = normalize_keywords(keywords)
    suffix = "running" if keyword_text == DEFAULT_KEYWORDS else safe_text_slug(keyword_text)
    return input_path.parent / f"{safe_stem(input_path)}_{suffix}_clips"


def append_log(log_path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as file:
        file.write(f"[{timestamp}] {message}\n")


def read_json_file(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def write_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def upsert_by_segment(items: list[dict], item: dict) -> list[dict]:
    segment_index = item.get("segment_index")
    return [existing for existing in items if existing.get("segment_index") != segment_index] + [item]


def parse_time_to_seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        raise ValueError("空时间值")

    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)

    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"无法解析时间：{value}")


def parse_confidence(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    confidence_words = {
        "high": 0.9,
        "medium": 0.6,
        "low": 0.3,
        "very high": 0.95,
        "very low": 0.15,
        "高": 0.9,
        "中": 0.6,
        "中等": 0.6,
        "低": 0.3,
    }
    if text in confidence_words:
        return confidence_words[text]
    if text.endswith("%"):
        return float(text.rstrip("%")) / 100
    return float(text)


def build_detection_prompt(segment_index: int, start_time: str, end_time: str, keywords: str) -> str:
    return (
        "你是视频内容检测器。"
        f"请检测这个视频片段里是否出现这些目标画面或动作：{keywords}。"
        "只返回合法 JSON，不要解释，不要 Markdown，不要代码块。"
        "如果没有目标画面或动作，返回 []。"
        "如果有，返回 JSON 数组。每个元素必须包含："
        "relative_start_time, relative_end_time, confidence, description, labels。"
        "relative_start_time 和 relative_end_time 必须是相对于当前片段开头的时间，格式 HH:MM:SS 或 MM:SS，"
        f"当前片段在原视频中的范围是 {start_time} 到 {end_time}。"
        "description 用中文，最多 40 个中文字。labels 是中文字符串数组。"
        "只标出真实可见或可听判断的目标内容；不确定时不要标出。"
    )


def detect_running_in_proxy(
    proxy_path: Path,
    segment_index: int,
    start_time: str,
    end_time: str,
    keywords: str,
    log_path: Path | None = None,
) -> list[dict]:
    video_b64 = read_b64(proxy_path)
    video_data_url = f"data:video/mp4;base64,{video_b64}"
    prompt = build_detection_prompt(segment_index, start_time, end_time, keywords)
    last_error = None

    for attempt in range(1, NVIDIA_RETRIES + 1):
        try:
            if log_path:
                append_log(log_path, f"segment {segment_index}: NVIDIA request attempt {attempt}")
            raw_response = call_nvidia_api(video_data_url, prompt)
        except Exception as exc:
            last_error = exc
            if log_path:
                append_log(log_path, f"segment {segment_index}: NVIDIA request attempt {attempt} failed: {exc}")
            if attempt < NVIDIA_RETRIES:
                time.sleep(2 * attempt)
                continue
            raise

        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        (DEBUG_DIR / f"running_detect_segment_{segment_index:03d}_last_response.txt").write_text(
            raw_response,
            encoding="utf-8",
        )
        if log_path:
            append_log(log_path, f"segment {segment_index}: NVIDIA response length {len(raw_response)} chars")

        try:
            json_text = extract_json_text(raw_response)
            parsed = json.loads(json_text)
            break
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            (DEBUG_DIR / f"running_detect_segment_{segment_index:03d}_raw_attempt_{attempt}.txt").write_text(
                raw_response,
                encoding="utf-8",
            )
            if log_path:
                append_log(log_path, f"segment {segment_index}: JSON parse failed on attempt {attempt}: {exc}")
            prompt = (
                "你刚才没有返回完整合法 JSON。请重新输出，只能返回 JSON 数组。"
                "没有目标内容就返回 []。"
            ) + build_detection_prompt(segment_index, start_time, end_time, keywords)
            if attempt < NVIDIA_RETRIES:
                time.sleep(1)
    else:
        raise ValueError(f"第 {segment_index} 段目标检测返回不是合法 JSON：{last_error}")

    if isinstance(parsed, dict):
        parsed = parsed.get("segments", parsed.get("running_segments", parsed.get("matched_segments", [])))
    if not isinstance(parsed, list):
        raise ValueError(f"第 {segment_index} 段目标检测返回的 JSON 不是数组。")
    return [item for item in parsed if isinstance(item, dict)]


def normalize_detection(
    item: dict,
    segment_index: int,
    segment_start_seconds: float,
    segment_duration: float,
    video_duration: float,
) -> dict | None:
    confidence = parse_confidence(item.get("confidence", 0))
    if confidence < MIN_RUNNING_CONFIDENCE:
        return None

    raw_start = item.get("relative_start_time", item.get("start_time", item.get("start")))
    raw_end = item.get("relative_end_time", item.get("end_time", item.get("end")))
    if raw_start is None or raw_end is None:
        return None

    start_value = parse_time_to_seconds(raw_start)
    end_value = parse_time_to_seconds(raw_end)

    if start_value > segment_duration + 3 and segment_start_seconds <= start_value <= segment_start_seconds + segment_duration + 3:
        absolute_start = start_value
    else:
        absolute_start = segment_start_seconds + start_value

    if end_value > segment_duration + 3 and segment_start_seconds <= end_value <= segment_start_seconds + segment_duration + 3:
        absolute_end = end_value
    else:
        absolute_end = segment_start_seconds + end_value

    absolute_start = max(0, min(video_duration, absolute_start))
    absolute_end = max(0, min(video_duration, absolute_end))
    if absolute_end <= absolute_start:
        return None

    labels = item.get("labels", [])
    if isinstance(labels, str):
        labels = [part.strip() for part in re.split(r"[，,、;；]", labels) if part.strip()]

    return {
        "segment_index": segment_index,
        "absolute_start_seconds": round(absolute_start, 3),
        "absolute_end_seconds": round(absolute_end, 3),
        "absolute_start_time": format_time(absolute_start),
        "absolute_end_time": format_time(absolute_end),
        "relative_start_time": format_time(absolute_start - segment_start_seconds),
        "relative_end_time": format_time(absolute_end - segment_start_seconds),
        "confidence": confidence,
        "description": str(item.get("description", "")).strip(),
        "labels": labels,
    }


def extract_original_quality_clip(
    video_path: str | Path,
    start_seconds: float,
    end_seconds: float,
    clip_index: int,
    keywords: str,
    output_dir: Path | None = None,
) -> Path:
    ffmpeg = check_ffmpeg()
    input_path = Path(video_path).expanduser().resolve()
    if output_dir is None:
        output_dir = get_video_output_dir(input_path, keywords)
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_slug = "running" if normalize_keywords(keywords) == DEFAULT_KEYWORDS else safe_text_slug(keywords)
    output_path = output_dir / (
        f"{safe_stem(input_path)}_{clip_slug}_{clip_index:03d}_"
        f"{format_time(start_seconds).replace(':', '-')}_to_{format_time(end_seconds).replace(':', '-')}.mp4"
    )
    duration = max(0.05, end_seconds - start_seconds)

    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-dn",
        "-sn",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"原画质切片失败：{result.stderr.strip() or result.stdout.strip()}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"原画质切片失败：输出文件为空 {output_path}")
    return output_path


def detect_and_extract_running_clips(
    video_path: str | Path,
    keywords: str | None = None,
    segment_seconds: int = SEGMENT_SECONDS,
    max_segments: int | None = None,
    fresh: bool = False,
    retry_failed: bool = False,
    stop_event: threading.Event | None = None,
    progress=None,
) -> list[dict]:
    check_ffmpeg()
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError("未设置 NVIDIA_API_KEY 环境变量。")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    keywords = normalize_keywords(keywords)

    duration = get_video_duration(video_path)
    total_segments = math.ceil(duration / segment_seconds)
    if max_segments is not None:
        total_segments = min(total_segments, max_segments)

    extracted = []
    video_output_dir = get_video_output_dir(video_path, keywords)
    video_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = video_output_dir / "matched_clips.json"
    failed_output_path = video_output_dir / "failed_segments.json"
    state_path = video_output_dir / "resume_state.json"
    run_log_path = video_output_dir / "run.log"
    latest_output_path = RESULTS_DIR / "matched_clips.json"
    latest_failed_output_path = RESULTS_DIR / "matched_clip_failed_segments.json"

    if fresh:
        for old_clip in video_output_dir.glob("*.mp4"):
            old_clip.unlink(missing_ok=True)
        for old_file in (output_path, failed_output_path, state_path):
            old_file.unlink(missing_ok=True)
        run_log_path.write_text("", encoding="utf-8")
        append_log(run_log_path, "fresh run requested; old clips/state/json removed")

    extracted = read_json_file(output_path, [])
    failed_segments = read_json_file(failed_output_path, [])
    state = read_json_file(state_path, {})
    completed_segments = set(state.get("completed_segments", []))
    skipped_failed_segments = {item.get("segment_index") for item in failed_segments if item.get("segment_index")}
    if retry_failed:
        skipped_failed_segments.clear()
        append_log(run_log_path, "retry failed segments requested")
    if not completed_segments and extracted:
        completed_segments.update(item.get("segment_index") for item in extracted if item.get("segment_index"))
    clip_index = len(extracted) + 1

    append_log(run_log_path, f"start/resume video={Path(video_path).expanduser().resolve()}")
    append_log(run_log_path, f"keywords={keywords}")
    append_log(run_log_path, f"duration={duration:.3f}s total_segments={total_segments} segment_seconds={segment_seconds}")

    for zero_index in range(total_segments):
        if stop_event and stop_event.is_set():
            append_log(run_log_path, "stop requested before next segment")
            if progress:
                progress("已请求停止，当前进度已保存")
            break

        segment_index = zero_index + 1
        segment_start = zero_index * segment_seconds
        actual_duration = min(segment_seconds, max(0.1, duration - segment_start))
        segment_end = min(duration, segment_start + actual_duration)
        start_time = format_time(segment_start)
        end_time = format_time(segment_end)

        if segment_index in completed_segments:
            if progress:
                progress(f"第 {segment_index}/{total_segments} 段已完成，跳过")
            append_log(run_log_path, f"segment {segment_index}: skipped completed")
            continue

        if segment_index in skipped_failed_segments:
            if progress:
                progress(f"第 {segment_index}/{total_segments} 段此前失败，跳过")
            append_log(run_log_path, f"segment {segment_index}: skipped previous failure")
            continue

        if progress:
            progress(f"正在生成检测 proxy：第 {segment_index}/{total_segments} 段 {start_time} - {end_time}")
        append_log(run_log_path, f"segment {segment_index}/{total_segments}: proxy start {start_time}-{end_time}")
        proxy_path = make_segment_proxy_video(video_path, segment_index, segment_start, actual_duration)
        append_log(run_log_path, f"segment {segment_index}: proxy={proxy_path} size={proxy_path.stat().st_size}")

        if stop_event and stop_event.is_set():
            append_log(run_log_path, f"segment {segment_index}: stop requested after proxy generation")
            if progress:
                progress("已请求停止，当前进度已保存")
            break

        if progress:
            progress(f"正在检测目标内容：第 {segment_index}/{total_segments} 段")
        try:
            detections = detect_running_in_proxy(proxy_path, segment_index, start_time, end_time, keywords, run_log_path)
        except Exception as exc:
            failure = {
                "segment_index": segment_index,
                "start_time": start_time,
                "end_time": end_time,
                "proxy_path": str(proxy_path),
                "error": str(exc),
            }
            failed_segments = upsert_by_segment(failed_segments, failure)
            skipped_failed_segments.add(segment_index)
            state = {
                "video_path": str(Path(video_path).expanduser().resolve()),
                "keywords": keywords,
                "duration": duration,
                "segment_seconds": segment_seconds,
                "total_segments": total_segments,
                "completed_segments": sorted(completed_segments),
                "failed_segments": sorted(skipped_failed_segments),
            }
            write_json_file(failed_output_path, failed_segments)
            write_json_file(latest_failed_output_path, failed_segments)
            write_json_file(state_path, state)
            append_log(run_log_path, f"segment {segment_index}: detection failed after retries: {exc}")
            if progress:
                progress(f"第 {segment_index}/{total_segments} 段检测失败，已记录并继续：{exc}")
            continue

        normalized = []
        for item in detections:
            try:
                normalized_item = normalize_detection(
                    item,
                    segment_index,
                    segment_start,
                    actual_duration,
                    duration,
                )
            except Exception as exc:
                append_log(run_log_path, f"segment {segment_index}: skipped invalid detection {item}: {exc}")
                continue
            if normalized_item:
                normalized.append(normalized_item)

        if not normalized and progress:
            progress(f"第 {segment_index}/{total_segments} 段未检测到目标内容")
        append_log(run_log_path, f"segment {segment_index}: normalized detections={len(normalized)}")

        for item in normalized:
            if stop_event and stop_event.is_set():
                append_log(run_log_path, f"segment {segment_index}: stop requested before extracting remaining clips")
                break
            if progress:
                progress(f"正在切出原画质目标片段：{item['absolute_start_time']} - {item['absolute_end_time']}")
            clip_path = extract_original_quality_clip(
                video_path,
                item["absolute_start_seconds"],
                item["absolute_end_seconds"],
                clip_index,
                keywords,
                video_output_dir,
            )
            item["keywords"] = keywords
            item["clip_path"] = str(clip_path)
            item["proxy_path"] = str(proxy_path)
            extracted.append(item)
            append_log(run_log_path, f"segment {segment_index}: extracted clip={clip_path}")
            clip_index += 1
            write_json_file(output_path, extracted)
            write_json_file(latest_output_path, extracted)

        completed_segments.add(segment_index)
        state = {
            "video_path": str(Path(video_path).expanduser().resolve()),
            "keywords": keywords,
            "duration": duration,
            "segment_seconds": segment_seconds,
            "total_segments": total_segments,
            "completed_segments": sorted(completed_segments),
            "failed_segments": sorted(skipped_failed_segments),
        }
        write_json_file(state_path, state)

        if progress:
            progress(f"第 {segment_index}/{total_segments} 段完成，累计原画质片段 {len(extracted)} 个")
        append_log(run_log_path, f"segment {segment_index}: done total_clips={len(extracted)}")

    write_json_file(output_path, extracted)
    write_json_file(latest_output_path, extracted)
    write_json_file(failed_output_path, failed_segments)
    write_json_file(latest_failed_output_path, failed_segments)
    state = {
        "video_path": str(Path(video_path).expanduser().resolve()),
        "keywords": keywords,
        "duration": duration,
        "segment_seconds": segment_seconds,
        "total_segments": total_segments,
        "completed_segments": sorted(completed_segments),
        "failed_segments": sorted(skipped_failed_segments),
    }
    write_json_file(state_path, state)
    append_log(run_log_path, f"finished clips={len(extracted)} failed_segments={len(failed_segments)}")
    return extracted


class RunningClipExtractorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NVIDIA NIM 关键片段原画质提取")
        self.root.geometry("920x680")
        self.video_path = tk.StringVar(value="未选择")
        self.keywords = tk.StringVar(value=DEFAULT_KEYWORDS)
        self.status = tk.StringVar(value="就绪")
        self.output_path = tk.StringVar(value="选择视频后自动生成")
        self.clip_dir = tk.StringVar(value="选择视频后自动生成")
        self.log_path = tk.StringVar(value="选择视频后自动生成")
        self.fresh_run = tk.BooleanVar(value=False)
        self.retry_failed = tk.BooleanVar(value=False)
        self.stop_event = threading.Event()
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=14, pady=14)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Button(frame, text="选择视频文件", command=self.choose_video).grid(row=0, column=0, sticky="w")
        self.start_button = tk.Button(frame, text="检测并切出目标片段", command=self.start_analysis, state=tk.DISABLED)
        self.start_button.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.stop_button = tk.Button(frame, text="停止处理", command=self.stop_analysis, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=2, sticky="w", padx=(10, 0))
        tk.Checkbutton(frame, text="重新从头处理", variable=self.fresh_run).grid(row=0, column=3, sticky="w", padx=(10, 0))
        tk.Checkbutton(frame, text="重试失败段", variable=self.retry_failed).grid(row=0, column=4, sticky="w", padx=(10, 0))

        tk.Label(frame, text="已选择的视频路径：").grid(row=1, column=0, sticky="nw", pady=(14, 0))
        tk.Label(frame, textvariable=self.video_path, anchor="w", justify="left", wraplength=700).grid(
            row=1,
            column=1,
            sticky="we",
            pady=(14, 0),
        )

        tk.Label(frame, text="筛查关键字：").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        keyword_entry = tk.Entry(frame, textvariable=self.keywords)
        keyword_entry.grid(row=2, column=1, columnspan=2, sticky="we", pady=(10, 0))
        keyword_entry.bind("<KeyRelease>", lambda _event: self._refresh_output_paths())

        tk.Label(frame, text="原画质片段目录：").grid(row=3, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.clip_dir, anchor="w", justify="left", wraplength=700).grid(
            row=3,
            column=1,
            sticky="we",
            pady=(10, 0),
        )

        tk.Label(frame, text="JSON 结果路径：").grid(row=4, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.output_path, anchor="w", justify="left", wraplength=700).grid(
            row=4,
            column=1,
            sticky="we",
            pady=(10, 0),
        )

        tk.Label(frame, text="运行日志路径：").grid(row=5, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.log_path, anchor="w", justify="left", wraplength=700).grid(
            row=5,
            column=1,
            sticky="we",
            pady=(10, 0),
        )

        tk.Label(frame, text="当前处理状态：").grid(row=6, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.status, anchor="w", justify="left", wraplength=700).grid(
            row=6,
            column=1,
            sticky="we",
            pady=(10, 0),
        )

        tk.Label(frame, text="检测与切片结果：").grid(row=7, column=0, columnspan=2, sticky="w", pady=(18, 6))
        self.output = scrolledtext.ScrolledText(frame, height=24, wrap=tk.WORD)
        self.output.grid(row=8, column=0, columnspan=3, sticky="nsew")

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(8, weight=1)

    def choose_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv *.webm"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        self.video_path.set(file_path)
        self._refresh_output_paths()
        self.status.set("已选择视频")
        self.output.delete("1.0", tk.END)
        self.start_button.config(state=tk.NORMAL)

    def _refresh_output_paths(self) -> None:
        selected = self.video_path.get()
        if selected == "未选择":
            return
        output_dir = get_video_output_dir(selected, self.keywords.get())
        self.clip_dir.set(str(output_dir))
        self.output_path.set(str(output_dir / "matched_clips.json"))
        self.log_path.set(str(output_dir / "run.log"))

    def start_analysis(self) -> None:
        selected = self.video_path.get()
        if selected == "未选择":
            messagebox.showwarning("未选择视频", "请先选择一个本地视频文件。")
            return

        try:
            check_ffmpeg()
        except RuntimeError as exc:
            messagebox.showerror("缺少 ffmpeg", str(exc))
            return

        if not os.getenv("NVIDIA_API_KEY"):
            messagebox.showerror("缺少 API Key", "请先设置 NVIDIA_API_KEY 环境变量。")
            return

        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        self.status.set("开始处理")
        keywords = normalize_keywords(self.keywords.get())
        self._refresh_output_paths()
        thread = threading.Thread(
            target=self._run_analysis,
            args=(selected, keywords, self.fresh_run.get(), self.retry_failed.get()),
            daemon=True,
        )
        thread.start()

    def stop_analysis(self) -> None:
        self.stop_event.set()
        self.status.set("正在停止，等待当前 ffmpeg/API 调用结束")
        self.output.insert(tk.END, "已请求停止；会在当前片段安全结束后停止。\n")
        self.output.see(tk.END)
        self.stop_button.config(state=tk.DISABLED)

    def _run_analysis(self, selected: str, keywords: str, fresh: bool, retry_failed: bool) -> None:
        try:
            def progress(message: str) -> None:
                self.events.put(("progress", message))

            results = detect_and_extract_running_clips(
                selected,
                keywords=keywords,
                fresh=fresh,
                retry_failed=retry_failed,
                stop_event=self.stop_event,
                progress=progress,
            )
            self.events.put(("success", json.dumps(results, ensure_ascii=False, indent=2)))
        except Exception as exc:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            (DEBUG_DIR / "running_clip_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self.events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            event, payload = self.events.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_events)
            return

        if event == "progress":
            self.status.set(payload)
            self.output.insert(tk.END, payload + "\n")
            self.output.see(tk.END)
        elif event == "success":
            self.status.set("完成，结果已保存")
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, payload)
            messagebox.showinfo("完成", "目标片段检测与原画质切片完成。")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
        else:
            self.status.set("失败")
            self.output.insert(tk.END, "\n" + payload)
            messagebox.showerror("处理失败", payload)
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

        self.root.after(100, self._poll_events)


def run_gui() -> None:
    if tk is None:
        raise RuntimeError("当前 Python 环境不可用 tkinter。请安装/切换到带 tkinter 支持的 Python 3。")
    root = tk.Tk()
    RunningClipExtractorApp(root)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect target content in 60s proxy clips and extract original-quality clips.")
    parser.add_argument("--video", help="Run without GUI and analyze this video path.")
    parser.add_argument("--keywords", default=DEFAULT_KEYWORDS, help="Target keywords to detect, separated by comma or Chinese comma.")
    parser.add_argument("--max-segments", type=int, help="Only analyze the first N segments in CLI mode.")
    parser.add_argument("--fresh", action="store_true", help="Ignore resume state and start from the first segment.")
    parser.add_argument("--retry-failed", action="store_true", help="Retry segments recorded in failed_segments.json.")
    args = parser.parse_args()

    if args.video:
        results = detect_and_extract_running_clips(
            args.video,
            keywords=args.keywords,
            max_segments=args.max_segments,
            fresh=args.fresh,
            retry_failed=args.retry_failed,
            progress=print,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        run_gui()


if __name__ == "__main__":
    main()
