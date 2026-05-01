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
from pathlib import Path

try:
    from tkinter import filedialog, messagebox, scrolledtext
    import tkinter as tk
except ModuleNotFoundError:
    filedialog = None
    messagebox = None
    scrolledtext = None
    tk = None

from app import (
    DEBUG_DIR,
    MODEL_NAME,
    RESULTS_DIR,
    TARGET_BYTES,
    check_ffmpeg,
    extract_json_text,
    read_b64,
)


APP_DIR = Path(__file__).resolve().parent
SEGMENT_PROXY_DIR = APP_DIR / "segment_proxies"
SEGMENT_SECONDS = 60
WINDOWS_LARGE_VIDEO_MODES = {"fast", "balanced", "gentle", "low_cpu"}


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_video_duration(video_path: str | Path) -> float:
    ffmpeg = check_ffmpeg()
    ffprobe = str(Path(ffmpeg).with_name("ffprobe"))
    if not Path(ffprobe).exists():
        ffprobe = "ffprobe"

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=nw=1:nk=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe 读取视频时长失败：{result.stderr.strip()}")
    return float(result.stdout.strip())


def windows_large_video_thread_args(mode: str) -> list[str]:
    if mode == "balanced":
        return ["-threads", "3", "-filter_threads", "1", "-filter_complex_threads", "1"]
    if mode == "gentle":
        return ["-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1"]
    if mode == "low_cpu":
        return ["-threads", "1", "-filter_threads", "1", "-filter_complex_threads", "1"]
    return []


def run_ffmpeg(cmd: list[str], below_normal_priority: bool = False) -> subprocess.CompletedProcess:
    creationflags = subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" and below_normal_priority else 0
    return subprocess.run(cmd, capture_output=True, text=True, creationflags=creationflags)


def make_windows_large_video_proxy(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    mode: str,
    log_callback=None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thread_args = windows_large_video_thread_args(mode)
    temp_nvenc = output_path.with_name(output_path.stem + "_nvenc_try.mp4")
    temp_cpu = output_path.with_name(output_path.stem + "_cpu_fallback.mp4")
    scale_filter = "scale=-2:240:flags=fast_bilinear,format=yuv420p"

    if log_callback:
        log_callback(
            "proxy mode=%s windows_large_video=True encoder=h264_nvenc threads=%s output=%s"
            % (mode, "unlimited" if not thread_args else " ".join(thread_args), temp_nvenc)
        )

    nvenc_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        *thread_args,
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-dn",
        "-sn",
        "-vf",
        scale_filter,
        "-c:v",
        "h264_nvenc",
        "-preset",
        "fast",
        "-b:v",
        "800k",
        "-profile:v",
        "main",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(temp_nvenc),
    ]
    result = run_ffmpeg(nvenc_cmd, below_normal_priority=True)
    if result.returncode == 0 and temp_nvenc.exists() and temp_nvenc.stat().st_size > 0:
        if output_path.exists():
            output_path.unlink()
        temp_nvenc.rename(output_path)
        temp_cpu.unlink(missing_ok=True)
        if log_callback:
            log_callback(f"proxy success encoder=h264_nvenc output={output_path}")
        return output_path

    nvenc_error = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
    temp_nvenc.unlink(missing_ok=True)
    if log_callback:
        log_callback(f"h264_nvenc 不可用或失败，已回退到 CPU 低线程模式: {nvenc_error}")

    cpu_cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-filter_complex_threads",
        "1",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-dn",
        "-sn",
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(temp_cpu),
    ]
    if log_callback:
        log_callback(f"proxy fallback mode={mode} windows_large_video=True encoder=libx264 threads=1 output={temp_cpu}")
    result = run_ffmpeg(cpu_cmd, below_normal_priority=True)
    if result.returncode != 0 or not temp_cpu.exists() or temp_cpu.stat().st_size == 0:
        cpu_error = result.stderr.strip() or result.stdout.strip() or "unknown ffmpeg error"
        temp_cpu.unlink(missing_ok=True)
        raise RuntimeError(f"Windows 大视频 proxy 生成失败：{cpu_error}")

    if output_path.exists():
        output_path.unlink()
    temp_cpu.rename(output_path)
    if log_callback:
        log_callback(f"proxy success encoder=libx264 fallback=True output={output_path}")
    return output_path


def make_segment_proxy_video(
    video_path: str | Path,
    segment_index: int,
    start_seconds: float,
    duration_seconds: float = SEGMENT_SECONDS,
    output_dir: Path = SEGMENT_PROXY_DIR,
    target_bytes: int = TARGET_BYTES,
    ffmpeg_threads: int | None = None,
    low_cpu: bool = False,
    include_audio: bool = True,
    windows_large_video_mode: str = "default",
    log_callback=None,
) -> Path:
    ffmpeg = check_ffmpeg()
    input_path = Path(video_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"视频文件不存在：{input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in input_path.stem)
    output_path = output_dir / f"{safe_stem}_segment_{segment_index:03d}_proxy.mp4"

    if os.name == "nt" and windows_large_video_mode in WINDOWS_LARGE_VIDEO_MODES:
        return make_windows_large_video_proxy(
            ffmpeg,
            input_path,
            output_path,
            start_seconds,
            duration_seconds,
            windows_large_video_mode,
            log_callback=log_callback,
        )

    if log_callback:
        log_callback(f"proxy mode=default windows_large_video=False encoder=libx264 output={output_path}")

    if low_cpu:
        attempts = [
            {"height": 240, "video_bitrate": "260k", "audio_bitrate": "24k"},
            {"height": 240, "video_bitrate": "180k", "audio_bitrate": "24k"},
            {"height": 180, "video_bitrate": "140k", "audio_bitrate": "16k"},
        ]
        preset = "ultrafast"
    else:
        attempts = [
            {"height": 480, "video_bitrate": "900k", "audio_bitrate": "64k"},
            {"height": 360, "video_bitrate": "650k", "audio_bitrate": "48k"},
            {"height": 360, "video_bitrate": "450k", "audio_bitrate": "32k"},
            {"height": 240, "video_bitrate": "300k", "audio_bitrate": "32k"},
            {"height": 240, "video_bitrate": "180k", "audio_bitrate": "24k"},
        ]
        preset = "veryfast"

    last_error = None
    for attempt_index, attempt in enumerate(attempts, start=1):
        candidate = output_dir / f"{safe_stem}_segment_{segment_index:03d}_try_{attempt_index}.mp4"
        scale_filter = f"scale=-2:{attempt['height']}:force_original_aspect_ratio=decrease"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            f"{start_seconds:.3f}",
            "-t",
            f"{duration_seconds:.3f}",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-dn",
            "-sn",
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-b:v",
            attempt["video_bitrate"],
            "-maxrate",
            attempt["video_bitrate"],
            "-bufsize",
            str(int(attempt["video_bitrate"].rstrip("k")) * 2) + "k",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ]
        if ffmpeg_threads is not None:
            cmd.extend(["-threads", str(max(1, ffmpeg_threads))])
        if include_audio:
            cmd.extend([
                "-map",
                "0:a?",
                "-c:a",
                "aac",
                "-b:a",
                attempt["audio_bitrate"],
            ])
        else:
            cmd.extend(["-an"])
        cmd.append(str(candidate))
        result = run_ffmpeg(cmd)
        if result.returncode != 0:
            last_error = result.stderr.strip() or result.stdout.strip()
            candidate.unlink(missing_ok=True)
            if log_callback:
                log_callback(f"proxy failed encoder=libx264 attempt={attempt_index}: {last_error}")
            continue

        if candidate.stat().st_size <= target_bytes:
            if output_path.exists():
                output_path.unlink()
            candidate.rename(output_path)
            if log_callback:
                log_callback(f"proxy success encoder=libx264 attempt={attempt_index} output={output_path}")
            return output_path

        if not output_path.exists() or candidate.stat().st_size < output_path.stat().st_size:
            if output_path.exists():
                output_path.unlink()
            candidate.rename(output_path)
        else:
            candidate.unlink(missing_ok=True)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        raise RuntimeError(
            f"第 {segment_index} 段 proxy 仍大于 10MB（{size_mb:.2f}MB），"
            "请缩短切片长度或继续降低码率。"
        )
    raise RuntimeError(f"第 {segment_index} 段 ffmpeg 转码失败：{last_error or '未知错误'}")


def build_segment_description_prompt(segment_index: int, start_time: str, end_time: str) -> str:
    return (
        "你是视频内容分析器。请描述这个视频片段中发生了什么。"
        "只返回一个紧凑的合法 JSON 对象，不要解释，不要 Markdown，不要代码块。"
        "所有字符串内容必须使用中文。"
        f"固定字段和值：segment_index={segment_index}, start_time=\"{start_time}\", end_time=\"{end_time}\"。"
        "字段：segment_index,start_time,end_time,summary,visible_people,scene,actions,audio_notes,confidence。"
        "summary 最多 80 个中文字；visible_people、scene、audio_notes 都最多 40 个中文字；"
        "actions 最多 6 个中文短语；confidence 是 0 到 1 的数字。"
        "只根据视频画面和音频判断，不要臆测。"
    )


def call_nvidia_segment_api(video_data_url: str, prompt: str) -> str:
    from openai import OpenAI

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 NVIDIA_API_KEY 环境变量。")

    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
    )

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": video_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0,
        top_p=1,
        max_tokens=1200,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
            "mm_processor_kwargs": {"use_audio_in_video": True},
        },
        stream=True,
    )

    chunks = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        content = getattr(chunk.choices[0].delta, "content", None)
        if content:
            chunks.append(content)

    raw_text = "".join(chunks).strip()
    if not raw_text:
        raise RuntimeError("NVIDIA API 返回为空，未收到 delta.content。")
    return raw_text


def describe_segment(proxy_path: Path, segment_index: int, start_time: str, end_time: str) -> dict:
    video_b64 = read_b64(proxy_path)
    video_data_url = f"data:video/mp4;base64,{video_b64}"
    prompt = build_segment_description_prompt(segment_index, start_time, end_time)
    last_error = None

    for attempt in range(1, 3):
        raw_response = call_nvidia_segment_api(video_data_url, prompt)
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        (DEBUG_DIR / f"segment_{segment_index:03d}_last_response.txt").write_text(raw_response, encoding="utf-8")

        try:
            json_text = extract_json_text(raw_response)
            break
        except ValueError as exc:
            last_error = exc
            raw_path = DEBUG_DIR / f"segment_{segment_index:03d}_raw_response_attempt_{attempt}.txt"
            raw_path.write_text(raw_response, encoding="utf-8")
            prompt = (
                "你刚才的输出不是完整合法 JSON。请重新分析同一个视频片段，"
                "只输出一个完整、紧凑、合法的 JSON 对象，不要 Markdown。"
            ) + build_segment_description_prompt(segment_index, start_time, end_time)
    else:
        raise ValueError(f"第 {segment_index} 段模型返回内容不是合法 JSON：{last_error}")

    parsed = json.loads(json_text)
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError(f"第 {segment_index} 段模型返回的 JSON 不是对象。")
    parsed["segment_index"] = segment_index
    parsed["start_time"] = start_time
    parsed["end_time"] = end_time
    if isinstance(parsed.get("actions"), str):
        parsed["actions"] = [
            item.strip()
            for item in re.split(r"[，,、;；]", parsed["actions"])
            if item.strip()
        ]
    parsed["proxy_path"] = str(proxy_path)
    return parsed


def analyze_video_segments(
    video_path: str | Path,
    segment_seconds: int = SEGMENT_SECONDS,
    max_segments: int | None = None,
    progress=None,
) -> list[dict]:
    check_ffmpeg()
    if not os.getenv("NVIDIA_API_KEY"):
        raise RuntimeError("未设置 NVIDIA_API_KEY 环境变量。")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    SEGMENT_PROXY_DIR.mkdir(parents=True, exist_ok=True)

    duration = get_video_duration(video_path)
    total_segments = math.ceil(duration / segment_seconds)
    if max_segments is not None:
        total_segments = min(total_segments, max_segments)

    results = []
    for zero_index in range(total_segments):
        segment_index = zero_index + 1
        start_seconds = zero_index * segment_seconds
        actual_duration = min(segment_seconds, max(0.1, duration - start_seconds))
        end_seconds = min(duration, start_seconds + actual_duration)
        start_time = format_time(start_seconds)
        end_time = format_time(end_seconds)

        if progress:
            progress(f"正在转码第 {segment_index}/{total_segments} 段：{start_time} - {end_time}")
        proxy_path = make_segment_proxy_video(video_path, segment_index, start_seconds, actual_duration)

        if progress:
            size_mb = proxy_path.stat().st_size / (1024 * 1024)
            progress(f"正在询问 NVIDIA：第 {segment_index}/{total_segments} 段，proxy {size_mb:.2f}MB")
        result = describe_segment(proxy_path, segment_index, start_time, end_time)
        results.append(result)

        output_path = RESULTS_DIR / "segment_descriptions.json"
        output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        if progress:
            progress(f"第 {segment_index}/{total_segments} 段完成")

    return results


class SegmentDescriberApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NVIDIA NIM 视频分段内容描述")
        self.root.geometry("920x680")
        self.video_path = tk.StringVar(value="未选择")
        self.status = tk.StringVar(value="就绪")
        self.output_path = tk.StringVar(value=str(RESULTS_DIR / "segment_descriptions.json"))
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=14, pady=14)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Button(frame, text="选择视频文件", command=self.choose_video).grid(row=0, column=0, sticky="w")
        self.start_button = tk.Button(frame, text="开始分段描述", command=self.start_analysis, state=tk.DISABLED)
        self.start_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

        tk.Label(frame, text="已选择的视频路径：").grid(row=1, column=0, sticky="nw", pady=(14, 0))
        tk.Label(frame, textvariable=self.video_path, anchor="w", justify="left", wraplength=700).grid(
            row=1, column=1, sticky="we", pady=(14, 0)
        )

        tk.Label(frame, text="结果保存路径：").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.output_path, anchor="w", justify="left", wraplength=700).grid(
            row=2, column=1, sticky="we", pady=(10, 0)
        )

        tk.Label(frame, text="当前处理状态：").grid(row=3, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.status, anchor="w", justify="left", wraplength=700).grid(
            row=3, column=1, sticky="we", pady=(10, 0)
        )

        tk.Label(frame, text="分段 JSON 结果：").grid(row=4, column=0, columnspan=2, sticky="w", pady=(18, 6))
        self.output = scrolledtext.ScrolledText(frame, height=26, wrap=tk.WORD)
        self.output.grid(row=5, column=0, columnspan=2, sticky="nsew")

        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(5, weight=1)

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
        self.status.set("已选择视频")
        self.output.delete("1.0", tk.END)
        self.start_button.config(state=tk.NORMAL)

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

        self.start_button.config(state=tk.DISABLED)
        self.output.delete("1.0", tk.END)
        self.status.set("开始处理")
        thread = threading.Thread(target=self._run_analysis, args=(selected,), daemon=True)
        thread.start()

    def _run_analysis(self, selected: str) -> None:
        try:
            def progress(message: str) -> None:
                self.events.put(("progress", message))

            results = analyze_video_segments(selected, progress=progress)
            self.events.put(("success", json.dumps(results, ensure_ascii=False, indent=2)))
        except Exception as exc:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            (DEBUG_DIR / "segment_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
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
            self.status.set("完成，结果已保存到 results/segment_descriptions.json")
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, payload)
            messagebox.showinfo("完成", "分段描述完成，结果已保存。")
            self.start_button.config(state=tk.NORMAL)
        else:
            self.status.set("失败")
            self.output.insert(tk.END, "\n" + payload)
            messagebox.showerror("处理失败", payload)
            self.start_button.config(state=tk.NORMAL)

        self.root.after(100, self._poll_events)


def run_gui() -> None:
    if tk is None:
        raise RuntimeError("当前 Python 环境不可用 tkinter。请安装/切换到带 tkinter 支持的 Python 3。")
    root = tk.Tk()
    SegmentDescriberApp(root)
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a video into 60s proxy clips and describe each with NVIDIA NIM.")
    parser.add_argument("--video", help="Run without GUI and analyze this video path.")
    parser.add_argument("--max-segments", type=int, help="Only analyze the first N segments in CLI mode.")
    args = parser.parse_args()

    if args.video:
        results = analyze_video_segments(args.video, max_segments=args.max_segments, progress=print)
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        run_gui()


if __name__ == "__main__":
    main()
