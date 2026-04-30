from __future__ import annotations

import base64
import json
import mimetypes
import os
import queue
import re
import shutil
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


APP_DIR = Path(__file__).resolve().parent
PROXY_DIR = APP_DIR / "proxies"
RESULTS_DIR = APP_DIR / "results"
DEBUG_DIR = APP_DIR / "debug"
TARGET_BYTES = 10 * 1024 * 1024
MODEL_NAME = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"


def check_ffmpeg() -> str:
    """Return the ffmpeg executable path, or raise RuntimeError."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("未找到 ffmpeg。请先安装 ffmpeg，并确保它在 PATH 中。")
    return ffmpeg_path


def make_proxy_video(video_path: str, output_dir: Path = PROXY_DIR, target_bytes: int = TARGET_BYTES) -> Path:
    """Create an MP4 proxy video, retrying lower resolutions/bitrates until under target size."""
    ffmpeg = check_ffmpeg()
    input_path = Path(video_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"视频文件不存在：{input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_proxy.mp4"

    attempts = [
        {"height": 480, "video_bitrate": "900k", "audio_bitrate": "64k"},
        {"height": 360, "video_bitrate": "650k", "audio_bitrate": "48k"},
        {"height": 360, "video_bitrate": "450k", "audio_bitrate": "32k"},
        {"height": 240, "video_bitrate": "300k", "audio_bitrate": "32k"},
        {"height": 240, "video_bitrate": "180k", "audio_bitrate": "24k"},
    ]

    last_error = None
    for idx, attempt in enumerate(attempts, start=1):
        candidate = output_dir / f"{input_path.stem}_proxy_{idx}.mp4"
        scale_filter = f"scale=-2:{attempt['height']}:force_original_aspect_ratio=decrease"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-dn",
            "-sn",
            "-vf",
            scale_filter,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
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
            "-c:a",
            "aac",
            "-b:a",
            attempt["audio_bitrate"],
            str(candidate),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            last_error = result.stderr.strip() or result.stdout.strip()
            continue

        if candidate.stat().st_size <= target_bytes:
            if output_path.exists():
                output_path.unlink()
            candidate.rename(output_path)
            return output_path

        if not output_path.exists() or candidate.stat().st_size < output_path.stat().st_size:
            if output_path.exists():
                output_path.unlink()
            candidate.rename(output_path)
        else:
            candidate.unlink(missing_ok=True)

    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        raise RuntimeError(f"已生成 proxy，但仍大于 10MB（{size_mb:.2f}MB）。请先裁剪视频或降低时长后重试。")
    raise RuntimeError(f"ffmpeg 生成 proxy 失败：{last_error or '未知错误'}")


def read_b64(file_path: str | Path) -> str:
    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def build_running_prompt() -> str:
    return (
        "你是视频内容分析器。请分析视频中是否出现奔跑、追逐、快跑、冲刺、逃跑、运动奔跑等画面。"
        "只返回合法 JSON，不要解释，不要 Markdown，不要代码块。"
        "如果没有奔跑画面，返回 []。"
        "如果有，返回 JSON 数组，每个元素包含："
        "start_time（字符串，格式 HH:MM:SS 或 MM:SS）、"
        "end_time（字符串，格式 HH:MM:SS 或 MM:SS）、"
        "confidence（0 到 1 的数字）、"
        "description（中文简短描述）、"
        "labels（字符串数组，例如 [\"奔跑\", \"追逐\"]）。"
        "只根据视频画面和音频判断，不要臆测。"
    )


def call_nvidia_api(video_data_url: str, prompt: str) -> str:
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
        max_tokens=4096,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 4096,
            "mm_processor_kwargs": {"use_audio_in_video": True},
        },
        stream=True,
    )

    chunks = []
    for chunk in completion:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            chunks.append(content)

    raw_text = "".join(chunks).strip()
    if not raw_text:
        raise RuntimeError("NVIDIA API 返回为空，未收到 delta.content。")
    return raw_text


def extract_json_text(raw_text: str) -> str:
    text = raw_text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        pass

    array_start = text.find("[")
    array_end = text.rfind("]")
    object_start = text.find("{")
    object_end = text.rfind("}")

    candidates = []
    if array_start != -1 and array_end != -1 and array_end > array_start:
        candidates.append(text[array_start : array_end + 1])
    if object_start != -1 and object_end != -1 and object_end > object_start:
        candidates.append(text[object_start : object_end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            continue

    raise ValueError("模型返回内容不是合法 JSON。")


def analyze_video(video_path: str) -> tuple[Path, str]:
    PROXY_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    proxy_path = make_proxy_video(video_path)
    video_b64 = read_b64(proxy_path)
    mime_type = mimetypes.guess_type(proxy_path)[0] or "video/mp4"
    video_data_url = f"data:{mime_type};base64,{video_b64}"

    raw_response = call_nvidia_api(video_data_url, build_running_prompt())
    try:
        json_text = extract_json_text(raw_response)
    except ValueError:
        (DEBUG_DIR / "raw_response.txt").write_text(raw_response, encoding="utf-8")
        raise

    (RESULTS_DIR / "running_segments.json").write_text(json_text, encoding="utf-8")
    return proxy_path, json_text


class RunningAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("NVIDIA NIM 视频奔跑分析")
        self.root.geometry("860x640")
        self.video_path = tk.StringVar(value="未选择")
        self.proxy_path = tk.StringVar(value="未生成")
        self.status = tk.StringVar(value="就绪")
        self.events: queue.Queue[tuple[str, str]] = queue.Queue()

        self._build_ui()
        self.root.after(100, self._poll_events)

    def _build_ui(self) -> None:
        frame = tk.Frame(self.root, padx=14, pady=14)
        frame.pack(fill=tk.BOTH, expand=True)

        tk.Button(frame, text="选择视频文件", command=self.choose_video).grid(row=0, column=0, sticky="w")
        self.analyze_button = tk.Button(frame, text="开始分析", command=self.start_analysis, state=tk.DISABLED)
        self.analyze_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

        tk.Label(frame, text="已选择的视频路径：").grid(row=1, column=0, sticky="nw", pady=(14, 0))
        tk.Label(frame, textvariable=self.video_path, anchor="w", justify="left", wraplength=650).grid(
            row=1, column=1, sticky="we", pady=(14, 0)
        )

        tk.Label(frame, text="proxy 视频路径：").grid(row=2, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.proxy_path, anchor="w", justify="left", wraplength=650).grid(
            row=2, column=1, sticky="we", pady=(10, 0)
        )

        tk.Label(frame, text="当前处理状态：").grid(row=3, column=0, sticky="nw", pady=(10, 0))
        tk.Label(frame, textvariable=self.status, anchor="w", justify="left", wraplength=650).grid(
            row=3, column=1, sticky="we", pady=(10, 0)
        )

        tk.Label(frame, text="模型返回的 JSON 文本：").grid(row=4, column=0, columnspan=2, sticky="w", pady=(18, 6))
        self.output = scrolledtext.ScrolledText(frame, height=24, wrap=tk.WORD)
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
        self.proxy_path.set("未生成")
        self.status.set("已选择视频")
        self.output.delete("1.0", tk.END)
        self.analyze_button.config(state=tk.NORMAL)

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

        self.analyze_button.config(state=tk.DISABLED)
        self.status.set("正在压缩视频并调用 NVIDIA API...")
        self.output.delete("1.0", tk.END)
        thread = threading.Thread(target=self._run_analysis, args=(selected,), daemon=True)
        thread.start()

    def _run_analysis(self, selected: str) -> None:
        try:
            proxy_path, json_text = analyze_video(selected)
            self.events.put(("success", json.dumps({"proxy_path": str(proxy_path), "json_text": json_text})))
        except Exception as exc:
            DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            (DEBUG_DIR / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            self.events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            event, payload = self.events.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_events)
            return

        if event == "success":
            data = json.loads(payload)
            self.proxy_path.set(data["proxy_path"])
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, data["json_text"])
            self.status.set("完成，结果已保存到 results/running_segments.json")
            messagebox.showinfo("完成", "分析完成，结果已保存。")
        else:
            self.status.set("失败")
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, payload)
            messagebox.showerror("分析失败", payload)

        self.analyze_button.config(state=tk.NORMAL)
        self.root.after(100, self._poll_events)


def main() -> None:
    if tk is None:
        raise RuntimeError("当前 Python 环境不可用 tkinter。请安装/切换到带 tkinter 支持的 Python 3。")
    root = tk.Tk()
    RunningAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
