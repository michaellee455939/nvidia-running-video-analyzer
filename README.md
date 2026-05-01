# NVIDIA Running Video Analyzer

Python tkinter MVP for analyzing whether a local video contains running, chasing, sprinting, escaping, or sports-running scenes with NVIDIA NIM / NVIDIA Build OpenAI-compatible API.

## Setup

macOS/Linux:

```bash
cd /Users/mac/nvidia-running-video-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY="your_api_key"
python app.py
```

You also need `ffmpeg` installed and available in `PATH`.

Windows:

1. Install Python 3 from [python.org](https://www.python.org/downloads/windows/). Enable `Add python.exe to PATH` during installation.
2. Install ffmpeg:

```bat
winget install Gyan.FFmpeg
```

Close and reopen PowerShell or Command Prompt after installing ffmpeg.

3. Download or clone this project, then run:

```bat
setup_windows.bat
```

4. Set your NVIDIA API key. For the current terminal only:

```bat
set NVIDIA_API_KEY=your_api_key
```

Or permanently:

```bat
setx NVIDIA_API_KEY "your_api_key"
```

After `setx`, close and reopen the terminal.

5. Start the keyword clip extractor GUI:

```bat
run_keyword_extractor_windows.bat
```

You can also run directly:

```bat
.venv\Scripts\python.exe running_clip_extractor.py
```

## Outputs

- Proxy video: `proxies/`
- Parsed JSON result: `results/running_segments.json`
- Raw model response on invalid JSON: `debug/raw_response.txt`

If no running scene is found, the model is instructed to return:

```json
[]
```

## Segment description GUI

To split a video every 60 seconds, compress each segment below 10MB, and ask NVIDIA to describe every segment:

```bash
cd /Users/mac/nvidia-running-video-analyzer
source /Users/mac/.zshrc
source .venv/bin/activate
python segment_describer.py
```

Outputs:

- Segment proxy videos: `segment_proxies/`
- Segment descriptions: `results/segment_descriptions.json`
- Raw invalid segment responses: `debug/segment_XXX_raw_response.txt`
- Segment errors: `debug/segment_error.txt`

CLI test mode:

```bash
python segment_describer.py --video "/path/to/video.mp4" --max-segments 1
```

## Keyword clip extraction

To detect target scenes/actions in 60-second proxy segments and cut those moments from the original video without re-encoding:

```bash
cd /Users/mac/nvidia-running-video-analyzer
source /Users/mac/.zshrc
source .venv/bin/activate
python running_clip_extractor.py
```

Outputs:

- Original-quality clips: a folder next to the source video named `<source_video_name>_<keywords>_clips/`
- JSON index: `<source_video_name>_<keywords>_clips/matched_clips.json`
- Resume state: `<source_video_name>_<keywords>_clips/resume_state.json`
- Failed segments: `<source_video_name>_<keywords>_clips/failed_segments.json`
- Run log: `<source_video_name>_<keywords>_clips/run.log`
- Latest-run JSON copy: `results/matched_clips.json`
- Debug responses and errors: `debug/`

The GUI has a `筛查关键字` input. The default is running-related keywords, but you can enter targets such as `打架、摔倒、抽烟、开枪、跳舞`.

### Windows Large Video Preprocessing

The GUI includes `Windows 大视频预处理模式`. This is a manual Windows-only optimization for very large or high-spec sources such as 60GB+ files, 4K, HEVC Main10, HDR/Dolby Vision, or high-bitrate Remux files.

Use `默认模式（普通视频推荐，保持原逻辑）` for ordinary videos such as small clips, normal 1080p/H.264 files, and short social media material. Default mode keeps the original proxy generation behavior and does not force h264_nvenc, low resolution, no audio, or thread limits.

For large Windows videos:

- `平衡模式（大视频推荐，约80% CPU）`: recommended for most large-video processing.
- `高速模式（CPU高，速度最快）`: use when the computer is idle or overnight.
- `温和模式（约60% CPU）`: use while doing other work on the PC.
- `低CPU模式（最慢，最低占用）`: use when the computer is prone to stalling.

Large-video modes use NVIDIA `h264_nvenc` for proxy encoding when ffmpeg supports it, with 8-bit-compatible proxy output:

```text
-vf scale=-2:240:flags=fast_bilinear,format=yuv420p
-c:v h264_nvenc
-preset fast
-b:v 800k
-profile:v main
-pix_fmt yuv420p
-an
```

If `h264_nvenc` is unavailable or fails, the tool automatically falls back to CPU libx264 low-thread proxy generation and logs:

```text
h264_nvenc 不可用或失败，已回退到 CPU 低线程模式
```

On older GPUs such as GTX970, NVENC only helps with H.264 proxy encoding. 4K HEVC Main10 decode may still rely heavily on CPU, so use `平衡模式`, `温和模式`, or `低CPU模式` when the PC becomes sluggish. This large-video preprocessing mode is not smart/dynamic CPU adjustment; it is a manual mode selector.

The extractor resumes automatically per video and keyword set. If a previous run stopped early, rerun the same command and completed or previously failed segments will be skipped. In the GUI, enable `重新从头处理` to ignore previous state and delete old clips/json for that video and keyword set before processing. Enable `重试失败段` to retry segments listed in `failed_segments.json`. Use `停止处理` to stop after the current ffmpeg/API call finishes safely.

In CLI mode, use `--fresh` to restart or `--retry-failed` to retry failed segments. For Windows large-video preprocessing, use `--windows-large-video-mode balanced`, `fast`, `gentle`, or `low_cpu`. The default is `default`, which keeps ordinary-video behavior.

CLI test mode:

```bash
python running_clip_extractor.py --video "/path/to/video.mp4" --max-segments 12
python running_clip_extractor.py --video "/path/to/video.mp4" --keywords "打架、追逐" --max-segments 12
python running_clip_extractor.py --video "/path/to/video.mp4" --fresh
python running_clip_extractor.py --video "/path/to/video.mp4" --retry-failed
python running_clip_extractor.py --video "/path/to/video.mp4" --windows-large-video-mode balanced
```

Windows CLI examples:

```bat
.venv\Scripts\python.exe running_clip_extractor.py --video "C:\Users\you\Videos\movie.mp4" --keywords "跑步,追逐"
.venv\Scripts\python.exe running_clip_extractor.py --video "C:\Users\you\Videos\movie.mp4" --fresh
.venv\Scripts\python.exe running_clip_extractor.py --video "C:\Users\you\Videos\movie.mp4" --retry-failed
.venv\Scripts\python.exe running_clip_extractor.py --video "C:\Users\you\Videos\movie.mp4" --windows-large-video-mode balanced
.venv\Scripts\python.exe running_clip_extractor.py --video "C:\Users\you\Videos\movie.mp4" --windows-large-video-mode gentle
```
