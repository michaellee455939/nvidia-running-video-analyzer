# NVIDIA Running Video Analyzer

Python tkinter MVP for analyzing whether a local video contains running, chasing, sprinting, escaping, or sports-running scenes with NVIDIA NIM / NVIDIA Build OpenAI-compatible API.

## Setup

```bash
cd /Users/mac/nvidia-running-video-analyzer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export NVIDIA_API_KEY="your_api_key"
python app.py
```

You also need `ffmpeg` installed and available in `PATH`.

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

The extractor resumes automatically per video and keyword set. If a previous run stopped early, rerun the same command and completed or previously failed segments will be skipped. In the GUI, enable `重新从头处理` to ignore previous state and delete old clips/json for that video and keyword set before processing. Enable `重试失败段` to retry segments listed in `failed_segments.json`. Use `停止处理` to stop after the current ffmpeg/API call finishes safely.

In CLI mode, use `--fresh` to restart or `--retry-failed` to retry failed segments.

CLI test mode:

```bash
python running_clip_extractor.py --video "/path/to/video.mp4" --max-segments 12
python running_clip_extractor.py --video "/path/to/video.mp4" --keywords "打架、追逐" --max-segments 12
python running_clip_extractor.py --video "/path/to/video.mp4" --fresh
python running_clip_extractor.py --video "/path/to/video.mp4" --retry-failed
```
