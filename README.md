# Local Transcription Workbench

一个本地优先的音视频转写工作台：上传文件或粘贴媒体链接，输出可复现素材包。

## 能力

- 本地文件与远程媒体链接转写
- 纯本地链路：`curl / yt-dlp + ffmpeg + whisper-cli`
- 默认保留 `source.*`、`audio.wav`、`transcript.txt/json/srt/vtt`
- 提供轻量 HTML 工作台

## 运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

打开：`http://127.0.0.1:8765`

> 需要本机已安装 `ffmpeg`、`whisper-cli`；如输入网页媒体链接，还需要 `yt-dlp`。

如果你已经有 Whisper 模型，可以直接复用，不必重新下载：

```bash
export TRANSCRIBE_MODEL_DIR=/path/to/your/whisper-models
```

程序会优先寻找显式模型路径、`TRANSCRIBE_MODEL_DIR` / `WHISPER_MODEL_DIR`，再看项目本地模型目录；都没有时才下载。

## 目录产物

```text
run_xxx/
├── manifest.json
├── run_config.json
└── items/
    └── 001_<source_label>/
        ├── metadata.json
        ├── source.<ext>
        ├── audio.wav
        ├── transcript.txt
        ├── transcript.json
        ├── transcript.srt
        └── transcript.vtt
```
