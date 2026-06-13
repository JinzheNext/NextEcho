# SPEC：主流媒体链接转录 + 轻量 Mac App MVP

## 目标

- 支持 `YouTube`、`B 站`、`小宇宙` 链接进入统一的本地转录链路。
- 输出统一进入 `outputs/transcriptions/run_xxx`，并保留来源元数据。
- 提供一个可双击打开的轻量 Mac App 外壳，承载本地工作台。

## 当前实现边界

- `workbench/sources.py`：负责平台识别、canonical URL 清理、页面元数据解析。
- `workbench/cli.py`：
  - `resolve-sources`
  - `transcribe-page`
  - `transcribe`
  - `doctor`
  - `serve`
  - `speaker-transcript`
- `workbench/transcription.py`：继续负责媒体下载、抽音频、whisper.cpp 转写、结果落盘。
- `app.py` + `templates/index.html` + `static/*`：本地 UI，支持先识别链接再启动任务。
- `scripts/build_mac_app.sh`：生成 `dist/NextEcho.app`。

## Agent 执行顺序

1. 跑基础检查：
   - `python3 -m workbench.cli doctor`
   - `python3 -m unittest discover -s tests -q`
2. 验证解析层：
   - `python3 -m workbench.cli resolve-sources "https://www.youtube.com/watch?v=96jN2OCOfLs" --json`
   - `python3 -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/6a19390a7460cabdeb57c0e5" --json`
   - `python3 -m workbench.cli resolve-sources "https://www.bilibili.com/video/BV1g6okBLEtL/" --json`
3. 验证单链接转写：
   - `python3 -m workbench.cli transcribe-page "<url>" --quality fast --json`
4. 验证工作台：
   - `python3 -m workbench.cli serve`
   - 打开 `http://127.0.0.1:8765`
5. 构建 Mac App：
   - `bash scripts/build_mac_app.sh`
   - 双击 `dist/NextEcho.app`

## 产物契约

每个 `item` 至少保留以下来源字段：

- `platform`
- `title`
- `author`
- `canonical_url`
- `resolved_media_url`
- `resolver`
- `duration_seconds`
- `published_at`
- `error`

失败项不应静默丢失，必须写入 `manifest.json`，并在有失败时写出 `errors.json`。

## 下一步建议

- 为 `yt-dlp` 增加 cookies 能力，但先不做小红书。
- 为小宇宙补更稳的 RSS / enclosure 匹配。
- 为 App 增加图标、签名、公证和自动更新。
