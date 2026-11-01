# NextEcho

> 让播客自动转录进知识库

[English README](README.en.md) | [开源合规说明](OPEN_SOURCE_COMPLIANCE.md) | [第三方许可证清单](THIRD_PARTY_LICENSES.md)

NextEcho 是一个本地优先的音视频转写工作台。你可以上传本地文件，或者粘贴 B 站、小宇宙、Apple Podcasts、播客 RSS 等链接，把内容转成可复用的文字资产，并落盘为结构化素材包。

它同时适合两类用户：

- 人类用户：直接用网页界面操作
- Agent：安装 Skill，发送链接，直接收到本地结果文件路径

![NextEcho Web View](NextEcho-webview.png)

## 直接交给 Agent：安装 Skill，再发链接

NextEcho **0.2.12** 提供可独立安装的 [Agent Skill](skills/nextecho/SKILL.md)，支持在本机部署并复用已有模型配置。完整安装包同时包含程序、对应源码和安装工具，安装步骤与运行要求见下文。

一次准备好运行时与本地模型后，可以直接对Agent说：

> 用 NextEcho 转录这个播客链接，润色并区分说话人，最后只给我结果文件路径。

Agent执行本地程序，完成识别、声学分人、保真润色和文件交付。默认不生成摘要；无需打开网站，也不需要Agent逐段阅读长篇转录。正常完成返回“栏目_单集标题_转录日期.md”的绝对路径；部分完成返回可用路径及简短原因，详细诊断留在独立回执中。Agent编排仍可能消耗少量token，本地识别/润色的计算与宿主Agent用量分别记录。

### 下载安装包

[v0.2.12 预览版下载页](https://github.com/JinzheNext/NextEcho/releases/tag/v0.2.12)

- [完整 Agent 安装包（推荐）](https://github.com/JinzheNext/NextEcho/releases/download/v0.2.12/NextEcho-0.2.12-agent-bundle.zip)：Skill、程序、对应源码与安装器。
- [仅下载 Skill](https://github.com/JinzheNext/NextEcho/releases/download/v0.2.12/nextecho-0.2.12-skill.zip)：适合已有兼容运行时的机器。
- [文件校验清单](https://github.com/JinzheNext/NextEcho/releases/download/v0.2.12/SHA256SUMS.txt)。

这是软件与Skill的预览发布；模型权重、系统依赖和独立worker需另行准备。模型候选状态及人工审核要求保持独立。

### 安装包与首次安装

| 文件 | 用途 |
| --- | --- |
| `NextEcho-0.2.12-agent-bundle.zip` | 完整Agent安装包：Skill、程序wheel、对应源码、安装器与SHA-256清单 |
| `nextecho-0.2.12-skill.zip` | 纯Skill；适合已安装NextEcho运行时的机器 |
| `nextecho-0.2.12-py3-none-any.whl` | Python程序包；不包含模型或Python解释器 |

使用macOS/Linux与Python≥3.11，解压完整Agent包后执行：

```bash
python3.11 install.py --check
python3.11 install.py
```

安装器创建独立程序环境，安装Skill并保存外部绑定。首次安装需要联网获取核心Python依赖；已有Skill/绑定时默认拒绝覆盖，显式`--replace`会备份并继承配置。模型、录音、cookies与本机绑定不会进入安装包，也不会在安装时自动下载模型或启动推理。安装完成后仍需绑定模型worker并运行doctor；**Skill安装完成不等于模型已准备好**。详细步骤、其他Agent目录、自定义数据路径及回滚见[Agent安装包说明](docs/agent-bundle.md)。

在源码目录构建本地安装包：

```bash
uv build --out-dir dist
python3.11 scripts/package_agent_bundle.py \
  --wheel dist/nextecho-0.2.12-py3-none-any.whl \
  --sdist dist/nextecho-0.2.12.tar.gz
```

### 模型与交付范围

当前本地链路接入SenseVoice int8、Fun-ASR-Nano CPU，以及本地2B/4B/9B润色档案；候选需要显式选择、绑定独立模型环境并通过资源检查。27B和35B MoE仍留待单独验证，不会随安装器启动。模型可运行、资源校准通过与产品质量放行分别记录，不能据参数量或安装成功推断准确率。

已明确选择的个人试用组合可保存到本机`agent.default_profile`和`agent.approved_trial_models`配置，之后只发链接即可调用，无需每次补`--profile`。该许可只对列出的模型ID生效，不修改产品质量状态，也不跳过文件校验或资源护栏，示例见[安装与绑定](skills/nextecho/references/install.md)。

Skill默认尝试声学分人与CTC对话时间锚。姓名需声音证据或用户确认；仅有节目标题不能把匿名声线直接命名，确认也只作用于对应录音。分人与精确字幕仍需分别验收。

交付稿包含来源与shownotes摘录、完整官方章节标题链接，以及实际捕获的公开评论和回复原文。目录使用Obsidian兼容标题链接；评论置于文末并提供跳转入口，不声称已抓全平台评论。阅读层连接明确的断词、清理无信息短停顿和机械重复，原稿、字幕及声学ID保持不变。项目署名为：

> NextEcho，让播客自动本地转录进知识库
>
> @金哲Next，一起用好AI走向Next Level，欢迎关注小红书&公众号～

支持持久请求、后台等待/取消、exact/refresh恢复、按句润色补修和只重做分人。原稿与阅读编辑审计分别落盘。命令参考见[无网页使用说明](docs/headless-usage.md)；中文专名与上下文校订见[中文上下文校订](docs/headless-usage.md#中文上下文校订显式试用)。

## 这是什么

NextEcho 适合做这些事：

- 把播客、视频、访谈快速转成全文
- 把一期节目沉淀进你的知识库或笔记系统
- 保留源文件、音频、中间产物和字幕，方便后续复用
- 让 Claude Code、Codex、Cursor Agent 之类工具直接调用本地转录能力

核心特点：

- 本地优先，不依赖云端 ASR
- 支持本地文件和远程链接
- 支持网页端和命令行
- 默认保留可读命名的 `录音原声.*`、`转录音频.wav`、`transcript.txt/json/srt/vtt`；旧版 `source.*`/`audio.wav` 仍可读取
- 小宇宙链接会保存shownotes、原始页面、公开页内点赞排序的前20条主评论及可见回复预览；评论只作术语线索，不作为声音身份或发言事实的证据
- 播客页面可自动生成带时间戳的说话人稿，并把节目页提供的主持人/嘉宾姓名作为可复核提示
- 转录文本默认带项目署名，便于传播时保留来源

## 兼容网页与旧 CLI：本地转录和知识加工

以下保留原网页及旧CLI的模型配置，和上面的Skill候选档案分别使用；新Agent安装器不会自动执行下面的模型下载命令。

在 Apple Silicon Mac 上，NextEcho 可选用 `Qwen3-ASR` 作为本地识别引擎：高精度档使用 1.7B，快速档使用 0.6B；模型运行在独立进程，结束后释放 Metal 内存。旧流程保留 whisper.cpp 兼容路径；是否允许回退由入口与配置决定，实际引擎及原因写入结果。长音频会分段处理，不因时长自动降到 `base`。

转录后可以选择完全本地的“智能加工”：

- 统一 4B（默认）：`qwen3:4b` 同时整理逐字稿和汇总知识稿，避免模型切换；
- 混合均衡：`qwen3:4b` 整理逐字稿，`qwen3.5:9b-q4_K_M` 汇总知识稿；
- 高质量：全程使用 9B（9B 不可用时自动回退 4B，并记录原因）；
- 快速：全程使用已安装的 4B；
- 关闭：只生成原始转录。

加工过程始终保留原始 `transcript.*`，另外生成 `transcript.enhanced.*`、`transcript.summary.json`、`transcript.knowledge.md` 和 `postprocess.audit.json`。知识稿含有可追溯的片段 ID 与时间戳，适合直接导入 Obsidian。

Apple Silicon 的安装脚本会安装可选的 MLX 运行时；模型权重和 Ollama 的 9B 模型不会被静默下载。请在网页模型状态入口确认后下载，或先执行：

```bash
ollama pull qwen3:4b
ollama pull qwen3.5:9b-q4_K_M
python -m workbench.cli doctor
```

对历史转录重新生成整理稿和知识稿，无需重新下载媒体或执行 ASR：

```bash
python -m workbench.cli enhance outputs/transcriptions/run_xxx --profile fast
```

## 支持什么输入

- 本地文件：`mp3`、`mp4`、`m4a`、`wav`、`flac`、`aac`、`mov`、`webm`
- 平台页面：B 站、小宇宙、Apple Podcasts
- 直链媒体：音频或视频文件 URL
- RSS / 播客 feed

## 推荐用法

如果你是普通用户，推荐只记住这两步，不需要打开终端：

### macOS

1. 双击 `Install NextEcho.command`
2. 安装完成后，双击 `Open NextEcho.command` 或 `dist/NextEcho.app`

### Windows

1. 双击 `Install NextEcho.bat`
2. 安装完成后，双击 `Open NextEcho.bat`

这一套一键脚本会尽量自动完成这些事情：

- 创建 Python 虚拟环境
- 安装 Python 依赖
- 检查并尽量自动安装 `ffmpeg`、`whisper-cli`
- 预下载默认可用模型 `base`
- 自动启动本地网站
- 自动打开浏览器到 `http://127.0.0.1:8765`

## 先安装

### macOS

推荐直接双击：

- `Install NextEcho.command`

它背后实际调用的是：

```bash
bash scripts/install_mac.sh
```

这个脚本会：

- 检查是否安装了 Python 3
- 自动创建 `.venv`
- 安装 `requirements.txt`
- 检查 `ffmpeg` 和 `whisper-cli`
- 如果检测到 Homebrew，自动补装缺失依赖
- 预下载 `base` 模型
- 启动本地网站并自动打开浏览器

如果你想手动安装，最少需要这些依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg whisper-cpp yt-dlp
python -m workbench.cli doctor
```

### Windows PowerShell

推荐直接双击：

- `Install NextEcho.bat`

它背后调用的是：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
```

如果手动安装，需要确保：

- 已安装 Python 3.11+
- `ffmpeg` 在 PATH 中
- `whisper-cli` 在 PATH 中
- 如果要解析平台页面，建议安装 `yt-dlp`

然后执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m workbench.cli doctor
```

### 安装完成后先自检

```bash
python -m workbench.cli doctor
```

看到这些就说明主链路基本可用：

- `ffmpeg` 已找到
- `whisper-cli` 已找到
- 至少有一个 Whisper 模型可用，或者允许首次下载

### 模型怎么选

如果你想根据自己电脑的性能手动挑选模型，而不是直接使用默认值，推荐按下面顺序做：

1. 先看当前机器建议：

```bash
python -m workbench.cli doctor
python -m workbench.cli list-models
```

2. 再决定要下载哪个模型：

- `tiny`：最省资源，适合老机器或只想快速预览
- `base`：轻量稳定，适合 8GB 左右机器
- `small`：精度和速度更平衡，适合 8GB 到 16GB 机器
- `medium`：更高精度，适合 16GB 以上机器
- `large-v3-turbo-q5_0`：默认高精度模型，适合性能更强的机器

3. 明确下载你要的模型：

```bash
python -m workbench.cli download-model base
python -m workbench.cli download-model small
python -m workbench.cli download-model large-v3-turbo-q5_0
```

4. 运行转录时，你可以显式指定模型，而不只依赖 `--quality`：

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --model base --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --model small --json
```

## 给人类用户怎么用

### 方式一：直接用网页

这是最适合大多数人的方式。

1. 双击 `Open NextEcho.command`（macOS）或 `Open NextEcho.bat`（Windows）
2. 浏览器会自动打开 `http://127.0.0.1:8765`

3. 在网页里任选一种输入方式：

- 上传本地音视频文件
- 粘贴一个或多个链接，每行一个

4. 选择质量：

- `高精度`：默认推荐，优先使用 `large-v3-turbo-q5_0`
- `更快`：速度优先，使用 `base`

5. 点击开始转录，等待生成结果。

网页端适合这些场景：

- 你只是想快速转一期节目
- 你不想碰命令行参数
- 你想直观看到最近的转录记录和产物路径

### 方式二：双击打开 Mac App

如果你在 macOS 上，希望像普通 App 一样启动：

```bash
bash scripts/build_mac_app.sh
```

生成后会得到：

```text
dist/NextEcho.app
```

双击后它会尝试：

- 检查本地环境
- 如果缺依赖，自动调用安装脚本
- 启动服务
- 自动打开 `http://127.0.0.1:8765`

日志写入：

```text
logs/app.log
```

### 方式三：直接用命令行

适合想要批量跑、写脚本、或者把转录结果接进自己工作流的人。

最常用的命令如下。

#### 1. 转录本地文件

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --quality accurate --json
python -m workbench.cli transcribe /path/to/audio.mp3 --model base --json
python -m workbench.cli transcribe /path/to/audio.mp3 --language zh --asr-engine auto --postprocess-profile fast --json
```

#### 2. 转录直链媒体

```bash
python -m workbench.cli transcribe "https://example.com/video.mp4" --quality fast
```

#### 3. 先识别来源，再决定是否转录

```bash
python -m workbench.cli resolve-sources "https://www.bilibili.com/video/BV1g6okBLEtL/" --json
python -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --json
python -m workbench.cli resolve-sources "https://podcasts.apple.com/us/podcast/the-daily/id1200361736" --json
```

#### 4. 直接转录平台页面

```bash
python -m workbench.cli transcribe-page "https://www.bilibili.com/video/BV1g6okBLEtL/" --quality fast --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --quality accurate --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --quality accurate --speaker-mode auto --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --model small --json
```

#### 5. 转录播客 RSS

```bash
python -m workbench.cli transcribe-feed "https://example.com/feed.xml" --limit 3 --quality fast --json
```

#### 6. 查看当前机器适合哪些模型

```bash
python -m workbench.cli list-models
python -m workbench.cli list-models --json
```

#### 7. 提前下载指定模型

```bash
python -m workbench.cli download-model base
python -m workbench.cli download-model large-v3-turbo-q5_0
```

#### 8. 对历史转录重新加工

```bash
python -m workbench.cli enhance outputs/transcriptions/run_20260906_120000_abcd1234 --profile fast --json
```

`--asr-engine auto|qwen3-asr|whisper-cpp`、`--asr-model`、`--postprocess-profile off|balanced|quality|fast` 和 `--speaker-mode off|auto|required` 同样适用于 `transcribe-page` 与 `transcribe-feed`。单链接脚本也支持这些选项：

```bash
python scripts/transcribe_link.py "https://www.xiaoyuzhoufm.com/episode/69a64629de29766da93331ec" \
  --profile fast --speaker-mode auto --json
```

小宇宙项目页会额外写入 `source.raw.html`、`source.shownotes.md`、`source.comments.top20.json` 和 `source.context.json`。默认使用本地 Next.js 数据解析；设置 `FIRECRAWL_API_KEY` 并将 `NEXTECHO_CRAWLER=firecrawl` 才会改用 Firecrawl。没有 pyannote/HF token 时会使用轻量本地聚类，结果必须人工复核姓名对应关系。

## 给 Agent 怎么用

NextEcho 既可以给 Agent 走命令行，也可以让 Agent 启动网页给用户操作。

### 方式一：让 Agent 走 CLI

需要通过Skill调用时，优先使用前面的Agent安装包。需要安装兼容网页或旧CLI时，把仓库交给Agent并让它读取：

```text
AGENT_INSTALL.md
```

推荐对 Agent 说：

> 请按 AGENT_INSTALL.md 安装 NextEcho。Skill模式无需启动网页；只有我需要网页界面时再启动。

当 Agent 通过 CLI 工作时，常见调用方式如下。

#### 1. 让 Agent 先做环境检查

```bash
python -m workbench.cli doctor
python -m workbench.cli doctor --json
python -m workbench.cli list-models --json
```

#### 2. 让 Agent 转录一个本地文件或直链

```bash
python -m workbench.cli transcribe /path/to/audio.mp3 --quality accurate --json
python -m workbench.cli transcribe "https://example.com/video.mp4" --quality accurate --json
python -m workbench.cli transcribe /path/to/audio.mp3 --model base --json
```

#### 3. 让 Agent 先解析平台页面

```bash
python -m workbench.cli resolve-sources "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --json
```

#### 4. 让 Agent 直接跑平台链接

```bash
python -m workbench.cli transcribe-page "https://www.bilibili.com/video/BV1g6okBLEtL/" --quality fast --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --quality accurate --json
python -m workbench.cli transcribe-page "https://www.xiaoyuzhoufm.com/episode/61ee26c84675a08411f51570" --model small --json
```

#### 5. 让 Agent 处理 RSS feed

```bash
python -m workbench.cli transcribe-feed "https://example.com/feed.xml" --limit 3 --quality fast --json
```

Agent 工作时建议遵循这套顺序：

1. 先跑 `doctor`
2. 再跑 `list-models`，根据用户机器选择 `tiny / base / small / medium / large-v3-turbo-q5_0`
3. 如果需要，先执行 `download-model <model>`
4. 如果输入是平台链接，先跑 `resolve-sources`
5. 再决定是 `transcribe`、`transcribe-page` 还是 `transcribe-feed`
6. 先读取 `manifest.json`，再读取每个 item 下的 `metadata.json` 和 `transcript.txt`

### 方式二：让 Agent 启动网页给用户操作

如果用户更习惯可视化操作，可以让 Agent 直接启动：

```bash
python -m workbench.cli serve
```

然后让用户打开：

```text
http://127.0.0.1:8765
```

这种方式适合：

- 用户要自己粘链接和上传文件
- Agent 只负责搭环境和打开入口
- 用户想人工检查最终结果

## 访谈逐字稿怎么用

如果你希望输出 `Speaker 1 / Speaker 2` 形式的访谈逐字稿，需要额外准备说话人分离依赖。

### 轻量本地 fallback

如果你不想配置 Hugging Face token，可以先装：

```bash
pip install -r requirements-speakers-lite.txt
```

这会启用 `segment-clustering` 后端，适合先跑通流程。

### 更稳的 pyannote 主方案

如果你要更强的 speaker diarization，再装：

```bash
pip install -r requirements-speakers.txt
```

然后设置：

```bash
export HF_TOKEN=your_token_here
```

PowerShell：

```powershell
$env:HF_TOKEN="your_token_here"
```

如果不想使用 Hugging Face token，但希望比纯声学统计聚类更稳，可以安装本地 speaker embedding 中间档：

```bash
pip install -r requirements-speakers-embedding.txt
```

它会自动优先于轻量聚类使用，模型在本机 CPU 上运行；仍然建议用 `speaker_map.json` 人工核对姓名，尤其是节目本身已经混音成单声道时。

### 无 Token 的 sherpa-onnx / 3D-Speaker 实验后端

Apple Silicon 和常见桌面平台可安装固定版本的 sherpa-onnx：

```bash
pip install -r requirements-speakers-sherpa.txt
```

它需要两份本地 ONNX 文件，默认位置为：

```text
models/speaker-diarization/segmentation.onnx
models/speaker-diarization/embedding.onnx
```

前者使用 sherpa-onnx 发布的 Pyannote segmentation 3.0 转换模型，后者使用中文 3D-Speaker ERes2Net。模型不会提交到 Git，也不会自动打包进安装器。可用 `doctor` 核对实际后端和模型状态。这个后端仍是实验候选；在私有金标完成前，不能仅依据社区 benchmark 把它设为生产默认。

### 使用命令

#### 1. 直接从音频生成访谈逐字稿

```bash
python -m workbench.cli speaker-transcript /path/to/audio.wav --quality accurate --json
python -m workbench.cli speaker-transcript /path/to/audio.wav --model base --json
```

#### 2. 从已有单条转录结果继续生成

```bash
python -m workbench.cli speaker-transcript /path/to/run_xxx
```

程序会输出：

- `diarization.raw.json`
- `diarization.raw.rttm`
- `transcript.speakers.json`
- `transcript.speakers.txt`
- `transcript.speakers.md`
- `speaker_map.json`

`diarization.raw.*` 只保存匿名声学分段，文本自我介绍、shownotes 和人工更名不会改写它。姓名与置信度保存在 `transcript.speakers.json` 的身份映射层；证据不足时显示 `unknown`。

## 产物会输出到哪里

每次运行都会生成一个 `run_xxx/` 目录，结构如下：

```text
outputs/transcriptions/run_xxx/
├── manifest.json
├── run_config.json
└── items/
    └── 001_<source_label>/
        ├── metadata.json
        ├── 录音原声.<ext>
        ├── 转录音频.wav
        ├── transcript.txt
        ├── transcript.json
        ├── transcript.srt
        ├── transcript.vtt
        ├── source.raw.html              # 小宇宙页面原文
        ├── source.shownotes.md          # 展开后的节目 shownotes
        ├── source.comments.top20.json   # 按点赞数排序的评论
        └── source.context.json           # 节目/嘉宾/资料索引
```

如果生成了访谈逐字稿，通常还会在 run 根目录看到：

- `transcript.speakers.json`
- `transcript.speakers.txt`
- `transcript.speakers.md`
- `speaker_map.json`

## 模型复用

如果你已经有 Whisper 模型，不必重复下载，可以直接复用：

```bash
export TRANSCRIBE_MODEL_DIR=/path/to/your/whisper-models
```

PowerShell：

```powershell
$env:TRANSCRIBE_MODEL_DIR="C:\path\to\your\whisper-models"
```

程序会优先查找：

- 显式模型路径
- `TRANSCRIBE_MODEL_DIR`
- `WHISPER_MODEL_DIR`
- 项目本地模型目录

如果都没有，首次运行时才会自动下载。

## 常见问题

- 平台页面解析失败：先确认 `yt-dlp` 可用，再重试。某些站点需要浏览器 cookies，或者会触发平台风控。
- B 站返回 `HTTP Error 412 / Precondition Failed`：先在浏览器里打开同一个视频，确认能正常播放，再回到 NextEcho 重试。必要时先更新 `yt-dlp`，或者在启动前设置 `TRANSCRIBE_YTDLP_BROWSER=chrome`，也可以换成 `firefox`、`edge`、`brave`。
- Apple Podcasts 链接已识别，但没有直接音频：优先改贴播客 RSS、节目官网里的单集页面，或者可直接访问的音频链接。Apple Podcasts 更适合作为节目入口，而不是最终音频源。
- 首次运行较慢：通常是在下载模型或首次加载模型。先跑 `python -m workbench.cli doctor`，检查模型路径和推荐模型。

## 关于 token 和成本

- 转录计算在本地完成，不调用云端 LLM 或云端 ASR
- 音视频解析本身不消耗 LLM token
- 如果通过 Agent 发起任务，Agent 在理解指令、运行命令、读取结果时，仍会消耗少量编排 token
- 如果采用 pyannote，首次模型下载可能需要 Hugging Face token，但这不是 LLM/API 计费

## 开源与合规

为了让仓库公开发布时更规范，仓库里已经补充了这些文档：

- [OPEN_SOURCE_COMPLIANCE.md](OPEN_SOURCE_COMPLIANCE.md)：开源发布前的合规结论、风险说明、发布策略
- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)：第三方依赖与许可证清单
- `NOTICE`：第三方组件与商标声明

当前仓库 License：

- `AGPL-3.0`

这意味着：

- 任何人都可以使用、修改和分发本项目
- 如果别人修改后再对外提供网络服务，也需要按 AGPL 的要求公开对应源码
