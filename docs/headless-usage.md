# 无网页使用 NextEcho

NextEcho将媒体解析、识别、声学分人、保真润色和文件交付放在本机执行。Agent只消费有界的任务回执，普通调用不把全文加载进聊天上下文。

## 安装与模型准备

优先下载[v0.2.12预览版完整Agent包](https://github.com/JinzheNext/NextEcho/releases/tag/v0.2.12)，读取包内INSTALL.md。安装步骤、目录和回滚见[Agent安装包说明](agent-bundle.md)。纯Skill包需要另行安装兼容运行时。

安装器只安装程序和Skill。Python≥3.11、FFmpeg、模型权重及独立模型worker需要另行准备；首次安装核心Python依赖需要网络。完整模型链路的真实验证范围是Apple Silicon macOS。POSIX后台入口提供给macOS/Linux；Windows使用原兼容安装器。

可使用本机已有模型根目录，按实际路径绑定：

```bash
nextecho prepare --model-root /path/to/models \
  --worker-python /path/to/model-env/bin/python \
  --vad-model /path/to/silero_vad.onnx --profile fast --no-web
nextecho doctor --capability asr --profile fast --json
nextecho doctor --capability speaker --json
```

模型根目录对应`fun-asr-nano-pt`和`sherpa-sense-voice`等已登记子目录，不能将分人模型当作ASR权重。需要资源校准时给prepare增加`--calibrate-audio /path/to/short.wav`，使用≤30秒、16kHz单声道WAV；资源护栏保持生效，校准不会自动把候选提升为ready。

Ollama/本地chat服务、准确模型名称和请求预算需按所选档案准备。先用`ollama list`核对已注册名称；例如当前4B档案使用`qwen3.5:4b-mlx`，这是本地登记名称，不应假设它是可以直接pull的公共标签。程序不自动改用其他模型。

## 默认调用与个人试用选择

支持SenseVoice int8快速识别、Fun-ASR-Nano CPU识别及本地2B/4B/9B等润色档案。候选需要明确试用授权，模型可运行、资源校准与产品质量验收分别记录。

用户已选择SenseVoice+4B时，可在本机配置保存：

```toml
[agent]
default_profile = "fast"
approved_trial_models = ["sensevoice-int8", "qwen35-4b"]

[profiles.fast]
asr = "sensevoice-int8"
polish = "qwen35-4b"
```

授权只对两个模型ID生效，不修改candidate/ready、不跳过文件或资源校验。未获准的候选默认仍阻断；也可以为单次受控试用显式传`--profile fast --polish-model qwen35-4b`。27B/35B不是本版自动启动的模型。

安装后的Skill将实际运行时绑定在技能目录之外。普通调用先保存请求，再执行：

```bash
python3 /path/to/nextecho/scripts/run.py --create-request "https://example.org/episode"
python3 /path/to/nextecho/scripts/run.py --request-file /returned/request.json
```

每个新的用户意图创建新请求；响应丢失或提交状态不明时重用原request_file。请求冻结参数、绑定和caller cwd，不能通过编辑旧JSON改变任务。已接受意图优先查原任务，不能仅因配置更新而重复提交。所有处理参数在创建请求时追加。

## 直接CLI与后台任务

```bash
nextecho run /path/to/recording.wav --profile fast --polish-model qwen35-4b
nextecho run "https://example.org/episode" --profile fast --polish-model qwen35-4b --speaker-mode auto --detach --output json
nextecho jobs wait JOB_ID --timeout 60 --output json
nextecho jobs status JOB_ID --output json
nextecho jobs cancel JOB_ID --output json
```

默认只润色，只有显式`--postprocess polish-and-summary`才生成摘要。分人模式为off、auto、required；Skill默认auto，CLI的具体默认以有效配置为准。未确认姓名保留匿名标签；用户确认姓名只用于对应录音，不建立跨录音注册声纹。

auto缺少可选分人资产时保留文字、继续润色并返回partial；required仍阻断。台账损坏、hash不符、资源/所有权错误始终阻断，不以“可选”隐藏。Native候选需显式配置其固定资产台账与独立运行时，不自动下载、安装或更换后端。CTC对话时间锚不等于精确字幕质量已验收。

后台接受只表示queued/running，不等于文件完成。使用原job_id有界等待，不重复提交。原始文字、采用润色层与阅读排版分别保存；回退原文、语义待审和分人缺项在独立回执中说明。

## 恢复与补修

```bash
nextecho jobs resume JOB_ID --mode exact --output json
nextecho jobs resume JOB_ID --mode refresh --output json
nextecho jobs resume JOB_ID --mode refresh --repair-polish --output json
```

exact沿用原计划、日期与选中的单集；refresh显式使用当前配置并记录差异。两者创建新attempt，旧任务不修改。Skill调用resume也先用--create-request持久化意图。按句补修用当前保真规则重验旧候选，只重试未通过的句子，不能降低数值、否定或元字段保护来制造成功。

只重新分人可使用`nextecho refine-speakers /path/to/published-run --backend BACKEND`，复用已核验的原稿和润色层；不重跑ASR或润色。结果另存新交付，模型候选质量状态独立保留。

## 阅读稿与来源资料

主文件命名为“栏目_单集标题_转录日期.md”，保留result.md兼容副本。顶部为：

> NextEcho，让播客自动本地转录进知识库
>
> @金哲Next，一起用好AI走向Next Level，欢迎关注小红书&公众号～

目录链接使用正文标题，适合Obsidian单文件导入。实际捕获的公开主评论和回复原文附在文末，明确快照范围；不声称抓全平台评论，不将评论当作发言或声纹身份的证据。

阅读层清理短停顿、机械重复和明确的断词，并逐ID审计；原稿、字幕和声学ID不改。没有已确认身份时，通用阅读标签为“发言人1”，它不代表已经验证了同一自然人的声纹。音频覆盖、处理完成与身份待审在receipt的completion中分列，partial不必然表示漏转。

### 中文上下文校订（显式试用）

可按配置启用简体派生稿、官方专名线索及有界shownotes/评论检索。来源只辅助拼写与音近词校对，不把编者知识纠正补成说话人的发言，不从标题双人名单推断声音身份。原始层与改动证据保留。

## 输出与故障

ready退出码0；partial为4；failed为3；blocked为5；cancelled为6。后台接受返回0不代表ready，等待超时为非终态。compact JSON≤2KiB，正文留在磁盘。

默认保留至少3GiB可用内存，监测swap增量与压力，重阶段共用本机租约。不要为重试停止无关进程、放宽护栏或下载未授权模型。

- MODEL_CANDIDATE：保存用户明确选择的试用组合，或单次显式选择profile。
- MODEL_MISSING/DEPENDENCY_MISSING：绑定实际缺失的模型、VAD或worker。
- RESOURCE_BLOCKED：核对校准与当前资源，条件满足后恢复。
- CACHE_MISMATCH：原交付缺失/改变或输入不一致，创建新attempt；只有恢复原字节才能重新验证旧交付。
- PROVIDER_BUSY_UNKNOWN：旧请求可能仍在服务端推理，人工确认服务端空闲后才能使用`jobs provider-idle ENDPOINT --confirm`。

这些状态不会自动触发云端推理或静默更换模型。程序和模型计算在本机执行；Agent自然语言编排仍可能消耗宿主token，不能宣称全流程token为零。
