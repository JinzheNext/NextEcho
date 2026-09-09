# NextEcho 说话人识别与本地知识加工 Review / 优化 Spec

- 日期：2026-09-09
- 执行模型：`gpt-5.6-luna`，reasoning effort `max`
- 审查范围：本地历史记录、当前仓库、M1 Pro 16GB 最新本地模型配置、官方 GitHub/模型卡
- 文档性质：实施前技术决策与验收契约，不代表候选模型已经通过本机金标

## 1. 执行结论

历史上确实实现并实践过说话人分离能力，并非从零开始：

1. `2196920`：首次加入基于 pyannote 的 `speaker-transcript`；
2. `9b1e315`：增加无需 Hugging Face token 的本地聚类 fallback；
3. `73e2705`：加入 Resemblyzer speaker embedding、小宇宙上下文和自动说话人模式；
4. `68917aa`：针对当前样例加入“自我介绍”文本锚点纠错；
5. `eda303c`、`6d29ef2`：改进对话稿和知识稿的可读输出。

当前方案已经能生成说话人稿，但应定义为 `best_effort`，还不是高精度“声纹身份识别”。根因是当前实现把三件不同的事混在了一层：

```text
说话活动检测：哪里有人说话
→ 说话人分离：每个时间段属于哪个匿名声音簇
→ 身份识别：匿名声音簇对应孟岩、李继刚还是未知人物
```

下一阶段的 P0 不是继续放大 ASR 或 LLM 参数，而是把这三层拆开，并让所有判断都有来源、置信度和可回滚的原始数据。

推荐路线：

- ASR：继续将 `aufklarer/Qwen3-ASR-1.7B-MLX-5bit` 作为实验候选，与不降级的 Whisper large-v3-turbo 对照；未过金标前不切生产默认。
- 说话人分离：本机并行评测 pyannote Community-1 与 sherpa-onnx；Resemblyzer 降级为兜底。
- 中文声纹：把 3D-Speaker 的 ERes2Net/CAM++ 作为 embedding 候选，优先评估 sherpa-onnx 已导出的 ONNX 模型。
- 后处理：统一使用配置驱动的稳定 4B；M1 Pro 16GB 不默认加载 9B。
- 资源：ASR、diarization、LLM 三个重型阶段严格串行，分别在短生命周期子进程中运行。

## 2. 历史记录核查

### 2.1 Git 历史

| Commit | 已实现能力 | 当前评价 |
|---|---|---|
| `2196920` | pyannote 说话人流水线、CLI、Web、安装说明 | 正确的第一版主链 |
| `9b1e315` | 本地声学特征聚类 fallback | 可用但精度上限低 |
| `d78f0b1` | 任务状态稳定性 | 仍未形成统一重型任务调度 |
| `73e2705` | shownotes、评论、speaker hints、Resemblyzer | 建立了当前实验基础 |
| `a0e488a` | 上下文摘要、对话格式 | 上下文证据边界仍不足 |
| `68917aa` | 自我介绍文本锚点 | 修复样例，但存在全局继承风险 |
| `eda303c` | 合并连续同说话人段落 | 人类可读层改进正确 |
| `6d29ef2` | 知识稿隐藏内部 segment ID | 展示层与机器层开始分离 |

对应核心实现：

- `workbench/speaker_transcript.py`
- `workbench/asr.py`
- `workbench/transcription.py`
- `workbench/postprocess.py`
- `workbench/episode_context.py`
- `app.py`

### 2.2 历史任务记录

2026-06-14 的本地 Codex 工作记录明确记载了：

- Apple Podcasts 页面解析；
- 多人访谈说话人误聚类；
- glossary 增强；
- fallback diarization 的后续回归测试需求。

历史任务 ID：`019ea464-ec63-7bb2-b069-0e81838d1b64`，任务标题为“音视频自动转文字稿”。未发现另一套已经成熟、可直接替换当前实现的私有声纹系统。

## 3. 当前运行快照

2026-09-09 `doctor --json` 的实际结果：

| 项目 | 状态 |
|---|---|
| 机器 | Apple M1 Pro，16GB，arm64 |
| Qwen MLX runtime | 可用，`mlx-qwen3-asr==0.3.5` |
| Qwen 1.7B / 0.6B / forced aligner 缓存 | 均存在 |
| Qwen 自动加载内存门禁 | 当前不通过，系统 swap 约 971MB |
| Whisper Turbo / base | 可用 |
| pyannote | 未安装，无 HF token |
| Resemblyzer | 已安装，当前选中的 diarization backend |
| Ollama | 当前未启动 |

当前 5 分钟实验：

- ASR：`aufklarer/Qwen3-ASR-1.7B-MLX-5bit`
- ASR 耗时：约 52 秒
- 后处理：`qwen3:4b`
- 4 个润色调用 + 2 个摘要调用，总耗时约 220 秒
- 说话人：Resemblyzer + 文本锚点

这说明 ASR 实验链路可跑，但当前 4B 调用次数过多，且说话人正确性依赖样例特化的文本锚点。

## 4. 当前问题清单

### P0-1：文本锚点被错误地当作全局 speaker 锁

当前逻辑检测到“大家好，我是李继刚”后，会持续把后续片段归给李继刚。它修复了当前前 5 分钟样例，却会在完整播客再次轮换时系统性误标。

必须改为：

```text
“我是李继刚”
→ 找到该句所在的匿名 diarization cluster
→ 记录 cluster → 李继刚 的身份映射证据
→ 后续切换仍完全由 diarization 时间线决定
```

文本锚点只能辅助身份映射，不得覆盖原始时间线。

### P0-2：ASR 片段粒度过粗，换人点可能位于片段内部

Qwen writer 当前最长可聚合到 15 秒或 50 字，后续按最大时间重叠把整个 segment 指给一个人。真实对话中的插话和交接经常发生在这 15 秒内部。

必须以词级时间戳与 exclusive diarization 时间线做对齐，再按以下边界重新成句：

- speaker 改变；
- 句末标点；
- 静音超过阈值；
- 达到字幕长度上限。

### P0-3：部分转录会被全期 shownotes / 评论污染

当前样例只转前 5 分钟，但后处理可读取 3 小时以上全期 shownotes 和评论。结果可能把后半段内容写进 5 分钟摘要，再引用前 5 分钟的 segment ID，形成伪证据。

必须区分：

- `segment_ids`：实际转录证据；
- `context_refs`：shownotes / 评论提供的外部线索；
- `inferred`：模型根据两类资料做出的推断。

部分转录只允许使用覆盖范围内可定位的 shownotes；无时间戳评论只能作为术语/选题线索，不能作为已发生事实。

### P0-4：Qwen 路径仍受到 Whisper 前置检查影响

当前转录入口会提前解析 Whisper 模型和依赖，即使实际选择了 Qwen。这使两个 backend 的依赖生命周期互相污染。

应延迟检查：

```text
qwen3-asr  → 只检查 MLX、ASR 模型、forced aligner
whisper-cpp → 只检查 whisper-cli、Whisper 模型
```

只有进入 fallback 时才解析下一后端依赖。

### P0-5：重型任务没有完整统一的锁与心跳

ASR、diarization、Ollama 仍可能竞争 16GB 统一内存；speaker job 也没有完整的 15 秒 heartbeat 契约。

必须提供统一 `HeavyTaskCoordinator`：

```text
ASR worker 退出并释放 Metal
→ diarization worker 退出并释放 PyTorch/ONNX
→ 4B worker 开始并按任务结束卸载
```

### P1-1：Resemblyzer 不是高精度 diarization 默认

Resemblyzer 的 embedding 可用于实验和兜底，但对麦克风变化、压缩、背景音乐、短回应和重叠语音敏感。当前出现的 79 秒错误切换就是典型表现。

### P1-2：音频增强可能改变声纹特征

当前 diarization 使用高通、低通、降噪、压缩后的音频。压缩和降噪可能改变 embedding 分布。应保留两路：

```text
转录音频：可做语音增强
声纹音频：原始 16kHz mono + 轻量响度归一化
```

是否使用降噪必须通过金标 A/B 决定。

### P1-3：后处理分块过小

当前 240–320 字一个 polish chunk，会增加模型切换外的调用开销，也容易破坏跨句语境。

建议按 speaker turn 与标点切块：

- 目标 1,200–2,500 中文字符；
- 硬上限 4,000；
- 携带前后只读上下文；
- 模型不得修改 speaker、segment ID、时间戳。

### P1-4：中文关键词 fallback 质量不足

字符 bigram 会产生“知有、入议、李继、经元”等无意义词。应改用轻量中文分词/词典，并优先保留人物、组织、产品、工具及 glossary 命中项。

### P1-5：配置散落在源码、CLI、脚本和 skill 中

ASR 模型、4B/9B、speaker backend、上下文长度和内存门禁目前由多处常量决定。应建立一份中央配置，并允许读取 `mac-local-llm-skill` 的最新 selection 结果。

### P1-6：Web 展示存在外部文本注入风险

页面多处使用 `innerHTML` 渲染标题、URL、错误和外部节目资料。shownotes 和评论是不可信数据，应改为 DOM `textContent` 或统一转义。

## 5. 候选项目 Review

| 候选 | 角色 | 优点 | 风险 | v1 建议 |
|---|---|---|---|---|
| pyannote Community-1 | 高精度 diarization | 生态成熟；有 DER 基准；提供 exclusive diarization；可下载后离线 | 初次需接受 HF 条款/token；PyTorch 较重；需验证 Python 版本和 M1 CPU 性能 | P1 主候选 |
| sherpa-onnx | 轻量跨平台 diarization | macOS arm64；ONNX；支持 diarization、speaker ID/verification、VAD；易打包 | 必须用中文播客金标验证准确率 | P1 主候选 |
| 3D-Speaker ERes2Net/CAM++ | 中文 speaker embedding | 中文会议与声纹模型丰富；参数小；ModelScope 友好 | 原生 toolkit 依赖较重；Apple Silicon 集成未验证 | 优先使用 ONNX 导出模型做候选 |
| Resemblyzer | fallback embedding | 已安装；无 token；低成本 | 不是完整高精度 diarization；容易误聚类 | 保留为 `best_effort` fallback |
| WhisperX | 对齐架构参考 | 词级对齐与 speaker assignment 流程完整 | 依赖重、偏 CUDA；会重复 NextEcho 现有 ASR 边界 | 借鉴算法，不整体引入 |
| Reverb diarization v1 | segmentation 候选 | 官方报告相对 pyannote 3.0 有 WDER 改善；sherpa-onnx 已支持 | 模型许可为 `other`；需核查分发条件 | 仅评测，不先绑定默认 |
| pyannote Precision-2 | 云端高质量 | 官方基准更高 | 云服务，不符合当前纯本地边界 | 排除 v1 |

官方参考：

- pyannote.audio：https://github.com/pyannote/pyannote-audio
- Community-1 模型卡：https://huggingface.co/pyannote/speaker-diarization-community-1
- sherpa-onnx：https://github.com/k2-fsa/sherpa-onnx
- sherpa-onnx diarization 文档：https://k2-fsa.github.io/sherpa/onnx/speaker-diarization/
- 3D-Speaker：https://github.com/modelscope/3D-Speaker
- WhisperX：https://github.com/m-bain/whisperX
- Reverb diarization：https://huggingface.co/Revai/reverb-diarization-v1

## 6. 推荐目标架构

```text
页面抓取 / 本地媒体
  ├─ 原始媒体与 source.context
  └─ 16kHz 音频
       ├─ ASR worker → word timestamps
       └─ Diarization worker
            ├─ VAD / speaker segmentation
            ├─ speaker embedding
            ├─ clustering / overlap
            └─ diarization.raw.json + RTTM

word timestamps + exclusive diarization
→ WordSpeakerAligner
→ 匿名 speaker turn
→ SpeakerIdentityResolver
     ├─ 已注册本地声纹
     ├─ 自我介绍锚点
     ├─ shownotes 主持人/嘉宾元数据
     └─ 人工映射
→ 可读多人逐字稿
→ 确定性术语处理
→ 配置驱动的本地 4B 润色
→ 证据约束知识摘要
```

### 6.1 关键接口

```python
class DiarizationBackend(Protocol):
    name: str
    def diarize(self, audio_path, options, progress_callback) -> DiarizationResult: ...

class SpeakerIdentityResolver(Protocol):
    def resolve(self, diarization, transcript, context, profiles) -> SpeakerIdentityMap: ...

class LocalLlmProvider(Protocol):
    def status(self) -> ProviderStatus: ...
    def generate_json(self, request, schema) -> dict: ...
    def unload(self) -> None: ...
```

后端实现建议：

```text
PyannoteCommunityBackend
SherpaOnnxBackend
ResemblyzerFallbackBackend

OllamaProvider
LlamaCppProvider
```

## 7. 数据与产物契约

### 7.1 不可变 diarization 原始层

新增：

```text
diarization.raw.rttm
diarization.raw.json
```

```json
{
  "schema_version": 1,
  "backend": "pyannote-community-1",
  "model": "pyannote/speaker-diarization-community-1",
  "model_revision": "...",
  "model_digest": "...",
  "audio_sha256": "...",
  "turns": [
    {
      "turn_id": "turn_00012",
      "raw_speaker_label": "SPEAKER_01",
      "start_ms": 126430,
      "end_ms": 139880,
      "confidence": 0.87,
      "overlap": false
    }
  ]
}
```

文本或人工纠错不得覆盖该文件。

### 7.2 身份映射层

`transcript.speakers.json` 升级为 schema v3：

```json
{
  "schema_version": 3,
  "speakers": [
    {
      "speaker_id": "spk_00",
      "raw_labels": ["SPEAKER_00"],
      "display_name": "孟岩",
      "confidence": 0.94,
      "identity_status": "resolved",
      "identity_evidence": [
        {"type": "metadata", "source": "xiaoyuzhou.host"},
        {"type": "self_introduction", "segment_ids": ["seg_00027"]}
      ]
    }
  ]
}
```

低于校准阈值时必须输出 `unknown`，不得为了凑齐 shownotes 中的姓名而强制一一配对。

### 7.3 声纹注册层

新增仅本地存储的 `speaker_profiles.json`：

```json
{
  "profiles": [
    {
      "person_id": "person_mengyan",
      "display_name": "孟岩",
      "sample_paths": ["/local/enrollment/mengyan_01.wav"],
      "embedding_model": "3d-speaker-eres2net",
      "embedding_digest": "...",
      "threshold": 0.78,
      "consent": true,
      "created_at": "2026-09-09T00:00:00+08:00"
    }
  ]
}
```

声纹 embedding 和注册音频属于敏感生物特征数据：默认不入 Git、不写云端、不自动跨用户共享，并提供删除入口。

### 7.4 证据约束摘要

```json
{
  "claim": "……",
  "evidence": {
    "segment_ids": ["seg_00012"],
    "context_refs": [],
    "speaker_ids": ["spk_00"],
    "time_range": {"start_ms": 80300, "end_ms": 92300},
    "confidence": 0.88,
    "inferred": false
  }
}
```

规则：

- 转录事实必须有 `segment_ids`；
- shownotes/comment 事实必须有 `context_refs`；
- 混合推断必须标记 `inferred: true`；
- 金句必须能在增强稿逐字找到；
- 无有效证据的 claim 丢弃。

## 8. 中央配置提案

当前没有统一配置文件。建议新增仓库默认配置 `config/nextecho.toml`，用户覆盖配置放在仓库外，例如 `~/Library/Application Support/NextEcho/config.toml`。

M1 Pro 16GB 的建议配置：

```toml
schema_version = 1

[hardware]
profile = "16gb-apple-silicon"
profile_source = "/Users/a1-6/Documents/jerry_projects/mac-local-llm-skill/configs/hardware-profiles.json"
reserve_memory_gb = 3.0
max_swap_delta_gb = 1.0

[asr]
engine = "auto"
language = "zh"
accurate_model = "aufklarer/Qwen3-ASR-1.7B-MLX-5bit"
fast_model = "Qwen/Qwen3-ASR-0.6B"
forced_aligner = "Qwen/Qwen3-ForcedAligner-0.6B"
whisper_fallback = "large-v3-turbo-q5_0"
experimental_accurate_model = true

[diarization]
backend = "auto"
benchmark_candidates = ["pyannote-community-1", "sherpa-onnx", "resemblyzer"]
fallback_order = ["pyannote-community-1", "sherpa-onnx", "resemblyzer"]
min_speakers = 2
max_speakers = 6
allow_unknown = true
use_exclusive_timeline = true
worker_python = "3.12"

[speaker_identity]
enable_metadata_hints = true
enable_self_introduction_hints = true
enable_voice_enrollment = false
minimum_confidence = 0.90
text_anchor_scope = "cluster_mapping_only"

[postprocess]
profile = "fast"
provider = "selection"
selection_source = "/Users/a1-6/Documents/jerry_projects/mac-local-llm-skill/configs/models.json"
preferred_model_id = "gemma-e4b-iq4xs"
fallback_model = "qwen3:4b"
context_tokens = 16384
polish_target_chars = 1800
polish_max_chars = 4000
keep_alive_seconds = 600

[context]
shownotes = true
comments_top_n = 20
enforce_transcript_coverage = true
comments_are_evidence = false

[runtime]
heavy_task_concurrency = 1
heartbeat_seconds = 15
stage_checkpointing = true
release_model_after_stage = true
```

这里的 `fallback_order` 只是评测期配置。生产默认必须由本机金标决定，不允许根据社区 benchmark 直接切换。

### 8.1 与最新本地模型配置的关系

`mac-local-llm-skill` 当前配置对 M1 Pro 16GB 的结论：

| 模型 | 实测/定位 | NextEcho 建议 |
|---|---|---|
| Gemma E4B IQ4_XS | 16K 约 31.7 tok/s；稳定默认 | 优先评估为统一后处理模型 |
| `qwen3:4b` | 短上下文约 36.3 tok/s，但 swap 增量约 3.7GiB | 速度候选和 fallback，不自动视为稳定默认 |
| Qwen3.5 9B MLX | 16K 约 23.4 tok/s；TTFT 约 107 秒 | 仅显式高质量实验 |
| Ornith 1.5 9B | 16K 约 19.2 tok/s | 不作为默认 |

因此 NextEcho 应把当前硬编码的 Ollama `qwen3:4b` 改成配置驱动，并支持本地 llama.cpp 服务；“统一 4B”策略保留，“固定某一个 4B runtime”应取消。

当前 Qwen 的自动内存门禁只检查系统累计 swap 是否低于 512MB，过于保守。应改为：

- 当前 memory pressure；
- 可用内存；
- 本阶段启动前后的 swap delta；
- 最近一次模型实测峰值；
- 预留至少 3GB 系统空间。

## 9. CLI / API / UI

### CLI

保留：

```bash
nextecho transcribe-page URL \
  --asr-engine auto \
  --speaker-mode auto \
  --postprocess-profile fast
```

扩展：

```bash
nextecho speaker-transcript RUN_OR_ITEM \
  --backend auto \
  --min-speakers 2 \
  --max-speakers 4

nextecho speaker-enroll "孟岩" sample-1.wav sample-2.wav
nextecho speaker-profiles list
nextecho speaker-profiles remove person_mengyan
nextecho benchmark speakers --gold-dir benchmarks/speakers/local-private
```

### API

```text
GET  /api/models/status
POST /api/models/download
POST /api/runs/<run_id>/enhance
POST /api/runs/<run_id>/speaker-transcript
GET  /api/runs/<run_id>/speaker-status
POST /api/runs/<run_id>/speaker-map
GET  /api/speaker-profiles
POST /api/speaker-profiles
DELETE /api/speaker-profiles/<person_id>
```

### UI

必须显示：

- 实际 ASR engine/model/revision；
- diarization backend/model；
- 匿名 speaker 与显示姓名；
- identity evidence 与置信度；
- fallback 原因；
- 转录覆盖时间范围；
- partial 状态；
- 各阶段耗时和模型内存门禁；
- 手工更正 speaker 的入口。

人工更正只修改身份映射层，并保留原始 diarization。

## 10. 任务状态与目录

统一状态机：

```text
queued
→ waiting_for_model
→ downloading
→ transcribing
→ diarizing
→ aligning
→ identifying_speakers
→ polishing
→ summarizing
→ packaging
→ ready / partial / failed
```

中间目录使用稳定中文或英文语义名：

```text
staging/原始音频/
staging/标准化音频/
staging/语音识别/
staging/说话人分离/
staging/智能整理/
```

短期系统临时目录任务结束后清理；`tmphv5ilncb` 等随机路径不得写入 manifest、UI 或最终报告。

指纹至少包含：

- 原始音频与 transcript hash；
- context hash 与覆盖范围；
- diarization model digest；
- speaker identity map hash；
- glossary hash；
- prompt version；
- postprocess profile；
- LLM model digest。

## 11. 实施顺序

### P0：纠正数据模型与现有逻辑

1. 将文本锚点从“全局 speaker 锁”改成“cluster identity evidence”；
2. 保存不可变 `diarization.raw.*`；
3. 引入 anonymous speaker ID、display name、unknown、confidence；
4. 用词级时间戳对齐 speaker turn；
5. 修复 Qwen/Whisper 延迟依赖检查；
6. 限制 shownotes/comments 到实际时间覆盖；
7. 给 ASR/diarization/LLM 加统一重型锁、heartbeat 和阶段 checkpoint；
8. 固化当前 5 分钟样例为私有回归样本。

### P1：本机 diarization 基准

比较：

```text
当前 Resemblyzer
pyannote Community-1
sherpa-onnx + pyannote/reverb segmentation + ERes2Net
3D-Speaker CAM++/ERes2Net 候选
```

如果 sherpa-onnx 的 DER 与最优方案差距不超过 1.5 个绝对百分点，同时速度至少快 1.5 倍或峰值内存低 30%，优先将其设为本地默认；否则以准确率更高者为默认。

### P1：声纹注册与人工纠正

1. 本地 speaker enrollment；
2. cluster embedding 与已注册 embedding 阈值校准；
3. 低置信度输出 unknown；
4. UI 手工映射；
5. 修正结果单独保存，不覆盖原始 diarization。

### P2：后处理与上下文

1. 接入配置驱动的 Gemma E4B / qwen3:4b provider；
2. 放大合理的 speaker-aware polish chunk；
3. 使用中文 tokenizer 与术语词典；
4. 分离 segment evidence 和 context evidence；
5. 增加完整事实校验与 partial 标识。

### P2：产品化

1. 模型状态、下载进度、缓存 revision 与 digest；
2. 历史任务重新 diarize / enhance；
3. manifest schema v3；
4. 模型许可、NOTICE、缓存和隐私说明；
5. UI 外部文本安全渲染。

## 12. 金标与验收

### 12.1 私有 20 分钟金标

至少包含 5 段，每段约 4 分钟：

- 棚录双人对谈；
- 三人或多人对谈；
- 网络压缩；
- 噪声/BGM；
- 中英混说、专有名词和快速插话。

标注格式需要覆盖：

```json
{
  "segments": [
    {
      "start_ms": 59750,
      "end_ms": 167940,
      "speaker_id": "person_mengyan",
      "text": "……",
      "overlap": false
    }
  ]
}
```

音频、姓名声纹和版权文本只留在本地忽略目录；仓库只提交评测工具、匿名指标和模板。

### 12.2 说话人门槛

- 报告 DER、JER、speaker confusion、false switch、speech miss；
- 当前 5 分钟样例在 2:47 处正确交接，之后的再次轮换也必须正确；
- 不出现连续 3 秒以上有声内容遗漏；
- speaker identity 准确率达到 95%，否则输出 unknown；
- 所有最终 turn 可回溯到 raw diarization；
- 时间戳严格单调且不越界；
- 重叠语音不能静默改写成唯一确定说话人。

### 12.3 ASR 与后处理门槛

- Qwen 1.7B 5-bit 必须记录 repo、revision、quantization、sha256 和 MLX 版本；
- 金标 CER 相对现生产链下降至少 20%，否则保留实验状态；
- 数字、URL、保护词保留率 100%；
- speaker、时间戳、segment ID 不允许被 LLM 修改；
- 所有摘要观点都有有效证据；
- 金句逐字可定位；
- 5 分钟 partial 测试不得泄漏三小时后段内容；
- 原始 `transcript.*` 永不被覆盖。

### 12.4 资源门槛

- ASR、diarization、LLM 严格串行；
- M1 Pro 16GB 始终保留约 3GB 系统空间；
- 记录每阶段峰值内存、耗时、swap delta；
- 9B 必须显式 opt-in；
- 服务中断后可从最后一个阶段 checkpoint 恢复；
- 任一增强阶段失败时原始转录仍可用，并标记 `partial`。

## 13. 边界与风险

- v1 不使用云端 diarization 或云端 LLM；pyannote token 只用于首次下载受限模型，推理在本机进行。
- 不承诺只靠 shownotes 就能把匿名声音可靠命名。
- 不把节目中提到的人自动视为说话人。
- 不在没有同意和明确用途时建立永久声纹库。
- 社区 5-bit ASR 模型在模型卡、revision 和许可证信息补齐前保持 experimental。
- Reverb 模型许可标记为 `other`，完成分发许可核查前不得打包进安装器。
- GitHub/模型方 benchmark 只用于候选筛选，最终默认由当前机器的私有金标决定。

## 14. 最终默认策略

在 P0 和 P1 验收前：

```text
ASR：Qwen3-ASR 1.7B 5-bit（实验）/ Whisper Turbo（稳定基线）
说话人：Resemblyzer（best_effort，必须显示需人工复核）
后处理：统一 4B，当前 qwen3:4b；不自动加载 9B
```

在 P1 金标完成后：

```text
ASR → diarization → speaker identity → deterministic cleanup
→ 配置驱动 4B polish → evidence-bound summary
```

pyannote Community-1 与 sherpa-onnx 的胜者成为 Apple Silicon 本地默认；另一者作为高质量或跨平台备选。Resemblyzer 只在两者不可用时兜底。

