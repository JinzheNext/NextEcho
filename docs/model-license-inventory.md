# 本机模型与运行时许可台账

2026-10-20。机器可读明细见同目录 [model-license-inventory.json](model-license-inventory.json)，包括逐文件字节数、完整SHA-256、模型/转换仓库固定revision、运行时版本及许可材料摘要。原始公开元数据和逐文件核验记录由维护者本地留存，个人路径与原始审计材料不随安装包分发。

本台账登记可观察身份及许可声明，不作整项目发行放行，也不改变项目AGPL-3.0-only。所有模型权重、外部运行时、原音频和声纹都不属于应用wheel或Skill包内容；重建包仍须单独检查。未知事项保持明确未知。

| 实际资产 | 身份与许可证据 | 尚未证明的事项 |
| --- | --- | --- |
| Nano PT及词表/配置 | [官方固定提交](https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512/tree/272c57b82523ada6fd87095e955f8e29100979ab)；本地model.pt SHA与LFS一致；模型卡Apache-2.0 | 外部运行时整包逐文件代码验证不在本次范围 |
| SenseVoice int8及浮点ONNX | [转换仓库2365bae](https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/tree/2365baeacb507f821a0c8120fcee3d484dba7a07)；本地两权重均匹配LFS，export脚本及LICENSE已hash | 原始checkpoint及实际转换执行revision未知。转换LICENSE仅转引FunASR；[原模型卡](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)声明自定义model-license，不能套用FunASR代码MIT |
| 旧Pyannote分段ONNX | [转换仓库9403a69](https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/tree/9403a6902bb58e3d5ae8c7e77c3422de279db2e0)；完整ONNX哈希匹配，包内MIT文本与脚本留证；原模型卡也声明MIT | 原始checkpoint与转换执行revision未知；原仓库仍有独立访问条件，本次未接受条款或下载受限权重 |
| ERes2Net base/large ONNX | 两文件完整hash；文件内metadata明确各自ModelScope模型名及PyTorch2.4.1导出器 | 不据上游代码Apache-2.0推断权重许可；原始checkpoint及转换revision/条款未核清，不能随包分发 |
| Silero VAD ONNX | 实际使用文件完整hash及官方sherpa release来源记录 | 上游项目MIT不替代精确发布资产的生成来源链；revision未知 |
| Qwen外部文本对齐器 | 官方c07281df，模型及配置/词表逐文件重验，模型卡Apache-2.0 | 对齐精度仍须人工50点，许可声明不证明质量 |
| Nemotron Q8_0 / NeMo-Speech.cpp Metal | 模型f667ed73及107,012,128字节GGUF哈希；运行时v0.2.0代码commit与所有已登记普通文件重验；模型卡OpenMDW-1.1，代码Apache-2.0并有第三方通知 | 转换执行revision未知；短探针不证明长上下文、质量或发行合规 |
| Qwen3.5 2B/4B/9B本地Ollama转换 | 当前元数据量化为mxfp8/nvfp4/nvfp4；实际本地manifest及全部声明blob重验：2,116个去重blob，共15,955,133,617字节；当前license blob声明Apache-2.0 | 不能由本地manifest推导出服务报告的tag digest；不能证明服务内存中实际权重。原始/转换revision未知，当前license文本hash也与历史抓取记录不同，两份hash分别保留 |
| Community-1 | 已有固定公开元数据和CC-BY-4.0声明 | 未取得权重、未接受访问条件，不能宣称可用或已完整许可核验 |
| 27B/35B | 延后，不加载、不下载、不新增评测 | 不从小模型继承许可或成绩 |

外部Python运行时按当前已安装版本登记METADATA及通知文件hash，包含FunASR、PyTorch、sherpa-onnx、SoundFile、Transformers等。该证据不等于整个运行时所有代码字节已审核，也不把它们包含进NextEcho发行物。PyTorch等组件的多个第三方许可通知不能简化成一个MIT/Apache标签。

SenseVoice自定义模型协议与软件代码MIT分开。当前原模型卡转引[FunASR MODEL_LICENSE](https://github.com/modelscope/FunASR/blob/main/MODEL_LICENSE)，其声明包含署名、保留模型名和附加条款；本地转换实际采用的历史版本及适用关系仍未建立。因此不将该转换物标为“已确认可商用/再分发”，也不以当前网页条款改写历史许可。

本台账不将未知ONNX标为已获商业使用许可。上述来源链和兼容模型范围仍需完整核查；安装验证、真实模型质量和人工审阅分别记录。本版仅分发应用代码与Skill，模型和外部运行时不在分发范围内。
