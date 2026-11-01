# Native Metal分人候选

`nemotron-metal-candidate`为显式Apple Silicon macOS实验后端。需用户单独准备固定运行时、模型、资产台账和CTC环境；程序不会随普通安装下载权重或切换到该后端。

资产原始代码、权重、转换物和声明分别列在[模型许可证台账](model-license-inventory.md)。NeMo-Speech.cpp运行时归档与模型不随本版程序、源码包或Skill分发，台账中的未知条件不表示已获模型再分发许可。

配置使用`diarization.backend`与`diarization.nemotron.asset_ledger/runtime_root`，所有文件必须符合台账hash与固定接口身份。不存在的资产在auto分人中记录partial并继续文字/润色；required阻断。台账损坏、hash不符、资源或所有权失败仍阻断。

测量的原始声学身份与姓名映射分层保存。标题名单、文字位置、同块编号或阅读层拼词不能独立证明人物身份。Native/CTC可复用已校验的完整阶段，但中断的Native阶段不冒充可恢复模型状态；精确字幕、声学质量及姓名依据需独立验收。
