# 故障处理

- MODEL_MISSING：检查绑定模型目录及必需文件、VAD文件。不要将diarization模型当作ASR权重。
- RESOURCE_BLOCKED：缺资源校准或本机内存不足。保留任务ID；资源条件满足后恢复，不擅自停止用户常驻服务。
- PROVIDER_UNAVAILABLE/MODEL_MISSING：检查已选择的localhost服务和模型，不自动转云端或换模型。
- PROVIDER_BUSY_UNKNOWN：旧HTTP请求可能仍在推理。确认服务空闲前不发新的重请求，不因客户端退出假设推理已停止。
- NEEDS_ALIGNER：当前仅有段级时间。文字仍保留；精确字幕需独立验收。Nano/SenseVoice的对话层可用原生CTC锚，需绑定声学权重、SenseVoice和worker；缺少它们时auto保持partial。
- SOURCE_AUTH_REQUIRED：使用用户提供的cookie文件路径或已有浏览器授权；cookie内容不读入对话。
- CACHE_MISMATCH：旧目录与新请求不一致。用新run目录；仅相同有效计划的中断任务使用jobs resume。
- partial：交付可用路径并简短提示，不把回退原文称作完整润色成功。

优先消费回执的错误代码与诊断路径。只有排障需要时读取相关日志末尾，避免读取全文或将日志无界带入上下文。

操作者确认服务端旧请求已结束后，可用 `nextecho jobs provider-idle <本地端点> --confirm`解除不确定占用。Agent不能仅因等待超时自行确认空闲。
