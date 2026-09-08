# NextEcho 本地 ASR 金标

真实音频、人工文字稿和逐字结果只放在本地忽略目录，不提交到仓库。建议准备 5 段约 4 分钟素材：棚录独白、多人对谈、网络压缩、噪声/BGM、中英混说与专有名词。

将匿名评测条目写入 `benchmarks/gold/manifest.json`：

```json
[
  {"id": "studio_01", "reference": "gold/studio_01.txt", "hypotheses": {"whisper-turbo": "runs/studio_01.txt", "qwen-1.7b": "runs/studio_01_qwen.txt"}}
]
```

使用 `python scripts/evaluate_asr.py benchmarks/gold/manifest.json` 计算归一化 CER。原始音频和版权文本不会被脚本复制到仓库；只提交去敏后的指标 JSON。
