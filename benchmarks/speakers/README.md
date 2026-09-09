# NextEcho 本地说话人金标

真实音频、人工说话人标注和模型输出只放在 `local-private/`，该目录不会提交到仓库。仓库只保存格式说明、评测器与匿名汇总指标。

建议至少准备 5 段、合计 20 分钟：棚录双人、三人对谈、网络压缩、噪声/BGM、中英混说和快速插话。

标注文件示例：

```json
{
  "segments": [
    {"start_ms": 59750, "end_ms": 167940, "speaker_id": "person_mengyan"},
    {"start_ms": 167940, "end_ms": 299850, "speaker_id": "person_lijigang"}
  ]
}
```

`manifest.json` 示例：

```json
{
  "frame_ms": 20,
  "switch_tolerance_ms": 500,
  "items": [
    {
      "id": "clear_dialogue_01",
      "reference": "local-private/clear_dialogue_01.gold.json",
      "hypotheses": {
        "resemblyzer": "local-private/clear_dialogue_01.resemblyzer.json",
        "pyannote-community-1": "local-private/clear_dialogue_01.pyannote.json",
        "sherpa-onnx": "local-private/clear_dialogue_01.sherpa.json"
      }
    }
  ]
}
```

运行：

```bash
python scripts/evaluate_speakers.py benchmarks/speakers/manifest.json
```

输出包括 DER、JER、漏检、误检、speaker confusion、错误切换、匿名簇到金标人物的最佳映射，以及逐样本与总计指标。
