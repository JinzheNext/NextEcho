# Third-Party Licenses

This file records the main third-party dependencies and license considerations for `NextEcho`.

The `Version` column reflects either:

- the version constraint declared in this repository, or
- the system dependency pattern currently expected by the project

System-installed dependencies may vary by machine and should be re-audited if bundled into release artifacts.

| Dependency | Version | License | Commercial Use | Redistribution | Copyleft Risk | Notes |
|------------|---------|---------|----------------|----------------|---------------|-------|
| Flask | `>=3.0` | BSD-3-Clause | Yes | Yes, keep license notice | Low | Declared in `requirements.txt`. |
| pyannote.audio | `>=4.0,<5.0` | MIT | Yes | Yes, keep license notice | Low | Optional dependency for speaker diarization. |
| sherpa-onnx | `1.13.7` | Apache-2.0 | Yes | Yes, keep license and notices | Low | Optional local speaker diarization runtime; model weights are downloaded separately. |
| Resemblyzer | `0.1.4` | Apache-2.0 | Yes | Yes, keep license and notices | Low | Optional best-effort speaker embedding and enrollment backend. |
| 3D-Speaker ERes2Net ONNX | model artifact | Apache-2.0 upstream project; verify the exact release asset | Yes | Preserve model provenance and license | Low | Chinese speaker embedding model used by the optional sherpa-onnx route. Not bundled by default. |
| numpy | `>=2.0` | BSD-style | Yes | Yes, keep license notice | Low | Optional dependency for local speaker clustering fallback. |
| scikit-learn | `>=1.6` | BSD-3-Clause | Yes | Yes, keep license notice | Low | Optional dependency for local speaker clustering fallback. |
| Whisper / OpenAI Whisper | indirect model family | MIT | Yes | Yes, keep license notice | Low | The repository does not directly depend on the Python `openai-whisper` package, but uses Whisper-compatible models through whisper.cpp. |
| whisper.cpp / whisper-cli | system dependency | MIT | Yes | Yes, keep license notice | Low | Required local binary used for transcription. |
| Whisper ggml model weights | model-dependent | MIT | Yes | Yes, keep provenance and license notice | Low | Downloaded from `ggerganov/whisper.cpp` paths unless reused locally. |
| yt-dlp | system dependency | Unlicense | Yes | Yes | Low | License is permissive, but platform terms and content rights remain a separate risk. |
| FFmpeg | system dependency | Varies by build: LGPL or GPL | Yes | Yes, but obligations depend on build | Medium to High | If bundled, audit exact compile flags. GPL-enabled builds can impose GPL redistribution duties. |
| Python runtime | environment dependency | PSF | Yes | Yes, keep license notice | Low | Not currently bundled by the project itself. |
| PyInstaller | not used | GPL-2.0-or-later with exception | Yes | Yes | Medium if adopted later | Not used by the current project. Re-audit if packaging strategy changes. |
| Tauri | not used | MIT OR Apache-2.0 | Yes | Yes | Low | Not used by the current project. |
| Electron | not used | MIT plus bundled notices | Yes | Yes | Low to Medium | Not used by the current project. Re-audit if adopted later. |
| pyannote speaker-diarization community model | model-dependent | See model card and attribution terms | Usually yes | Usually yes with attribution | Low | Check the model card before bundling or redistributing model weights. |
| sherpa-onnx Pyannote segmentation 3.0 ONNX | model artifact | Verify upstream Pyannote model terms before redistribution | Local use only until audited | Do not bundle by default | Medium | Downloaded separately for local evaluation; conversion/runtime license does not replace model-weight terms. |

## Notes for Maintainers

- Keep this file updated when `requirements*.txt`, packaging scripts, or bundled assets change.
- If you start shipping app bundles, installers, or binary releases, replace `system dependency` entries with exact versions.
- FFmpeg requires the most caution. A user-installed FFmpeg binary and a bundled FFmpeg binary are different compliance situations.
# 本地模型运行时补充

- `mlx-qwen3-asr` 0.3.5：Apache-2.0；仅作为 Apple Silicon 可选运行时。
- Qwen3-ASR 0.6B / 1.7B：Apache-2.0；模型权重由用户主动下载并保存在本地。
- 社区 Qwen3-ASR 1.7B MLX 5-bit：保持 experimental；每次产物记录仓库、revision、量化、权重摘要和 MLX 版本，分发前重新核验模型卡与来源。
- Qwen3.5 9B：Apache-2.0；通过本机 Ollama 主动拉取并仅用于本地推理。
