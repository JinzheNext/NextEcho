# Third-Party Licenses

This file records the main third-party dependencies and license considerations for `NextEcho`.

The `Version` column reflects either:

- the version constraint declared in this repository, or
- the system dependency pattern currently expected by the project

System-installed dependencies may vary by machine and should be re-audited if bundled into release artifacts.

The dated [observed model inventory](docs/model-license-inventory.md) and its [machine-readable hashes](docs/model-license-inventory.json) separate original weights, conversions and runtime code. It records known identities and unresolved terms, not redistribution approval.

This is a maintained inventory, not release approval or a complete license bill of materials. Each row describes that component; it does not authorize closed-source distribution of AGPL-covered NextEcho code. Unknown converted-weight terms remain unresolved, even when the runtime or upstream code is permissively licensed.

| Dependency | Version | License | Commercial Use | Redistribution | Copyleft Risk | Notes |
|------------|---------|---------|----------------|----------------|---------------|-------|
| Flask | `>=3.0` | BSD-3-Clause | Yes | Yes, keep license notice | Low | Declared in `requirements.txt`. |
| pyannote.audio | `>=4.0,<5.0` | MIT | Yes | Yes, keep license notice | Low | Optional dependency for speaker diarization. |
| sherpa-onnx | `1.13.7` | Apache-2.0 | Yes | Yes, keep license and notices | Low | Optional local speaker diarization runtime; model weights are downloaded separately. |
| Resemblyzer | `0.1.4` | Apache-2.0 | Yes | Yes, keep license and notices | Low | Optional best-effort speaker embedding and enrollment backend. |
| 3D-Speaker ERes2Net ONNX | model artifact; exact conversion pending | Upstream code declares Apache-2.0; exact converted-weight terms not established | Unverified for this converted artifact | Do not bundle until exact provenance and terms are recorded | Unresolved | Optional sherpa-onnx embedding. [Upstream code LICENSE](https://github.com/modelscope/3D-Speaker/blob/main/LICENSE) alone does not establish this ONNX artifact's license. |
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
| pyannote speaker-diarization community model | Exact model-dependent identity required | See the pinned Community-1 candidate entry below; older conversions require their own evidence | Assess exact model and access conditions | No blanket permission for arbitrary derivatives | Unresolved per artifact | Code license, model-card license and gated-access agreement are separate evidence. |
| sherpa-onnx Pyannote segmentation 3.0 ONNX | model artifact; exact conversion pending | Upstream model and exact ONNX conversion terms remain to be recorded | Unverified for this converted artifact | Do not bundle until audited | Unresolved | Downloaded separately for local evaluation; conversion/runtime license does not replace model-weight terms. |

## Notes for Maintainers

- Keep this file updated when `requirements*.txt`, packaging scripts, or bundled assets change.
- If you start shipping app bundles, installers, or binary releases, replace `system dependency` entries with exact versions.
- FFmpeg requires the most caution. A user-installed FFmpeg binary and a bundled FFmpeg binary are different compliance situations.
# 本地模型运行时补充

- `mlx-qwen3-asr` 0.3.5：Apache-2.0；仅作为 Apple Silicon 可选运行时。
- Qwen3-ASR 0.6B / 1.7B：Apache-2.0；模型权重由用户主动下载并保存在本地。
- 社区 Qwen3-ASR 1.7B MLX 5-bit：保持 experimental；每次产物记录仓库、revision、量化、权重摘要和 MLX 版本，分发前重新核验模型卡与来源。
- Qwen3.5 9B：Apache-2.0；通过本机 Ollama 主动拉取并仅用于本地推理。

## Candidate CPU ASR and headless package (0.2.0)

中文简体派生层使用 `opencc-python-reimplemented==0.1.7`（Apache-2.0，来源：https://github.com/yichen0831/opencc-python）。使用包内 `t2s` 字典，不使用会替换地域词汇的 `tw2sp`；不修改外部模型环境，原始识别稿保留。

The application wheel does not bundle model weights, external environments, credentials or evaluation audio. Versions below come from the declared extras and the inspected local evaluation environment; they do not constitute a redistribution approval for every derivative model.

| Component | Version / identity | License evidence | Distribution in NextEcho |
| --- | --- | --- | --- |
| FunASR | 1.4.16 | Installed package metadata: MIT | Optional external worker dependency |
| PyTorch | 2.14.1 | Installed metadata SPDX: Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | Optional dependency; preserve its complete notices if distributed |
| kaldi-native-fbank | 1.22.3 | Installed metadata: Apache-2.0 | Optional dependency |
| sherpa-onnx | 1.13.8 ASR/native-CTC speaker path; legacy speaker extra 1.13.7 | Installed metadata: Apache-2.0 | Separate native-CTC and legacy speaker environments; native path additionally uses soundfile and numpy |
| SoundFile | >=0.12; tested 0.14.0 | Installed metadata: BSD-3-Clause; libsndfile has separate notices | Optional dependency |
| Fun-ASR-Nano-2512 | FunAudioLLM/Fun-ASR-Nano-2512 | Local model card declares Apache-2.0 | User-supplied weights only |
| SenseVoice ONNX | SenseVoiceSmall int8; conversion bundle `2365baeacb507f821a0c8120fcee3d484dba7a07` | Bundle LICENSE forwards to FunASR; original model card declares custom model-license, distinct from runtime MIT; exact original conversion checkpoint/terms unresolved | User-supplied weights only; not redistribution-cleared |
| Silero VAD ONNX | sherpa-onnx release asset silero_vad.onnx | Record exact upstream release and hash; audit the model's own terms before redistribution | User-supplied, not bundled |
| Local text models | Qwen3.5 2B/4B/9B; mlx-community/Qwen3.8-27B-4bit via Splash | Exact model cards and derivative notices remain per-profile release gates | User-managed services and weights only |

The validation VAD asset was fetched from the official k2-fsa/sherpa-onnx `asr-models` release. Its SHA-256 and calibration provenance are stored in ignored local validation artifacts. No download is triggered by ordinary `run`.

## 2026-10-20外部文本对齐候选

Qwen/Qwen3-ForcedAligner-0.6B-hf固定revision为`c07281df297b9905d24a508279258cccf987a064`；[官方模型卡](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf/tree/c07281df297b9905d24a508279258cccf987a064)声明Apache-2.0。原生safetensors未做转换，SHA-256为`00568245ceca5af1991d28562a75fe1ddc9bfeb041c27fda66947ea05c47fb86`；本机独立台账覆盖配置、处理器、词表、模板及模型卡，不随wheel/Skill/安装包分发权重。运行时复用外部Transformers5.19.0与PyTorch2.14.1，原生模型实现文件保留Apache-2.0声明；具体依赖的完整通知仍依其发行物。详见[候选验证说明](docs/forced-alignment-candidate.md)。

Community-1仅取得公开元数据：revision`3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`，官方模型卡声明CC-BY-4.0；受访问条件限制，尚未下载权重或完成本机依赖验证。不能把pyannote.audio代码许可代替权重许可，也不据此宣称所有社区派生权重可商用。现有其他转换物与项目许可矛盾的完整台账仍待任务13.7核验；本条不改变项目LICENSE。

## 2026-10-20本机Nemotron候选资产

NeMo-Speech.cpp官方v0.2.0代码commit `6a3ca369370782acee0dd155e34394ba60b920c4`的LICENSE为Apache-2.0；官方macOS aarch64 Metal归档SHA-256 `5cb02ba7c04f0b5585ce5cde9c830be5f0b7dfb4c83083c500109c381b4f2da9`，其第三方通知随原归档保留。模型`nvidia/Nemotron-3-Diarization` revision `f667ed73aee57d40cc39428eb768b4fd87a0a29e`的模型卡声明OpenMDW-1.1；Q8_0 GGUF SHA-256 `08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1`。转换脚本revision未知，不将运行时代码许可套用于权重或推断转换版本。精确台账和限制见[候选说明](docs/nemotron-candidate.md)。这些外部本机候选文件未作为Python依赖、wheel/Skill或安装包内容分发；本条未完成其余R20审计，不改变项目LICENSE，也不作未知转换物可商用声明。
