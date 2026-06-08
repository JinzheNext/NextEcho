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

## Notes for Maintainers

- Keep this file updated when `requirements*.txt`, packaging scripts, or bundled assets change.
- If you start shipping app bundles, installers, or binary releases, replace `system dependency` entries with exact versions.
- FFmpeg requires the most caution. A user-installed FFmpeg binary and a bundled FFmpeg binary are different compliance situations.
