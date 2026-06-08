# Open Source Compliance Guide

This document turns the repository's open source compliance review into concrete release rules for this project.

## Current Compliance Status

Status: `Yellow`

Reason:

- The codebase is structurally suitable for commercial and closed-source distribution.
- The repository now documents third-party licenses and release risks.
- The repository now has a top-level `AGPL-3.0` license.
- FFmpeg and platform-download workflows require explicit release policy to avoid preventable GPL and platform-terms issues.

## Scope of This Project

The project is a local-first transcription workbench that:

- runs transcription locally with `whisper-cli`
- uses `ffmpeg` for media normalization and extraction
- optionally uses `yt-dlp` for page resolution and media download
- optionally uses `pyannote.audio` or a local clustering fallback for speaker attribution

The current macOS app wrapper does not bundle a full standalone runtime. It launches the existing local project environment and system tools.

## Release Rules

Use these rules when publishing the repository, binaries, or documentation.

### 1. Repository License

The repository now uses:

- `AGPL-3.0`

This is a strong copyleft license and is intentionally suitable for maintainers who want public source distribution while requiring source availability for modified network-deployed versions.

### 2. Third-Party Notice Files

Keep these files in the repository root:

- `THIRD_PARTY_LICENSES.md`
- `NOTICE`
- `OPEN_SOURCE_COMPLIANCE.md`

Update them whenever dependencies, packaging strategy, or bundled assets change.

### 3. FFmpeg Distribution Policy

FFmpeg is the biggest copyleft-sensitive dependency in this project.

Current project policy:

- The repository may depend on a user-installed FFmpeg binary.
- Do not assume every FFmpeg binary is LGPL-only.
- If you bundle FFmpeg in an app, installer, or release artifact, audit the exact build flags first.

High-risk case:

- FFmpeg builds with `--enable-gpl` or GPL codec/tooling can impose GPL redistribution obligations.

Safer case:

- Prefer an `LGPL-only` FFmpeg build if you need to redistribute FFmpeg with a commercial or closed-source product.

### 4. Download and Platform Terms Policy

`yt-dlp` is license-friendly, but platform terms are not the same thing as open source license rights.

Project policy:

- Position the product as a local transcription and knowledge workflow tool.
- Do not market it as a downloader, scraper bypass tool, or content rehosting tool.
- Tell users to process only content they own or are authorized to use.
- Do not claim support for bypassing DRM, access controls, or login protections.

### 5. Model and Attribution Policy

Whisper code and whisper.cpp are MIT-friendly, but model artifacts still need attribution hygiene.

Project policy:

- If model files are redistributed inside a release package, document their origin and applicable license in `THIRD_PARTY_LICENSES.md`.
- If pyannote community models are used or bundled, preserve the model card attribution requirements.

### 6. Data and Regional Compliance Policy

The tool is local-first, which lowers some hosted-service risk, but not all compliance risk.

Project policy:

- Treat uploaded media and transcripts as potentially sensitive personal data.
- Avoid default cloud upload behavior.
- Add user-facing wording that they should process only authorized media.
- If a future hosted version is added, re-audit for privacy policy, retention policy, ICP, and content moderation obligations.

## Dependency Summary

See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the maintained inventory.

Important conclusions:

- Commercial sale: generally allowed
- Closed-source distribution: generally allowed
- License notice inclusion: required
- GPL contamination risk: conditional, mainly around bundled FFmpeg builds
- Patent risk: codec-related, especially if shipping media stacks
- Platform terms risk: meaningful for YouTube/Bilibili and similar sources
- China compliance risk: moderate if the product expands beyond local offline usage

## Release Checklist

Use this checklist before tagging a public release:

1. Keep the repository `LICENSE` present and unchanged unless you intentionally relicense the project.
2. Review and refresh `THIRD_PARTY_LICENSES.md`.
3. Confirm whether release artifacts bundle `ffmpeg`, `yt-dlp`, `whisper-cli`, Python runtime, or model files.
4. If FFmpeg is bundled, record the exact build and confirm whether it is LGPL-only or GPL.
5. Keep `NOTICE` aligned with actual redistributed third-party components.
6. Check README wording to avoid positioning the tool as a platform-download bypass utility.
7. Verify that docs tell users to process only content they have rights to use.
8. Re-run this audit whenever packaging strategy or major dependencies change.

## Current Repository Decision

Based on the current implementation, the repository now has a defined project license. Remaining release work is mainly about packaging discipline, third-party notices, and platform-risk communication.
