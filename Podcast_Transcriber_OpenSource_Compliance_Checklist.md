# 播客转录助手（Mac App）开源协议与合规检查清单

## 1. Mac App 分发结论

### 不上架 App Store 是否可以分发？

可以。

最常见方式：

- DMG
- ZIP
- PKG

用户可以直接从：

- 个人网站
- GitHub Release
- Gumroad
- Lemon Squeezy
- 飞书文档下载链接

下载并安装。

### 没有 Apple Developer Account 是否能运行？

可以运行，但用户体验较差。

首次打开时，macOS Gatekeeper 可能提示：

- 无法验证开发者
- App 已损坏
- 来自身份不明的开发者

用户通常需要：

系统设置 → 隐私与安全性 → 仍要打开

或者：

右键 → 打开

然后确认运行。

### Apple Developer Account 的作用

费用：

- 99 美元/年

主要作用：

- App 签名（Code Signing）
- Notarization（苹果公证）
- 减少安全警告
- 提升用户信任度

建议：

MVP 阶段：

- 可以先不购买

开始收费后：

- 建议购买并完成签名与公证

---

## 2. 当前计划依赖

### Whisper

项目：

https://github.com/openai/whisper

协议：

MIT License

允许：

- 商业使用
- 修改
- 再分发
- 闭源

要求：

- 保留版权声明
- 保留 License

风险等级：

低

---

### yt-dlp

项目：

https://github.com/yt-dlp/yt-dlp

协议：

Unlicense

允许：

- 商业使用
- 修改
- 分发
- 售卖

要求：

- 建议保留项目声明

风险等级：

低

注意：

协议本身没有问题。

真正风险来自内容来源与平台服务条款。

---

### FFmpeg（高优先级检查）

项目：

https://ffmpeg.org

常见协议：

- LGPL
- GPL

必须重点确认：

你的具体打包方式

如果引入 GPL 组件：

可能触发更严格的开源义务。

风险等级：

中

建议：

让 Codex 重点检查。

---

## 3. 产品定位风险

推荐定位：

### 播客知识助手

输入：

- RSS
- 小宇宙链接
- 本地音频

输出：

- 转录
- 摘要
- 金句
- Markdown
- Obsidian 笔记

风险较低。

---

避免定位：

### Youtube Downloader

### 视频搬运工具

### 批量下载器

因为可能涉及：

- 平台服务条款
- 版权内容下载

风险较高。

---

## 4. API 成本风险

优先方案：

本地 Whisper

特点：

- 零边际成本
- 不依赖云服务
- 隐私友好

---

谨慎方案：

OpenAI API
Claude API
Gemini API

原因：

终身会员模式容易亏损。

示例：

售价：49 元

用户长期使用：

API 成本 > 收入

---

## 5. 建议保留的合规文件

项目根目录：

LICENSE

NOTICE

THIRD_PARTY_LICENSES.md

README.md

---

THIRD_PARTY_LICENSES.md 建议记录：

依赖名称

版本号

官方网站

License 类型

是否允许商用

是否要求开源

是否要求保留声明

---

## 6. 交给 Codex 的任务

请完成一次完整 Open Source Compliance Audit。

输出表格：

| Dependency | Version | License | Commercial Use | Redistribution | Copyleft Risk | Notes |
|------------|----------|----------|----------------|----------------|---------------|-------|

至少检查：

- Whisper
- Faster-Whisper
- yt-dlp
- FFmpeg
- PyInstaller
- Tauri（如使用）
- Electron（如使用）
- Python Runtime
- 所有 requirements.txt 中依赖

并最终回答：

1. 是否允许商业售卖
2. 是否允许闭源
3. 是否需要附带 License
4. 是否存在 GPL 污染风险
5. 是否存在专利风险
6. 是否存在平台条款风险
7. 是否存在中国地区合规风险

最终给出：

Green（可上线）
Yellow（需调整）
Red（存在重大风险）

并附整改建议。
