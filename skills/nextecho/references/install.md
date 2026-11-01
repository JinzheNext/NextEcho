# 安装与绑定

## 默认调用与个人候选试用

0.2.10支持在本机配置中保存用户已明确选择的候选试用组合，普通链接调用无需每次重复`--profile`。例如用户已同意SenseVoice+4B时，在本机配置中设置：

```toml
[agent]
default_profile = "fast"
approved_trial_models = ["sensevoice-int8", "qwen35-4b"]

[profiles.fast]
polish = "qwen35-4b"
```

仅这两个模型ID获得本机试用许可，不修改candidate/ready、不跳过模型文件校验或资源限制，不批准其他模型。未设置时保留原有候选门槛。确认用户选择后再保存；安装程序不会替用户同意所有候选。

auto分人遇到缺失的可选资产时，保留识别结果并继续润色，回执partial记录缺项；required仍阻断。台账损坏或hash不符始终阻断，不冒充依赖缺失。升级备份保存在用户数据目录的install-backups中，避免被宿主发现为第二个同名Skill。

## 安装包与模型绑定

0.2.12 Agent安装包包含程序wheel、对应源码、Skill、安装器及hash清单；先读取包内INSTALL.md，使用Python≥3.11运行`install.py --check`，再运行`install.py`。安装器不下载模型、不启动推理，已有安装需显式`--replace`并保留备份，可复用原有模型配置和任务目录。

Skill目录可以独立复制到支持Agent Skills的宿主。运行时需要Python≥3.11，并安装NextEcho程序包；Skill本身不包含模型或凭证。

单独安装Skill时，在独立虚拟环境安装wheel或源码包，例如`python -m pip install /path/to/NextEcho`；模型worker环境保持独立。先验证`python -m workbench.cli --version`的版本与protocol=1。0.2.10包含评论正文、Obsidian目录、阅读断词修复及用户确认姓名；普通持久请求、exact/refresh与分人精修最低0.2.5，按句润色补修最低0.2.6，跨副本姓名修订需≥0.2.8。

使用Skill脚本绑定实际解释器：

```text
python3 <skill目录>/scripts/bind_runtime.py --python <NextEcho环境>/bin/python
```

用户配置用`--config`选择，模型/任务runtime目录用`--runtime-dir`选择；不指定时用系统用户目录。绑定文件在Skill目录外，升级Skill保留绑定和模型。

模型准备入口：

```text
<NextEcho环境>/bin/python -m workbench.cli prepare --model-root <模型根目录> --worker-python <模型环境>/bin/python --vad-model <Silero-VAD文件> --no-web
```

该命令绑定现有权重，普通run不自动下载。缺依赖时按所选模型安装可选extras；Nano与SenseVoice都不强制要求whisper-cli。准确档目标Nano CPU，快速档SenseVoice CPU；balanced使用4B，fast使用2B。本期先交付2B/4B/9B评测；试用9B时显式加`--polish-model qwen35-9b`或在安装配置绑定`profiles.quality.polish="qwen35-9b"`。原quality的27B服务适配保留，但27B/35B MoE全量评测按作者要求延后，不自动启动。

候选文件就绪、资源校准、质量验收是不同状态。未通过产品验收的档案不设生产默认。当前新增完整后台流程首发macOS Apple Silicon；其他平台保留旧CLI兼容路径。

资源校准需要给prepare增加 `--profile fast --calibrate-audio <≤30秒16kHz单声道WAV>`，balanced/quality则测Nano。校准有独立进程资源限制，诊断路径由回执给出；不会自动提升ready。默认档案尚未验收时会MODEL_CANDIDATE，受控试用需显式指定profile。

卸载仅删除宿主技能目录；绑定、模型、任务记录分别保存在用户目录，不随Skill删除。完整参数与退出码见运行时仓库docs/headless-usage.md。

需要自动分人的安装，在prepare追加 `--speaker-model-root <segmentation.onnx与embedding.onnx所在目录>`。该选项绑定同一worker及模型根目录下的sherpa-sense-voice，不下载或修改外部环境。新对话路径使用sherpa-onnx1.13.8、soundfile、numpy；`nextecho doctor --capability speaker --json`分别检查声学权重、原生CTC权重和环境。时间锚仅用于对话归属，不代表precise字幕已经验收。运行条件ready与分人质量candidate分开；先回听小样和整期审核页，再判断是否适合日常使用。

## 显式Native分人候选

需要试用Native Metal分人时，在已绑定配置的`diarization.backend`选择`nemotron-metal-candidate`，并显式绑定`diarization.nemotron.asset_ledger`与`runtime_root`，以及独立CTC worker/模型。仅支持已固定验证的Apple Silicon macOS资产；不随Skill分发模型/录音/概率，不在一次转写中自动下载或安装。`attribution_profile`默认`native-padded`，`posterior-priority`为待人工验收的实验投影。正常调用仍返回紧凑回执；完整阶段可复用，Native中断恢复仅从头重放。未显式选择时不改变现有后端与默认质量门。
