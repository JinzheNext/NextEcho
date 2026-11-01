# NextEcho Agent 安装包

将完整ZIP解压后，把此文件交给可使用终端的Agent。本说明适用于0.2.12 Agent安装包，支持本机部署、复用已有模型配置和有备份的升级。

## 包内内容

- `skills/nextecho/`：可独立复制的Agent Skill及运行脚本。
- `runtime/nextecho-0.2.12-py3-none-any.whl`：NextEcho程序，不含Python解释器或模型。
- `runtime/constraints.txt`：本版核心Python依赖约束。
- `source/nextecho-0.2.12.tar.gz`：与wheel运行代码核对一致的对应源码，含构建和安装脚本。
- `install.py`、`bundle.json`：安装入口及SHA-256文件清单；校验用于发现文件损坏，不替代来源可信度判断。
- 项目许可证、署名与第三方许可证清单。

包不包含录音、模型权重、声纹样本、cookies、本机绑定或个人配置。首次安装需要联网获取核心Python依赖；模型及独立worker环境须另行准备。

## 安装

当前自动安装入口支持macOS/Linux POSIX、Python≥3.11；完整真实模型链路在Apple Silicon macOS上验证。Windows保留仓库的旧安装器，未据此声明新后台链路已验证。

在解压目录执行，先检查清单与目标：

```bash
python3.11 install.py --check
python3.11 install.py
```

可用Python3.12或更高版本替代。安装器默认使用`$CODEX_HOME/skills/nextecho`，未设置CODEX_HOME时使用`~/.codex/skills/nextecho`。程序安装到用户数据目录下独立的`agent-runtime/0.2.12`，绑定在Skill目录之外。

已有Skill或绑定时默认拒绝覆盖。明确升级时使用`--replace`，会备份旧Skill和绑定，并继承已有模型配置、任务runtime目录；旧程序环境、模型和任务文件保留。同版本运行时内容来源不同时拒绝复用，可选择新的`--data-dir`。

0.2.10起，备份写入用户数据目录的`install-backups/<批次>/`，位于技能发现目录之外，避免备份被当作另一个同名Skill加载。

其他支持Agent Skills的宿主，可用`--skill-dir /已确认的宿主技能目录/nextecho`指定安装位置。自定义位置示例：

```bash
python3.11 install.py --data-dir /path/to/NextEcho-data \
  --skill-dir /path/to/agent/skills/nextecho \
  --config /path/to/nextecho.toml
```

自定义`--data-dir`也会改变绑定路径。调用Skill时传`--binding /path/to/NextEcho-data/skill-binding.json`，或在Agent环境设置`NEXTECHO_SKILL_BINDING`为该文件；安装器不会修改shell或宿主全局配置。

安装成功只表示程序和Skill已就绪。按Skill的`references/install.md`绑定独立ASR/CTC/润色环境、模型和资源校准，再运行doctor；不要把安装成功当成模型或质量验收通过。安装器不会启动网站、推理、模型下载或修改已有模型服务。

## 给Agent的完整提示

> 请读取我提供的NextEcho Agent安装包中的INSTALL.md。用本机Python≥3.11先运行install.py --check，再安装Skill和隔离运行时。已有安装时保留模型、配置及历史任务，升级采用有备份的--replace。安装后核对版本与Skill绑定，按references/install.md检查模型/worker及doctor结果，缺少模型就明确报告具体缺项。不要自动下载大模型或启动网站。全部就绪后，我发媒体链接时使用本机转录，并只返回实际结果文件路径。

## 卸载与回滚

删除对应宿主内的nextecho Skill即可取消技能发现；不必删除模型和任务数据。需要回滚时恢复安装器返回的Skill/绑定备份，旧运行时一直保留。只有不再需要某版本且无任务使用它时才删除对应agent-runtime版本目录。

## 在源码目录重新构建

```bash
uv build --out-dir dist
python3.11 scripts/package_agent_bundle.py \
  --wheel dist/nextecho-0.2.12-py3-none-any.whl \
  --sdist dist/nextecho-0.2.12.tar.gz
```

构建脚本输出完整Agent包、纯Skill包和产物hash清单。完整包提供对应源码，纯Skill包需要另行安装运行时。不要打包整个工作目录；私有outputs、模型、缓存和凭证不属于分发包。
