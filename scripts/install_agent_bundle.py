"""Install an extracted NextEcho Agent bundle without launching models or a web UI."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from uuid import uuid4
import venv


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def defaults():
    if sys.platform == 'darwin':
        data = Path.home() / 'Library/Application Support/NextEcho'
    else:
        data = Path(os.environ.get('XDG_CONFIG_HOME', str(Path.home() / '.config'))) / 'nextecho'
    skills = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'skills/nextecho'
    return data, skills


def verify_bundle(root):
    manifest = json.loads((root / 'bundle.json').read_text())
    if manifest.get('schema_version') != 1 or not manifest.get('files'):
        raise ValueError('无效的bundle.json')
    if not re.fullmatch(r'\d+\.\d+\.\d+', str(manifest.get('version', ''))):
        raise ValueError('无效的运行时版本')
    files = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
             and '__pycache__' not in p.parts and p.name not in {'.DS_Store', 'bundle.json'}}
    if files != set(manifest['files']):
        raise ValueError('安装包包含清单之外的文件或缺少文件')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('安装包不能包含符号链接')
    for name, expected in manifest['files'].items():
        relative = Path(name)
        path = root / relative
        if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or digest(path) != expected):
            raise ValueError('安装包文件缺失、越界或校验失败：' + name)
    wheel = manifest['wheel']
    if wheel not in manifest['files'] or not wheel.endswith('.whl'):
        raise ValueError('缺少经过校验的运行时wheel')
    if 'skills/nextecho/SKILL.md' not in manifest['files']:
        raise ValueError('缺少Skill入口')
    return manifest


def checked(command):
    # Install output goes to stderr; stdout is one bounded JSON result.
    subprocess.run(command, check=True, stdout=sys.stderr, stderr=sys.stderr)


def install(root, *, data_dir, skill_dir, binding=None, config=None, replace=False, check_only=False):
    root = root.resolve()
    manifest = verify_bundle(root)
    if sys.version_info < (3, 11) or os.name != 'posix':
        raise ValueError('此Agent安装器需要macOS/Linux与Python>=3.11；Windows请按旧安装文档操作')
    data_dir, skill_dir = data_dir.expanduser().absolute(), skill_dir.expanduser().absolute()
    binding = (binding or data_dir / 'skill-binding.json').expanduser().absolute()
    config = config.expanduser().absolute() if config else None
    runtime = data_dir / 'agent-runtime' / manifest['version']
    for a, b in [(skill_dir, root), (runtime, root), (binding, root),
                 (runtime, skill_dir), (binding, skill_dir), (binding, runtime)]:
        if a.resolve().is_relative_to(b.resolve()) or b.resolve().is_relative_to(a.resolve()):
            raise ValueError('安装包、运行时、Skill与绑定路径不能相互嵌套')
    if any(p.is_symlink() for p in (skill_dir, binding, runtime)):
        raise ValueError('安装目的地不能是符号链接')
    if config and not config.is_file():
        raise ValueError('指定的配置文件不存在')
    existing = json.loads(binding.read_text()) if binding.exists() else {}
    if (skill_dir.exists() or binding.exists()) and not replace:
        raise ValueError('Skill或绑定已存在；升级请显式使用--replace，原文件会备份')
    if existing and (existing.get('schema_version') != 1
                     or not all(isinstance(existing.get(k, ''), str) for k in ('config', 'runtime_dir'))):
        raise ValueError('已有绑定无法安全继承，请先检查该文件')
    result = {'status': 'verified' if check_only else 'installed', 'version': manifest['version'],
              'skill_dir': str(skill_dir), 'binding': str(binding), 'runtime_python': str(runtime / 'bin/python'),
              'models_installed': False, 'next_action': '按Skill的references/install.md绑定模型worker，并运行doctor。'}
    if check_only:
        return result
    wheel_sha = manifest['files'][manifest['wheel']]
    marker = runtime / 'agent-install.json'
    created = False
    backups = []
    staged = None
    staged_binding = None
    skill_written = binding_written = False
    try:
        if runtime.exists():
            if not marker.is_file() or json.loads(marker.read_text()).get('wheel_sha256') != wheel_sha:
                raise ValueError('同版本运行时已存在但来源不同；请使用新的--data-dir，保留旧环境')
        else:
            runtime.parent.mkdir(parents=True, exist_ok=True)
            runtime.mkdir()
            created = True
            venv.EnvBuilder(with_pip=True).create(runtime)
            checked([str(runtime / 'bin/python'), '-m', 'pip', 'install', '--disable-pip-version-check',
                     '--constraint', str(root / 'runtime/constraints.txt'), str(root / manifest['wheel'])])
        probe = subprocess.run([str(runtime / 'bin/python'), '-m', 'workbench.cli', '--version'],
                               cwd=tempfile.gettempdir(), check=True, capture_output=True, text=True, timeout=15)
        if probe.stdout.strip() != f"nextecho {manifest['version']} protocol=1":
            raise ValueError('运行时版本或协议不匹配')
        if created:
            marker.write_text(json.dumps({'version': manifest['version'], 'wheel_sha256': wheel_sha}))
        skill_dir.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix='.nextecho-install-', dir=skill_dir.parent))
        shutil.copytree(root / 'skills/nextecho', staged, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        payload = {'schema_version': 1, 'python': str(runtime / 'bin/python'), 'runtime_version': manifest['version'],
                   'config': str(config) if config else existing.get('config', ''),
                   'runtime_dir': existing.get('runtime_dir') or str(data_dir / 'runtime')}
        binding.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, name = tempfile.mkstemp(prefix='.nextecho-binding-', dir=binding.parent)
        staged_binding = Path(name)
        with os.fdopen(fd, 'w') as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        backup_root = data_dir / 'install-backups' / uuid4().hex[:12]
        for target, role in ((skill_dir, 'skill'), (binding, 'binding.json')):
            if target.exists():
                backup_root.mkdir(parents=True, exist_ok=True)
                backup = backup_root / role
                target.rename(backup)
                backups.append((target, backup))
        staged.rename(skill_dir)
        skill_written = True
        staged = None
        staged_binding.replace(binding)
        binding_written = True
        staged_binding = None
        result['backups'] = [str(backup) for _, backup in backups]
        return result
    except BaseException:
        if binding_written:
            binding.unlink(missing_ok=True)
        if skill_written:
            shutil.rmtree(skill_dir)
        for target, backup in reversed(backups):
            backup.rename(target)
        if created:
            shutil.rmtree(runtime)
        raise
    finally:
        if staged:
            shutil.rmtree(staged)
        if staged_binding:
            staged_binding.unlink(missing_ok=True)


def main():
    data, skills = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--data-dir', type=Path, default=data)
    parser.add_argument('--skill-dir', type=Path, default=skills)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--check', action='store_true', help='只核对文件与安装目标，不安装、不联网')
    args = parser.parse_args()
    try:
        result = install(args.bundle, data_dir=args.data_dir, skill_dir=args.skill_dir,
                         binding=args.binding, config=args.config, replace=args.replace, check_only=args.check)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({'status': 'failed', 'message': str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
