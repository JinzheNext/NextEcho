"""Build local Agent/Skill ZIPs from matching wheel and source distribution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import zipfile
from email.parser import BytesParser


SKILL_FILES = ('SKILL.md', 'scripts/run.py', 'scripts/bind_runtime.py', 'scripts/request_intent.py',
               'references/install.md', 'references/troubleshooting.md')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def runtime_inputs(wheel, sdist):
    with zipfile.ZipFile(wheel) as archive:
        metadata = [n for n in archive.namelist() if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('wheel元数据不唯一')
        info = BytesParser().parsebytes(archive.read(metadata[0]))
        if info['Name'] != 'nextecho':
            raise ValueError('wheel不是NextEcho')
        version = info['Version']
        code = {n: archive.read(n) for n in archive.namelist() if n.startswith('workbench/') or n == 'app.py'}
    with tarfile.open(sdist, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if (path.is_absolute() or '..' in path.parts or member.issym() or member.islnk()
                    or any(p in {'models', 'outputs', '.git', '__pycache__'} for p in path.parts)
                    or path.name.startswith(('cookies', '.env', 'skill-binding'))
                    or path.suffix.lower() in {'.wav', '.m4a', '.mp3', '.onnx', '.safetensors', '.gguf', '.pyc'}):
                raise ValueError('源码包包含不应分发的文件：' + member.name)
        roots = {Path(m.name).parts[0] for m in members}
        if len(roots) != 1:
            raise ValueError('源码包根目录不唯一')
        prefix = next(iter(roots)) + '/'
        lookup = {m.name: m for m in members if m.isfile()}
        for name, data in code.items():
            member = lookup.get(prefix + name)
            if member is None or archive.extractfile(member).read() != data:
                raise ValueError('wheel与源码包内容不一致：' + name)
        pkg = BytesParser().parsebytes(archive.extractfile(lookup[prefix + 'PKG-INFO']).read())
        if pkg['Version'] != version:
            raise ValueError('wheel与源码包版本不一致')
    return version


def write_zip(path, files, prefix):
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(prefix + '/' + name)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)


def package(root, wheel, sdist, output):
    version = runtime_inputs(wheel, sdist)
    skill = {}
    for relative in SKILL_FILES:
        path = root / 'skills/nextecho' / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError('Skill文件缺失或为符号链接：' + relative)
        skill[relative] = path.read_bytes()
    # A standalone Skill archive also carries the actual license and notices.
    for name in ('LICENSE', 'NOTICE'):
        skill[name] = (root / name).read_bytes()
    files = {'skills/nextecho/' + name: data for name, data in skill.items()}
    files.update({'install.py': (root / 'scripts/install_agent_bundle.py').read_bytes(),
                  'INSTALL.md': (root / 'docs/agent-bundle.md').read_bytes(),
                  'runtime/' + wheel.name: wheel.read_bytes(),
                  'runtime/constraints.txt': (root / 'requirements-agent.txt').read_bytes(),
                  'source/' + sdist.name: sdist.read_bytes()})
    for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_LICENSES.md', 'OPEN_SOURCE_COMPLIANCE.md'):
        files[name] = (root / name).read_bytes()
    revision = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout)
    manifest = {'schema_version': 1, 'version': version, 'wheel': 'runtime/' + wheel.name,
                'source': 'source/' + sdist.name, 'source_commit': revision, 'uncommitted_source': dirty,
                'models_included': False, 'files': {name: sha(data) for name, data in sorted(files.items())}}
    files['bundle.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    output.mkdir(parents=True, exist_ok=True)
    bundle_path = output / f'NextEcho-{version}-agent-bundle.zip'
    skill_path = output / f'nextecho-{version}-skill.zip'
    # Run the shipped installer verifier before producing either artifact.
    with tempfile.TemporaryDirectory(prefix='nextecho bundle check ') as tmp:
        stage = Path(tmp)
        for name, data in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        import importlib.util
        spec = importlib.util.spec_from_file_location('bundle_installer', root / 'scripts/install_agent_bundle.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.verify_bundle(stage)
    write_zip(bundle_path, files, f'NextEcho-{version}-agent-bundle')
    write_zip(skill_path, skill, 'nextecho')
    results = {p.name: {'path': str(p.resolve()), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
               for p in (bundle_path, skill_path, wheel, sdist)}
    (output / f'nextecho-{version}-artifacts.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--sdist', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('dist'))
    args = parser.parse_args()
    print(json.dumps(package(args.root.resolve(), args.wheel.resolve(), args.sdist.resolve(), args.output_dir.resolve()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
