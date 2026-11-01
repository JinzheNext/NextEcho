from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


installer = module('install_agent_bundle')
packager = module('package_agent_bundle')


class AgentBundleTests(unittest.TestCase):
    def fixture(self, parent):
        root = parent / 'bundle'
        for name, data in {'skills/nextecho/SKILL.md': b'skill', 'runtime/runtime.whl': b'fixed wheel',
                           'runtime/constraints.txt': b'', 'install.py': b'installer'}.items():
            p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        manifest = {'schema_version': 1, 'version': '1.2.3', 'wheel': 'runtime/runtime.whl',
                    'files': {str(p.relative_to(root)): installer.digest(p) for p in root.rglob('*') if p.is_file()}}
        (root / 'bundle.json').write_text(json.dumps(manifest))
        return root, manifest

    def runtime(self, parent, manifest):
        data = parent / 'data'
        runtime = data / 'agent-runtime/1.2.3'
        runtime.mkdir(parents=True)
        (runtime / 'agent-install.json').write_text(json.dumps({'wheel_sha256': manifest['files'][manifest['wheel']]}))
        return data, runtime

    def probe(self):
        return patch.object(installer.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'nextecho 1.2.3 protocol=1\n', ''))

    def test_check_verifies_without_creating_install_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root, _ = self.fixture(parent)
            result = installer.install(root, data_dir=parent / 'data', skill_dir=parent / 'skills/nextecho', check_only=True)
            self.assertEqual(result['status'], 'verified')
            self.assertFalse((parent / 'data').exists())
            self.assertFalse((parent / 'skills').exists())

    def test_corruption_extra_files_and_path_traversal_fail_before_install(self):
        for kind in ('corruption', 'extra', 'traversal', 'version'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                parent = Path(tmp); root, manifest = self.fixture(parent)
                if kind == 'corruption':
                    (root / manifest['wheel']).write_bytes(b'corrupted')
                elif kind == 'extra':
                    (root / 'skills/nextecho/unlisted.py').write_text('unlisted')
                elif kind == 'traversal':
                    manifest['files']['../private.txt'] = 'x'
                    (root / 'bundle.json').write_text(json.dumps(manifest))
                else:
                    manifest['version'] = '../outside'
                    (root / 'bundle.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    installer.install(root, data_dir=parent / 'data', skill_dir=parent / 'skills/nextecho')
                self.assertFalse((parent / 'data').exists())

    def test_upgrade_requires_replace_preserves_config_and_creates_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root, manifest = self.fixture(parent)
            data, _ = self.runtime(parent, manifest)
            skill = parent / 'skills/nextecho'; skill.mkdir(parents=True)
            (skill / 'SKILL.md').write_text('old skill')
            old = {'schema_version': 1, 'python': '/old/python', 'config': '/models/settings.toml', 'runtime_dir': '/tasks/runtime'}
            binding = data / 'skill-binding.json'; binding.write_text(json.dumps(old))
            with self.assertRaises(ValueError):
                installer.install(root, data_dir=data, skill_dir=skill)
            with self.probe():
                result = installer.install(root, data_dir=data, skill_dir=skill, replace=True)
            saved = json.loads(binding.read_text())
            self.assertEqual(saved['config'], old['config'])
            self.assertEqual(saved['runtime_dir'], old['runtime_dir'])
            backups = [Path(p) for p in result['backups']]
            self.assertEqual(len(backups), 2)
            self.assertEqual((backups[0] / 'SKILL.md').read_text(), 'old skill')
            self.assertEqual(json.loads(backups[1].read_text()), old)
            self.assertTrue(all(p.is_relative_to(data / 'install-backups') for p in backups))
            self.assertFalse(list(skill.parent.glob('nextecho.backup-*')))

    def test_deployment_failure_rolls_back_skill_binding_and_keeps_existing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root, manifest = self.fixture(parent)
            data, runtime = self.runtime(parent, manifest)
            skill = parent / 'skills/nextecho'; skill.mkdir(parents=True)
            (skill / 'SKILL.md').write_text('old skill')
            binding = data / 'skill-binding.json'
            original = '{"schema_version":1,"config":"old","runtime_dir":"tasks"}'
            binding.write_text(original)
            with self.probe(), patch.object(Path, 'replace', side_effect=OSError('simulated disk failure')):
                with self.assertRaises(OSError):
                    installer.install(root, data_dir=data, skill_dir=skill, replace=True)
            self.assertEqual((skill / 'SKILL.md').read_text(), 'old skill')
            self.assertEqual(binding.read_text(), original)
            self.assertTrue(runtime.is_dir())
            self.assertFalse(list(skill.parent.glob('.nextecho-install-*')))

    def test_new_runtime_install_failure_removes_only_its_own_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root, _ = self.fixture(parent)
            data = parent / 'data'; models = data / 'models/existing.bin'
            models.parent.mkdir(parents=True); models.write_bytes(b'existing user asset')
            with patch.object(installer.venv.EnvBuilder, 'create', side_effect=RuntimeError('install failed')):
                with self.assertRaises(RuntimeError):
                    installer.install(root, data_dir=data, skill_dir=parent / 'skills/nextecho')
            self.assertFalse((data / 'agent-runtime/1.2.3').exists())
            self.assertEqual(models.read_bytes(), b'existing user asset')

    def test_runtime_with_different_wheel_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root, manifest = self.fixture(parent)
            data, runtime = self.runtime(parent, manifest)
            (runtime / 'agent-install.json').write_text('{"wheel_sha256":"different"}')
            with self.assertRaises(ValueError):
                installer.install(root, data_dir=data, skill_dir=parent / 'skills/nextecho', replace=True)
            self.assertTrue(runtime.is_dir())

    def test_bundle_rejects_symlink_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root, _ = self.fixture(parent)
            original = root / 'runtime/runtime.whl'
            original.rename(parent / 'outside.whl')
            original.symlink_to(parent / 'outside.whl')
            with self.assertRaises(ValueError):
                installer.verify_bundle(root)

    def test_install_paths_cannot_modify_bundle_or_contain_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); root, _ = self.fixture(parent)
            cases = [dict(data_dir=root / 'data', skill_dir=parent / 'skills/nextecho'),
                     dict(data_dir=parent / 'data', skill_dir=parent / 'data'),
                     dict(data_dir=parent / 'data', skill_dir=parent / 'skills/nextecho', binding=root / 'binding.json')]
            for case in cases:
                with self.subTest(case=case), self.assertRaises(ValueError):
                    installer.install(root, **case, check_only=True)

    def test_source_and_wheel_match_and_private_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp); wheel = parent / 'nextecho.whl'; sdist = parent / 'nextecho.tar.gz'
            source = b'__version__ = "1.2.3"\n'
            with zipfile.ZipFile(wheel, 'w') as archive:
                archive.writestr('nextecho-1.2.3.dist-info/METADATA', 'Name: nextecho\nVersion: 1.2.3\n')
                archive.writestr('workbench/__init__.py', source)
            def write_source(extra=False, mismatch=False):
                with tarfile.open(sdist, 'w:gz') as archive:
                    files = {'PKG-INFO': b'Name: nextecho\nVersion: 1.2.3\n',
                             'workbench/__init__.py': b'different' if mismatch else source}
                    if extra:
                        files['cookies.txt'] = b'synthetic credential'
                    for name, data in files.items():
                        info = tarfile.TarInfo('nextecho-1.2.3/' + name); info.size = len(data)
                        archive.addfile(info, io.BytesIO(data))
            write_source()
            self.assertEqual(packager.runtime_inputs(wheel, sdist), '1.2.3')
            write_source(mismatch=True)
            with self.assertRaises(ValueError):
                packager.runtime_inputs(wheel, sdist)
            write_source(extra=True)
            with self.assertRaises(ValueError):
                packager.runtime_inputs(wheel, sdist)


if __name__ == '__main__':
    unittest.main()
