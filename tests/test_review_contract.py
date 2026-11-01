import copy
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave

from workbench.identity_review import write_identity_review
from workbench.io_utils import file_hash, write_json
from workbench.review import write_review
from workbench.review_contract import ANNOTATION_JS, inspect_audio, interval, validate_annotations


def wav(path, seconds=2, sample=b"\x01\x00"):
    with wave.open(str(path), "wb") as f:
        f.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        f.writeframes(sample * (16000 * seconds))


def payload(page, kind):
    return json.loads(page.read_text().split(f'<script id="{kind}-data" type="application/json">')[1].split('</script>')[0])


class ReviewContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="审核范围 ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.audio = self.root / "音频 #1.wav"
        wav(self.audio)
        self.hash = file_hash(self.audio)
        self.dialogue = self.root / "dialogue.json"
        self.data = {"transcription": [{"id": "r1", "text": "我是李继刚", "start_ms": 1500, "end_ms": 2100,
            "speaker_id": "s1", "speaker": "说话人1", "identity_status": "anonymous", "segment_ids": ["parent1"]}],
            "speakers": {"speakers": []}, "speaker_labels": {}, "alignment_audit": {}, "audio_sha256": self.hash,
            "raw_sha256": "1" * 64, "text_layer": "raw"}
        write_json(self.dialogue, self.data)
        self.page = self.root / "identity.html"

    def identity(self, **kwargs):
        write_identity_review(self.page, self.dialogue, self.audio, audio_origin_ms=1000, **kwargs)
        return payload(self.page, "identity")

    def polish(self, **overrides):
        row = {"id": "r1", "suite": "dialogue", "program": "节目", "speaker": "说话人1", "time_label": "1.5–2.1秒",
            "original": "金额36万元不能增加", "models": {"4B": {"text": "金额36万元不能增加。", "fallback": False}},
            "audio": str(self.audio), "media": {"sha256": self.hash, "origin_ms": 1000, "start_ms": 1500, "end_ms": 2100}}
        row.update(overrides)
        page = self.root / "polish.html"
        write_review(page, [row])
        return payload(page, "review")

    def test_nonzero_audio_origin_uses_real_local_seek_and_content_identity(self):
        data = self.identity()
        self.assertEqual(data['rows'][0]['audio_start'], .5)
        self.assertEqual(data['rows'][0]['audio_end'], 1.1)
        self.assertEqual(data['media']['duration_ms'], 2000)
        self.assertEqual(data['media']['sha256'], self.hash)
        self.assertEqual(data['review_items'][0]['source']['start_ms'], 1500)
        self.assertNotIn('#1', data['audio'])
        self.assertIn('%23', data['audio'])

    def test_wrong_or_missing_expected_hash_origin_rejected_before_replacing_page(self):
        self.identity()
        old = self.page.read_bytes()
        for changes in ({'audio_sha256': '0' * 64}, {'audio_sha256': None}, {'audio_origin_ms': 4}):
            with self.subTest(changes=changes):
                write_json(self.dialogue, {**self.data, **changes})
                with self.assertRaises(ValueError):
                    self.identity()
                self.assertEqual(self.page.read_bytes(), old)
        write_json(self.dialogue, self.data)
        with self.assertRaises(ValueError):
            write_identity_review(self.page, self.dialogue, self.audio)

    def test_same_length_audio_replacement_does_not_inherit_old_identity(self):
        self.identity()
        wav(self.audio, sample=b'\x02\x00')
        with self.assertRaises(ValueError):
            self.identity()

    def test_truncated_pcm_rejected_even_when_current_hash_is_declared(self):
        self.audio.write_bytes(self.audio.read_bytes()[:-100])
        with self.assertRaisesRegex(ValueError, '截断'):
            inspect_audio(self.audio, expected_sha256=file_hash(self.audio), origin_ms=0)

    def test_time_bounds_finite_and_half_open(self):
        media = inspect_audio(self.audio, expected_sha256=self.hash, origin_ms=1000)
        self.assertEqual(interval(media, 1000, 3000), (0, 2))
        for a, b in [(0, 1500), (1000, 3000.1), (1500, 1500), (float('nan'), 2000), (1000, float('inf')), (True, 2000)]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                interval(media, a, b)
        for origin in (None, True, -1, 1.5):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                inspect_audio(self.audio, expected_sha256=self.hash, origin_ms=origin)

    def test_existing_point_location_is_preserved_without_inventing_word_duration(self):
        self.data['transcription'][0].update(start_ms=1500, end_ms=1500,
            alignment_spans=[{'native_start_ms': None, 'native_end_ms': None, 'assignment_reason': 'missing_duration'}])
        write_json(self.dialogue, self.data)
        data = self.identity()
        row = data['rows'][0]
        self.assertEqual(row['audio_start'], row['audio_end'])
        self.assertEqual(row['location_kind'], 'review-point')
        self.assertIsNone(row['alignment_spans'][0]['native_end_ms'])
        self.assertEqual(data['review_items'][0]['source']['start_ms'], 1500)

    def test_path_copy_renderer_and_runtime_metadata_keep_identity_version(self):
        first = self.identity()
        copied = self.root / '副本'
        copied.mkdir()
        target = copied / self.audio.name
        shutil.copyfile(self.audio, target)
        copied_dialogue = copied / 'dialogue.json'
        write_json(copied_dialogue, {**self.data, 'resources': {'seconds': 100}, 'cached': True, 'created_at': 'later'})
        write_identity_review(copied/'page.html', copied_dialogue, target, audio_origin_ms=1000)
        later = payload(copied/'page.html', 'identity')
        self.assertEqual(first['fingerprint'], later['fingerprint'])
        self.assertEqual(first['review_items'], later['review_items'])
        changed = copy.deepcopy(self.data)
        changed['transcription'][0]['speaker'] = '另一人'
        write_json(self.dialogue, changed)
        self.assertNotEqual(first['fingerprint'], self.identity()['fingerprint'])

    def test_polish_media_local_only_explicit_and_old_offsets_cannot_disagree(self):
        data = self.polish()
        row = data['records'][0]
        self.assertEqual((row['audio_start'], row['audio_end']), (.5, 1.1))
        for bad in ({'media': None}, {'audio': 'https://example.org/a.wav'}, {'audio': 'javascript:alert(1)'},
                    {'audio': '//example.org/a.wav'}, {'audio_start': 1.5}, {'audio_end': 2.1}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.polish(**bad)

    def test_polish_versions_ignore_paths_but_change_for_model_text_and_identity(self):
        first = self.polish()
        self.assertEqual(first['fingerprint'], self.polish(program='改显示名称', time_label='显示不同')['fingerprint'])
        model = {'4B': {'text': '金额36万元不能增加。', 'fallback': False, 'model_identity': {'revision': 'new'}}}
        self.assertNotEqual(first['fingerprint'], self.polish(models=model)['fingerprint'])
        before = self.polish(models=model)
        model['4B']['model_identity']['token'] = 'do not export'
        model['4B']['model_identity']['local_path'] = '/private/model'
        after = self.polish(models=model)
        self.assertEqual(before['fingerprint'], after['fingerprint'])
        self.assertNotIn('token', after['review_items'][0]['source']['model_identity'])
        model['4B']['text'] = '不能增加预算。'
        self.assertNotEqual(first['fingerprint'], self.polish(models=model)['fingerprint'])

    def test_template_marker_and_script_ending_in_text_are_inert(self):
        hostile = '__ANNOTATION_JS__ </script> __IDENTITY_DATA__ __REVIEW_DATA__'
        self.data['transcription'][0]['text'] = hostile
        write_json(self.dialogue, self.data)
        self.assertEqual(self.identity()['rows'][0]['text'], hostile)
        self.assertEqual(self.polish(original=hostile)['records'][0]['original'], hostile)

    @unittest.skipUnless(shutil.which('node'), 'Node not available')
    def test_actual_js_export_roundtrip_whitelist_old_schema_and_stale_answers(self):
        data = self.identity()
        key = data['review_items'][0]['item_id']
        request = {'data': data, 'answers': {key+':verdict': 'pass', key+':note': '听辨判断',
                   key+':secret': 'not allowed', 'stale:verdict': 'pass', 'reviewer': 'stale'}}
        js = ANNOTATION_JS + '''
const input=JSON.parse(require('fs').readFileSync(0,'utf8'));
const bad=ReviewAnnotations.restore(JSON.stringify({schema_version:1,review_kind:'identity',answers:input.answers}),'identity');
if(Object.keys(bad).length)throw Error('old schema adopted');
const restored=ReviewAnnotations.restore(JSON.stringify(ReviewAnnotations.state(input.answers,'identity')),'identity');
console.log(JSON.stringify(ReviewAnnotations.exportReview(input.data,restored,'作者','2026-10-20T00:00:00Z')));
'''
        r = subprocess.run([shutil.which('node'), '-e', js], input=json.dumps(request), text=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
        self.assertEqual(r.returncode, 0, r.stderr)
        export = json.loads(r.stdout)
        self.assertEqual(validate_annotations(export, data), export)
        self.assertEqual(export['items'][0]['decision'], {'verdict': 'pass', 'note': '听辨判断'})
        self.assertNotIn('secret', r.stdout)
        export_file = self.root / 'author.json'
        write_json(export_file, export)
        checker = Path(__file__).resolve().parents[1] / 'scripts/verify_review_annotations.py'
        checked = subprocess.run([sys.executable, str(checker), '--page', str(self.page), '--annotations', str(export_file)],
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)['judged'], 1)
        self.assertEqual(json.loads(checked.stdout)['human_opinion_quality'], 'not-automatically-verified')
        for mutate in (lambda e: e.update(schema_version=1), lambda e: e.update(token='secret'),
                       lambda e: e['items'].append(e['items'][0]), lambda e: e['items'][0]['decision'].update(verdict='perfect'),
                       lambda e: e['items'][0]['source'].update(raw_sha256='changed'), lambda e: e.update(items=[])):
            bad = copy.deepcopy(export)
            mutate(bad)
            with self.assertRaises(ValueError):
                validate_annotations(bad, data)

    @unittest.skipUnless(shutil.which('node'), 'Node not available')
    def test_both_emitted_pages_contain_syntactically_valid_current_javascript(self):
        self.identity()
        self.polish()
        for page in (self.page, self.root/'polish.html'):
            script = re.findall(r'<script>(.*?)</script>', page.read_text(), re.S)[0]
            js = self.root / (page.stem + '.js')
            js.write_text(script)
            r = subprocess.run([shutil.which('node'), '--check', str(js)], capture_output=True, text=True, timeout=10)
            self.assertEqual(r.returncode, 0, r.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node not available')
    def test_identity_emitted_playback_uses_latest_request_local_time_and_duration_check(self):
        data = self.identity()
        # DOM/media test double: exercises emitted JS, not actual playback.
        data['audio_seconds'] = 10
        data['rows'][0].update(audio_start=3.5, audio_end=4, start_ms=4500, end_ms=5000)
        script = re.findall(r'<script>(.*?)</script>', self.page.read_text(), re.S)[0]
        test = r'''
const {script,data}=JSON.parse(require('fs').readFileSync(0,'utf8')),vm=require('vm');
class E{constructor(){this.value='';this.dataset={};this.children=[];this.readyState=0;this.duration=NaN;this.played=0;this.paused=0}append(...v){this.children.push(...v)}replaceChildren(){this.children=[]}load(){}play(){this.played++;return Promise.resolve()}pause(){this.paused++}}
const nodes={},get=s=>nodes[s]||(nodes[s]=new E());get('#identity-data').textContent=JSON.stringify(data);get('#filter').value='all';
const context={document:{querySelector:get,createElement:()=>new E()},localStorage:{getItem:()=>null,setItem:()=>{}},console};
vm.createContext(context);vm.runInContext(script,context);
const p=get('#player'),row=data.rows[0];context.listen(row);const stale=p.onloadedmetadata;
context.listen({...row,audio_start:5,audio_end:6,start_ms:6000,end_ms:7000});p.readyState=1;p.duration=10;
stale();if(p.played!==0)throw Error('stale request played');p.onloadedmetadata();
if(p.played!==1||p.currentTime!==3)throw Error('wrong origin/local seek');
p.currentTime=8;p.ontimeupdate();if(!p.paused)throw Error('failed bounded playback');
const before=p.played;p.duration=11;context.listen(row);if(p.played!==before||p.paused<2)throw Error('duration mismatch played');
console.log('latest request/local seek/bounded stop/mismatched duration passed');
'''
        r = subprocess.run([shutil.which('node'), '-e', test], input=json.dumps({'script': script, 'data': data}),
                           text=True, capture_output=True, timeout=10)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == '__main__':
    unittest.main()
