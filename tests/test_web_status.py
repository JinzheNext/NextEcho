"""Execute the shipped status-label function; no browser/DOM claim."""

from pathlib import Path
import shutil
import subprocess
import unittest


@unittest.skipUnless(shutil.which("node"), "Node.js JavaScript verification")
class WebStatusTests(unittest.TestCase):
    def test_shipped_polling_waits_for_pipeline_and_stops_on_invalid_delivery(self):
        source = Path(__file__).resolve().parents[1] / "static/app.js"
        script = """const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const start=source.indexOf('async function waitForJob('),end=source.indexOf('\\nasync function resolveSources(',start);
assert(start>=0 && end>start);
async function scenario(rows,expectedError){
 let calls=0;const context={fetch:async()=>({ok:true,json:async()=>rows[Math.min(calls++,rows.length-1)]}),updateProgress:()=>{},setTimeout:(f)=>f()};
 vm.runInNewContext(source.slice(start,end),context);
 if(expectedError){await assert.rejects(context.waitForJob('id'),/blocked|cancelled|failed/);assert.strictEqual(calls,1);}
 else {const result=await context.waitForJob('id');assert.strictEqual(result,'final-pipeline');assert.strictEqual(calls,2);}
}
(async()=>{await scenario([{status:'polishing',result:'early-asr'},{status:'partial',result:'final-pipeline'}],false);
for(const status of ['blocked','cancelled','failed'])await scenario([{status,message:status}],true);})().catch(e=>{console.error(e);process.exitCode=1});
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script, str(source)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_running_partial_failed_and_ready_have_distinct_headlines(self):
        source = Path(__file__).resolve().parents[1] / "static/app.js"
        script = """const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const start=source.indexOf('function runStatusHeadline(');
const end=source.indexOf('\\nfunction renderResults(',start);
assert(start>=0 && end>start);
const context={};vm.runInNewContext(source.slice(start,end),context);
for(const phase of ['running','queued','polishing','packaging','unknown']) assert.strictEqual(context.runStatusHeadline(phase),'本次处理尚未完成');
for(const phase of ['failed','blocked','cancelled']) assert.strictEqual(context.runStatusHeadline(phase),'本次处理未完成');
assert.strictEqual(context.runStatusHeadline('partial'),'本次处理部分完成');
for(const phase of ['ready','completed']) assert.strictEqual(context.runStatusHeadline(phase),'本次转录已完成');
"""
        result = subprocess.run(
            [shutil.which("node"), "-e", script, str(source)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
