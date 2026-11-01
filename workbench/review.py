"""Offline, explicit-layer review with literal Chinese character differences."""
from __future__ import annotations

import difflib
import html
import json
import unicodedata
from pathlib import Path

from .io_utils import atomic_write, file_hash
from .review_contract import ANNOTATION_JS, make_contract, make_item, model_reference, record_media


def text_change(before: str, after: str) -> dict:
    matcher = difflib.SequenceMatcher(None, before, after, autojunk=False)
    left, right, added, deleted = [], [], 0, 0
    for operation, a, b, c, d in matcher.get_opcodes():
        old, new = html.escape(before[a:b]), html.escape(after[c:d])
        if operation == "equal":
            left.append(old)
            right.append(new)
        else:
            if old:
                left.append('<del class="removed">' + old + '</del>')
                deleted += b - a
            if new:
                right.append('<ins class="added">' + new + '</ins>')
                added += d - c
    def content(text):
        return "".join(c for c in text if unicodedata.category(c)[0] in {"L", "N"})
    kind = "unchanged" if before == after else "punctuation" if content(before) == content(after) else "wording"
    return {"kind": kind, "added": added, "deleted": deleted,
            "left": "".join(left), "right": "".join(right)}


def write_review(path: Path, records: list[dict], *, metadata: dict | None = None) -> dict:
    """Records contain original text and actual adopted model text, never raw copies.

    Media URLs are supplied by the local artifact builder. No external scripts,
    fonts, network calls or model invocations occur in this page.
    """
    cases, items, media_cache = [], [], {}
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("审核样本ID重复")
    for record in records:
        case = {**record, "models": {}}
        case["audio"], media = record_media(record, path, media_cache)
        case["media"] = media
        if media:
            case["audio_start"] = (media["start_ms"] - media["origin_ms"]) / 1000
            case["audio_end"] = (media["end_ms"] - media["origin_ms"]) / 1000
        for name, result in record["models"].items():
            diff = text_change(record["original"], result["text"])
            source = {"row_id": record["id"], "suite": record.get("suite"), "model_key": name,
                      "media": media, "raw_sha256": record.get("raw_sha256"),
                      "model_identity": model_reference(result.get("model_identity"))}
            item = make_item("polish", source, [record["original"], result["text"], result.get("fallback", False),
                record.get("speaker"), record.get("source_hints"), result.get("pipeline")])
            items.append(item)
            case["models"][name] = {**result, "diff": diff,
                "signature": item["item_id"]}
        cases.append(case)
    payload = {**make_contract("polish", items), "metadata": metadata or {}, "records": cases}
    # Script data must not terminate the script element, even for hostile media.
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    for (audio_path, expected, _origin) in media_cache:
        if file_hash(Path(audio_path)) != expected:
            raise ValueError("审核音频在页面构建时改变")
    atomic_write(path, PAGE.replace("__ANNOTATION_JS__", ANNOTATION_JS).replace("__REVIEW_DATA__", encoded))
    return {"records": len(cases), "fingerprint": payload["fingerprint"],
            "changed_variants": sum(v["diff"]["kind"] != "unchanged" for c in cases for v in c["models"].values())}


PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NextEcho · 原稿与润色差异审核</title>
<style>
:root{--ink:#192a39;--muted:#5a6c7a;--line:#dce4e9;--blue:#205d85;--bg:#f3f6f8}*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}
main{max-width:1440px;margin:auto;padding:28px}h1{font-size:28px;margin:0 0 8px}p{margin:8px 0}.muted,small{color:var(--muted)}
header{position:sticky;top:0;z-index:2;background:#fffffff5;backdrop-filter:blur(12px);border-bottom:1px solid var(--line);padding:12px 28px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;max-width:1384px;margin:auto}
button,select,input,textarea{font:inherit;border:1px solid #bfccd5;border-radius:7px;background:white;color:var(--ink);padding:7px 11px}
button{cursor:pointer}button:hover{border-color:var(--blue)}button.active{background:var(--blue);border-color:var(--blue);color:white}
input[type=search]{min-width:220px;flex:1}textarea{width:100%;min-height:64px;resize:vertical}label{display:inline-flex;gap:6px;align-items:center}
.intro{padding:18px 22px;border:1px solid #cbdde9;background:#ecf4fa;border-radius:10px;margin:20px 0}
.legend{display:flex;gap:18px;flex-wrap:wrap}ins,.added{background:#d6f5e2;color:#084c2a;text-decoration:none;border-bottom:2px solid #239a57}
del,.removed{background:#ffe2df;color:#8b2923;text-decoration:line-through;text-decoration-thickness:1px}
article{background:white;border:1px solid var(--line);border-radius:12px;margin:22px 0;overflow:hidden;scroll-margin-top:230px}
.cardhead{padding:16px 22px;background:#fbfcfd;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:14px;align-items:start}
.cardhead h2{font-size:18px;margin:0}.badge{display:inline-block;border-radius:5px;padding:1px 9px;font-size:13px;background:#e5edf3;margin-right:6px}.speaker{background:#e0edf9;color:#164e7a;font-weight:650}
.speaker2{background:#eee7f9;color:#633b94}.warning{background:#fff1d2;color:#865600}.same{background:#e9edef;color:#52626c}
.comparison{display:grid;grid-template-columns:1fr 1fr}.column{padding:18px 22px;min-width:0}.column+.column{border-left:1px solid var(--line)}
.column h3{font-size:14px;margin:0 0 12px;color:var(--muted)}.diff{line-height:1.95;white-space:pre-wrap;overflow-wrap:anywhere;font-size:17px}
.single .comparison{display:block}.single .original{display:none}.single .column+.column{border-left:0}
.assessment{padding:14px 22px;border-top:1px solid var(--line);display:grid;gap:12px}.choices{display:flex;gap:15px;flex-wrap:wrap}
.context{background:#f9fbfc}.context .cardhead{padding:10px 22px}.context .assessment{padding:10px 22px}.contexttext{padding:10px 22px;white-space:pre-wrap;line-height:1.9}
.audio{max-width:1384px;margin:8px auto 0;display:flex;gap:12px;align-items:center}audio{height:34px;max-width:520px;width:100%}
#count,#progress{font-size:14px;color:var(--muted)}.empty{padding:50px;text-align:center}.counter{font-variant-numeric:tabular-nums}
details{padding:12px 0}summary{cursor:pointer}.notice{border-left:3px solid #cf9d32;padding-left:12px}
@media(max-width:760px){main{padding:14px}header{padding:10px}.comparison{display:block}.column+.column{border-left:0;border-top:1px solid var(--line)}h1{font-size:23px}.cardhead{display:block}.diff{font-size:16px}.audio{flex-wrap:wrap}.controls{gap:7px}}
@media print{header,.assessment,.listen{display:none}article{break-inside:avoid}body{background:white}main{max-width:none}}
</style></head><body>
<main><h1>原稿与润色，差异在哪里</h1><p class="muted">红色删除 · 绿色新增或替换 · 未改文字不着色。原稿始终保留，回退和未修改单独标记。</p>
<div class="intro" id="intro"><b>已修正旧页误显示原稿的问题。</b>本页展示实际采用的润色文字，未修改的内容明确标出，不再要求重复判断。
<details><summary>怎么审：先听谁说，再看改了什么</summary><p>“新对话稿”先在完整语境中识别，再用真实语音时间标记和声学分人结果对齐。对话人编号仅代表本片段内的声音，不猜真实身份；交叠发言或归属不清的内容保留待核对。点击回听，核对声音和文字。左侧是本轮润色前的新识别稿，右侧是规则清理与所选模型共同生成的整理稿；不把换识别模型产生的差异计作润色。删除机械口误、补标点可以；数字、否定、条件、因果和观点不得改变。</p><p>“旧60段修正对照”使用旧原稿与当时实际采用的模型结果，保留混合说话人切片，用于复核旧评测。短句集合没有对应对话音频。已通顺的文字允许不改；修改数量不代替质量判断。未看过的保留未审核，批注保存在本机，可随时导出。</p></details></div>
</main>
<header><div class="controls">
<label>集合 <select id="suite"></select></label><div id="models"><button data-model="2B">2B</button> <button data-model="4B">4B</button> <button data-model="9B">9B</button></div>
<select id="filter" aria-label="差异筛选"><option value="context">按对话看 · 未改仅作上下文</option><option value="changed">只看有修改</option><option value="wording">只看文字调整</option><option value="fallback">只看原文回退</option><option value="unchanged">只看完全未改</option><option value="all">全部</option></select>
<input id="search" type="search" placeholder="搜索原文、样本编号或节目"><label><input id="inline" type="checkbox">只看修改稿</label>
<input id="reviewer" placeholder="审核人姓名" size="8"><button id="export">导出审核结果</button><span id="count" role="status"></span>
</div><div class="audio"><audio id="player" controls preload="none"></audio><small id="playing">点击条目“回听本段”，核对原声与说话人。</small></div><div class="controls"><span id="availability"></span></div><div class="controls"><span id="progress"></span></div></header>
<main><div class="legend"><span><del>原稿中删除的内容</del></span><span><ins>润色中新增或替换的内容</ins></span><span class="muted">“仅标点”与“文字调整”分别统计，不用变化数量代替质量判断。</span></div><div id="cards"></div>
<details><summary>数据来源与版本</summary><pre id="meta" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details></main>
<script id="review-data" type="application/json">__REVIEW_DATA__</script>
<script>
'use strict';
__ANNOTATION_JS__
const data=JSON.parse(document.querySelector('#review-data').textContent),$=s=>document.querySelector(s),key='nextecho-annotations-v3-polish';
let saved={},storageAvailable=true;try{saved=ReviewAnnotations.restore(localStorage.getItem(key),'polish')}catch(e){storageAvailable=false}let model=data.metadata.default_model||'9B';
const modelLabels=data.metadata.model_labels||{'2B':'2B','4B':'4B','9B':'9B'};
$('#models').replaceChildren();for(const [value,label]of Object.entries(modelLabels)){const b=document.createElement('button');b.dataset.model=value;b.textContent=label;$('#models').append(b)}
if(data.metadata.title){document.querySelector('h1').textContent=data.metadata.title;document.title=data.metadata.title}
if(data.metadata.description)$('#intro').textContent=data.metadata.description;
const suites=data.metadata.suites||[['dialogue','新对话稿 · 原音频分说话人'],['business60','旧60段修正对照 · 原切片'],['cs2w100','100条短句 · 无对话音频']];
for(const [value,label] of suites){if(data.records.some(r=>r.suite===value)){const o=document.createElement('option');o.value=value;o.textContent=label;$('#suite').append(o)}}
$('#filter').value=$('#suite').value==='dialogue'?'context':'changed';
$('#reviewer').value='作者';$('#meta').textContent=JSON.stringify({...data.metadata,fingerprint:data.fingerprint},null,2);
const names={unchanged:'完全未改',punctuation:'仅标点 / 空白调整',wording:'文字调整'};
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function persist(){try{localStorage.setItem(key,JSON.stringify(ReviewAnnotations.state(saved,'polish')));storageAvailable=true}catch(e){storageAvailable=false}}
function reviewKey(row,result){return result.signature}
function choose(label,options,value,changed){const wrapper=el('label',label+' '),s=document.createElement('select');for(const [v,t]of options){const o=el('option',t);o.value=v;s.append(o)}s.value=value||'pending';s.onchange=()=>changed(s.value);wrapper.append(s);return wrapper}
function status(){const states=data.review_items.map(i=>ReviewAnnotations.clean(i,saved,'polish')),judged=states.filter(v=>v.meaning!=='pending').length,voices=states.filter(v=>v.voice_flag==='flagged').length;$('#progress').textContent=judged+' 项文字判断 / '+voices+' 条分人问题。'+(storageAvailable?'已保存到本机，完成一批可导出。':'当前浏览器不能持久保存，请保留该页并及时导出。')}
let playbackEnd=null,playbackRequest=0;
$('#player').ontimeupdate=()=>{if(playbackEnd!==null&&$('#player').currentTime>=playbackEnd)$('#player').pause()};
function listen(row){const token=++playbackRequest,player=$('#player'),start=Math.max(0,row.audio_start-1),end=Math.min(row.media.duration_ms/1000,row.audio_end+1);playbackEnd=end;
 const play=()=>{if(token!==playbackRequest)return;if(!Number.isFinite(player.duration)||Math.abs(player.duration-row.media.duration_ms/1000)>0.05){player.pause();$('#playing').textContent='播放器时长与构建校验不符，停止回听';return}player.currentTime=start;player.play().catch(()=>{$('#playing').textContent='请点击播放器开始播放。'})};
 player.onloadedmetadata=play;if(player.dataset.media===row.audio){if(player.readyState>=1)play()}else{player.dataset.media=row.audio;player.src=row.audio;player.load()}
 $('#playing').textContent=row.program+' · '+row.time_label+' · '+row.speaker;
}
function render(){const suite=$('#suite').value,filter=$('#filter').value,q=$('#search').value.trim().toLowerCase(),root=$('#cards');root.replaceChildren();let n=0,changed=0,unchanged=0,wording=0,fall=0;
 const available=Object.keys(modelLabels).filter(m=>data.records.some(r=>r.suite===suite&&r.models[m]));$('#availability').textContent=(data.metadata.availability_notes||{})[suite]||('本集合已完成：'+available.map(m=>modelLabels[m]).join('、')+(suite==='dialogue'&&!available.includes('9B')&&data.records.some(r=>r.suite==='business60'&&r.models['9B'])?'。9B请切到“旧60段修正对照”查看；新对话稿本轮未完成9B。':''));
 for(const b of document.querySelectorAll('[data-model]')){b.classList.toggle('active',b.dataset.model===model);b.disabled=!data.records.some(r=>r.suite===suite&&r.models[b.dataset.model]);b.title=b.disabled?'此集合没有这个版本，请切换集合查看已完成结果':''}
 for(const row of data.records){if(row.suite!==suite)continue;const r=row.models[model];if(!r)continue;const kind=r.diff.kind;
 if(kind==='unchanged')unchanged++;else changed++;if(kind==='wording')wording++;if(r.fallback)fall++;
 if(filter==='changed'&&kind==='unchanged'||filter==='wording'&&kind!=='wording'||filter==='unchanged'&&kind!=='unchanged'||filter==='fallback'&&!r.fallback)continue;
 if(q&&!(row.original+' '+r.text+' '+row.id+' '+row.program).toLowerCase().includes(q))continue;n++;
 const article=el('article'),head=el('div',undefined,'cardhead'),info=el('div');info.append(el('h2',row.program+' · '+row.time_label));
 info.append(el('span',row.speaker,'badge '+(row.overlap||row.uncertain?'warning':'speaker '+(row.speaker.endsWith('2')?'speaker2':''))),el('small',' '+row.id));head.append(info);
 const badges=el('div');badges.append(el('span',names[kind],'badge '+(kind==='unchanged'?'same':'')));
 if(r.fallback)badges.append(el('span',r.partial_fallback?'部分片段模型未采用 · 已保留输入稿':r.rule_cleaned?'模型未采用 · 仅保留规则清理':'模型未采用 · 保留原稿','badge warning'));
 if(row.uncertain&&row.speaker.startsWith('对话人'))badges.append(el('span','有交界需回听核对','badge warning'));
 if(r.protection_errors&&r.protection_errors.length)badges.append(el('span','保护检查有问题 · 请核对','badge warning'));
 if(row.audio){const b=el('button','▶ 回听本段','listen');b.onclick=()=>listen(row);badges.append(b)}head.append(badges);article.append(head);
 const columns=el('div',undefined,'comparison'),left=el('section',undefined,'column original'),right=el('section',undefined,'column');
 left.append(el('h3',(row.original_label||(row.suite==='dialogue'?'按原声分角色的新识别稿':'原ASR稿'))+' · 红色为删除或替换部分'));right.append(el('h3',(r.pipeline||(modelLabels[model]||model)+' 实际润色')+' · 绿色为新增或替换部分'));
 const old=el('div',undefined,'diff'),edit=el('div',undefined,'diff');old.innerHTML=r.diff.left;edit.innerHTML=r.diff.right;left.append(old);right.append(edit);columns.append(left,right);
 if(kind==='unchanged'){article.classList.add('context');article.append(el('div',row.original,'contexttext'))}else article.append(columns);
 const a=el('div',undefined,'assessment');a.append(el('small','实际字符差异：新增 '+r.diff.added+' / 删除 '+r.diff.deleted+'。'+(kind==='unchanged'?'没有文字变化，直接跳过本项质量判断。':'')));
 if(row.alignment_note)a.append(el('p',row.alignment_note,'notice'));
 if(kind!=='unchanged'){
  const k=reviewKey(row,r),choices=el('div',undefined,'choices');
  choices.append(choose('语义',[['pending','未审核'],['pass','原意保留'],['fail','有改义 / 遗漏 / 新增'],['uncertain','需回听核对']],saved[k+':meaning'],v=>{saved[k+':meaning']=v;persist();status()}));
  choices.append(choose('可读性',[['pending','未审核'],['better','更顺、更清楚'],['same','没有明显改善'],['worse','比原稿更差']],saved[k+':readability'],v=>{saved[k+':readability']=v;persist();status()}));a.append(choices);
  const note=document.createElement('textarea');note.placeholder='指出具体改错处、说话人问题或你的偏好；不确定时可记“需回听”。';note.maxLength=4000;note.value=saved[k+':note']||'';note.oninput=()=>{saved[k+':note']=note.value;persist()};a.append(note);
 }
 if(row.suite==='dialogue'){
  const k=reviewKey(row,r),label=el('label',''),check=document.createElement('input'),note=document.createElement('textarea');check.type='checkbox';check.checked=saved[k+':voice_flag']==='flagged';label.append(check,el('span','分人有误 / 需要核对（可选）'));note.placeholder='例如：这句话应属于对话人2，或这里两人同时讲话。';note.maxLength=4000;note.value=saved[k+':voice_note']||'';note.hidden=!check.checked;check.onchange=()=>{saved[k+':voice_flag']=check.checked?'flagged':'pending';note.hidden=!check.checked;persist();status()};note.oninput=()=>{saved[k+':voice_note']=note.value;persist()};a.append(label,note);
 }
  if(row.source_hints){const hints=row.source_hints.hints||[],terms=row.source_hints.terminology||[],confirmed=row.source_hints.user_confirmed_terms||[];if(hints.length||terms.length||confirmed.length){const details=el('details');details.style.padding='12px 22px';details.append(el('summary','本段实际提供的参考线索（只读，不代表原声事实）'));if(terms.length)details.append(el('p','官方名称：'+terms.map(t=>t.term).join('、')));if(confirmed.length)details.append(el('p','用户确认术语（拼写线索）：'+confirmed.join('、')));for(const hint of hints)details.append(el('p',(hint.context_ref||'')+' · '+hint.kind+'：'+hint.text));article.append(details)}}
  article.append(a);root.append(article);
 }
 root.classList.toggle('single',$('#inline').checked);$('#count').textContent='本集合 '+model+'：有修改 '+changed+'（文字 '+wording+'） / 完全未改 '+unchanged+' / 回退 '+fall+'；当前显示 '+n;
 if(!n)root.append(el('div','当前筛选没有匹配条目。可以切换模型或选择“全部”。','empty'));status();
}
$('#suite').onchange=()=>{$('#filter').value=$('#suite').value==='dialogue'?'context':'changed';if(!data.records.some(r=>r.suite===$('#suite').value&&r.models[model]))model=Object.keys(modelLabels).find(m=>data.records.some(r=>r.suite===$('#suite').value&&r.models[m]));render()};$('#filter').onchange=render;$('#search').oninput=render;$('#inline').onchange=render;
for(const b of document.querySelectorAll('[data-model]'))b.onclick=()=>{model=b.dataset.model;render()};
$('#reviewer').maxLength=80;
$('#export').onclick=()=>{const reviewer=$('#reviewer').value.trim();if(!reviewer){alert('请先填写审核人姓名或代号');return}
 let result;try{result=ReviewAnnotations.exportReview(data,saved,reviewer,new Date().toISOString())}catch(e){alert(e.message);return}const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='nextecho-diff-review-'+reviewer+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
render();
</script></body></html>'''
