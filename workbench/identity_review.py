"""Offline human review of measured dialogue and identity evidence."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

from .io_utils import atomic_write, file_hash
from .review_contract import ANNOTATION_JS, inspect_audio, interval, make_contract, make_item


def write_identity_review(path: Path, dialogue_path: Path, audio: Path, *, semantic_review: Path | None = None,
                          audio_origin_ms: int | None = None, expected_audio_sha256: str | None = None):
    data = json.loads(dialogue_path.read_text())
    rows = data["transcription"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("对话条目ID重复")
    media = inspect_audio(audio, expected_sha256=expected_audio_sha256 or data.get("audio_sha256"),
                          origin_ms=audio_origin_ms if audio_origin_ms is not None else data.get("audio_origin_ms"))
    if data.get("audio_sha256") and data["audio_sha256"] != media["sha256"]:
        raise ValueError("对话声明的音频hash与试听音频不符")
    if data.get("audio_origin_ms") is not None and data["audio_origin_ms"] != media["origin_ms"]:
        raise ValueError("对话声明的origin与试听音频不符")
    items = []
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("审核条目ID必须为非空字符串")
        start, end = interval(media, row.get("start_ms"), row.get("end_ms"), allow_point=True)
        location_kind = "review-point" if start == end else "review-range"
        source = {"row_id": row["id"], "raw_sha256": data.get("raw_sha256"),
                  "speaker_payload_sha256": data.get("speaker_payload_sha256"),
                  "recording_id": data.get("recording_id"),
                  "media": media, "start_ms": row["start_ms"], "end_ms": row["end_ms"], "location_kind": location_kind}
        content = {k: row.get(k) for k in (
            "text", "speaker_id", "speaker_uuid", "speaker", "identity_status", "segment_ids",
            "alignment_spans", "alignment_summary", "native_anchor_chars", "timing_precision", "annotation_kind",
        )}
        item = make_item("identity", source, [content, data["speakers"], data["speaker_labels"], data.get("text_layer_sha256")])
        items.append(item)
        row.update(item_version=item["item_id"], audio_start=start, audio_end=end, location_kind=location_kind)
    def relative(target):
        return quote(os.path.relpath(target.resolve(), path.parent.resolve()), safe="/")
    payload = {**make_contract("identity", items), "rows": rows, "speakers": data["speakers"],
               "labels": data["speaker_labels"], "audit": data["alignment_audit"],
               "acoustic_audit": data.get("acoustic_audit", {}),
               "audio": relative(audio), "semantic_review": relative(semantic_review) if semantic_review else "",
               "audio_seconds": media["duration_ms"] / 1000, "media": media, "text_layer": data.get("text_layer", "unknown"),
               "source": {"raw_sha256": data.get("raw_sha256"), "audio_sha256": file_hash(audio),
                          "dialogue_sha256": file_hash(dialogue_path),
                          "speaker_payload_sha256": data.get("speaker_payload_sha256"),
                          "identity_payload_sha256": data.get("identity_payload_sha256"),
                          "recording_id": data.get("recording_id"), "quality_status": "human_review_pending"}}
    encoded = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    if file_hash(audio) != media["sha256"]:
        raise ValueError("审核音频在页面构建时改变")
    atomic_write(path, PAGE.replace("__ANNOTATION_JS__", ANNOTATION_JS).replace("__IDENTITY_DATA__", encoded))
    return {"rows": len(rows), "fingerprint": payload["fingerprint"]}


PAGE = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NextEcho · 说话人与身份审核</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#203244;font:16px/1.7 system-ui,-apple-system,"PingFang SC",sans-serif}
main,header>div{max-width:1160px;margin:auto}main{padding:24px}h1{font-size:28px;margin:0}h2{font-size:20px}p{margin:8px 0}
header{position:sticky;top:0;background:#fffffff5;border-bottom:1px solid #d5dfe7;padding:12px 24px;z-index:2}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center}button,select,input,textarea{font:inherit;padding:7px 10px;border:1px solid #bdcbd5;border-radius:6px;background:white;color:inherit}button{cursor:pointer}button:hover{border-color:#25689c}
input[type=search]{flex:1;min-width:200px}textarea{width:100%;min-height:58px}article,.intro{padding:18px 22px;background:white;border:1px solid #d5dfe7;border-radius:10px;margin:18px 0}
.intro{background:#edf5fb}.muted,small{color:#5c7183}.badge{display:inline-block;padding:2px 9px;border-radius:5px;background:#e2edf8;margin-right:8px}.warning{background:#fff0c9}.text{white-space:pre-wrap;font-size:18px;padding:10px 0}details{margin:12px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}
audio{height:36px;width:min(100%,560px);margin-top:8px}.cardhead{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}.judgment{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0}a{color:#1d6398}#progress{font-size:14px}
@media(max-width:650px){main{padding:12px}header{padding:10px}h1{font-size:23px}article{padding:14px}}
</style></head><body>
<main><h1>先确认是谁在说，再确认叫什么</h1><p class="muted">声学簇 · 原生 CTC 时间锚 · 本轮实际文字层 · 人工审核待完成</p><p id="scope" class="muted"></p>
<div class="intro"><p><b>你自己审核即可。</b>先听每个声音簇分布在录音前、中、后的样例，确认同一声音有没有被拆成多个编号；再核对“我是……”附近的声音与姓名。标题和 shownotes 的人物名单只是候选。</p>
<p>“声音归属待核”表示目前没有可用的单一说话人 ID；“姓名未确认”表示已有录音内声学簇、但没有可靠自然人姓名；“预测交叠”仍需听辨，不能直接当作真实重叠。模型可能拆错簇或合错人，原生锚覆盖率不能当作分人准确率。</p>
<p>优先审核自我介绍、短应答、换人交界、长时间后重返和片尾。回听按钮多播放前后各 2 秒；显示时间是对话定位边界，不能当作精确词级字幕。</p>
<p>页面把“没有原生时间锚”单独筛出；回听包络只用于定位，不代表该文字有准确词级起止。当前页面来源、文字层和音频 hash 会绑定到导出结果。</p>
<p id="semantic"></p><p>选择判断并填写修正意见，结果仅保存在当前浏览器；完成一批就导出。导出不会自动改写原稿或确认整期通过。</p></div>
<details open><summary>声音簇与姓名证据</summary><div id="speakers"></div></details></main>
<header><div class="controls"><select id="filter" aria-label="审核范围"><option value="all">全部对话</option><option value="unknown-speaker">声音归属待核</option><option value="unknown-name">姓名未确认</option><option value="overlap">预测交叠 / 多活动</option><option value="no-anchor">无原生时间锚</option><option value="missing-duration">原生区间缺失</option><option value="resolved">已具名</option><option value="flagged">已标记问题</option></select>
<input id="search" type="search" placeholder="搜索姓名、文字或条目编号"><input id="reviewer" placeholder="审核人" value="作者" size="6"><button id="export">导出审核结果</button></div>
<div><audio id="player" controls preload="none"></audio><span id="playing" class="muted"></span></div><div id="progress"></div></header>
<main><p id="summary" class="muted"></p><div id="cards"></div><details><summary>原稿、音频与对话版本</summary><pre id="source"></pre></details></main>
<script id="identity-data" type="application/json">__IDENTITY_DATA__</script><script>
'use strict';
__ANNOTATION_JS__
const data=JSON.parse(document.querySelector('#identity-data').textContent),$=s=>document.querySelector(s),key='nextecho-annotations-v3-identity';
let saved={},storage=true,playUntil=null,playbackRequest=0;try{saved=ReviewAnnotations.restore(localStorage.getItem(key),'identity')}catch(e){storage=false}
$('#reviewer').value='作者';$('#source').textContent=JSON.stringify({...data.source,alignment_audit:data.audit,acoustic_audit:data.acoustic_audit},null,2);
const textLayerLabel={enhanced:'实际采用的润色文字',raw:'原始识别文字','historical-enhanced-replay-not-new-polish':'历史采用润色稿复用（不是本轮新润色）'};
$('#scope').textContent='构建时已按预期hash校验PCM；音频整集起点 '+clock(data.media.origin_ms)+'，文件时长：'+(data.audio_seconds===null?'未知':Math.floor(data.audio_seconds/60)+'分'+Math.floor(data.audio_seconds%60)+'秒')+'；文字层：'+(textLayerLabel[data.text_layer]||'见来源记录')+'。播放器仅另查时长，未重新证明当前音频hash。未提供的范围不作结论。';
function el(tag,text){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e}
function rowKinds(row){const spans=Array.isArray(row.alignment_spans)?row.alignment_spans:[],reasons=new Set([...(row.alignment_reason_counts||[]).map(x=>x.reason),...spans.map(x=>x.assignment_reason)].filter(Boolean)),anchored=spans.some(x=>Number.isFinite(x.native_start_ms)),hasAnchorCount=Object.prototype.hasOwnProperty.call(row,'native_anchor_chars');return {unknownSpeaker:!row.speaker_id||row.identity_status==='unassigned',unknownName:Boolean(row.speaker_id)&&row.identity_status!=='resolved',overlap:String(row.speaker||'').includes('交叠')||reasons.has('acoustic_overlap')||reasons.has('unaligned_overlap'),noAnchor:hasAnchorCount?Number(row.native_anchor_chars)===0:spans.length>0&&!anchored,missingDuration:reasons.has('missing_duration')}}
function sampleRows(rows){const clean=rows.filter(r=>!rowKinds(r).overlap&&Number(r.end_ms)-Number(r.start_ms)>=4000&&Number(r.end_ms)-Number(r.start_ms)<=30000).sort((a,b)=>a.start_ms-b.start_ms),pool=clean.length?clean:rows.filter(r=>!rowKinds(r).overlap).sort((a,b)=>a.start_ms-b.start_ms),n=pool.length;if(n<=3)return pool;return [...new Set([pool[0],pool[Math.floor((n-1)/2)],pool[n-1]])]}
function persist(){try{localStorage.setItem(key,JSON.stringify(ReviewAnnotations.state(saved,'identity')));storage=true}catch(e){storage=false}progress()}
function progress(){const judged=data.review_items.filter(item=>ReviewAnnotations.clean(item,saved,'identity').verdict!=='pending').length;$('#progress').textContent=judged+' / '+data.rows.length+' 条已有判断。'+(storage?'已保存到当前浏览器。':'无法持久保存，请及时导出。')}
function clock(ms){const sec=Math.floor(ms/1000);return Math.floor(sec/3600)+':'+String(Math.floor(sec/60)%60).padStart(2,'0')+':'+String(sec%60).padStart(2,'0')}
function listen(row){const token=++playbackRequest,p=$('#player');playUntil=Math.min(data.audio_seconds,row.audio_end+2);const start=Math.max(0,row.audio_start-2);const play=()=>{if(token!==playbackRequest)return;if(!Number.isFinite(p.duration)||Math.abs(p.duration-data.audio_seconds)>0.05){p.pause();$('#playing').textContent='播放器时长与构建时校验不符，停止回听';return}p.currentTime=start;p.play().catch(()=>{$('#playing').textContent='请点击播放器开始播放'})};p.onloadedmetadata=play;if(p.dataset.loaded&&p.readyState>=1)play();else if(!p.dataset.loaded){p.dataset.loaded='1';p.src=data.audio;p.load()}$('#playing').textContent=clock(row.start_ms)+' · '+row.speaker}
$('#player').ontimeupdate=()=>{if(playUntil!==null&&$('#player').currentTime>=playUntil)$('#player').pause()};
if(data.semantic_review){const a=el('a','继续审核原稿与实际 9B 润色差异');a.href=data.semantic_review;a.target='_blank';a.rel='noopener';$('#semantic').append(a)}
for(const speaker of data.speakers.speakers){const box=el('article'),label=speaker.anonymous_label||(speaker.raw_labels||[]).map(r=>data.labels[r]||r).join(' / ')||speaker.speaker_id,samples=sampleRows(data.rows.filter(r=>r.speaker_id===speaker.speaker_id));box.append(el('b',label+' · '+(speaker.identity_status==='resolved'?'具名证据已记录，仍待回听':'姓名未确认')));box.append(el('small','录音内声学簇 '+speaker.speaker_id+' · UUID '+(speaker.speaker_uuid||'未提供')+' · '+(speaker.turn_count??data.rows.filter(r=>r.speaker_id===speaker.speaker_id).length)+' 条单一归属轮次。UUID只标记本录音里的稳定声学簇，不代表自然人身份。'));for(const row of samples){const b=el('button','回听 '+clock(row.start_ms)+'–'+clock(row.end_ms));b.onclick=()=>listen(row);box.append(b)}const detail=el('details');detail.append(el('summary','查看声学及姓名证据（分数不是概率）'),el('pre',JSON.stringify(speaker,null,2)));box.append(detail);$('#speakers').append(box)}
function render(){const filter=$('#filter').value,q=$('#search').value.trim().toLowerCase(),root=$('#cards');root.replaceChildren();let shown=0;
for(const row of data.rows){const state=ReviewAnnotations.clean({item_id:row.item_version},saved,'identity'),kinds=rowKinds(row),uncertain=kinds.unknownSpeaker||kinds.overlap;if(filter==='unknown-speaker'&&!kinds.unknownSpeaker||filter==='unknown-name'&&!kinds.unknownName||filter==='overlap'&&!kinds.overlap||filter==='no-anchor'&&!kinds.noAnchor||filter==='missing-duration'&&!kinds.missingDuration||filter==='resolved'&&row.identity_status!=='resolved'||filter==='flagged'&&!['wrong','uncertain'].includes(state.verdict))continue;if(q&&!(row.speaker+' '+row.speaker_id+' '+row.text+' '+row.id).toLowerCase().includes(q))continue;shown++;
const card=el('article'),head=el('div');head.className='cardhead';const name=el('span',row.speaker+' · '+clock(row.start_ms)+'–'+clock(row.end_ms));name.className='badge'+(uncertain?' warning':'');const play=el('button','▶ 回听本段');play.onclick=()=>listen(row);head.append(name,play);card.append(head);card.append(el('small',row.id+' · 声学簇 '+(row.speaker_id||'未分配')+' · 来源段 '+row.segment_ids.join('、')+(row.location_kind==='review-point'?' · 仅定位点，无原生词时长；前后试听不是词时长':'')));for(const [active,label]of [[kinds.unknownSpeaker,'声音归属待核'],[kinds.unknownName,'姓名未确认'],[kinds.overlap,'预测交叠/多活动'],[kinds.noAnchor,'无原生时间锚'],[kinds.missingDuration,'原生区间缺失']])if(active)card.append(el('small',label+'　'));const text=el('div',row.text);text.className='text';card.append(text);
const controls=el('div');controls.className='judgment';controls.append(el('span','声音 / 身份判断'));const select=el('select');for(const [v,t]of [['pending','未审核'],['pass','此处归属与标记合适'],['wrong','分人或姓名有误'],['uncertain','仍需核对']]){const o=el('option',t);o.value=v;select.append(o)}select.value=state.verdict||'pending';select.onchange=()=>{saved[row.item_version+':verdict']=select.value;persist()};controls.append(select);card.append(controls);const note=el('textarea');note.placeholder='例如：说话人 3 与说话人 1 是同一人；此处应换人；姓名应为……（附听辨依据）';note.maxLength=4000;note.value=state.note||'';note.oninput=()=>{saved[row.item_version+':note']=note.value;persist()};card.append(note);root.append(card)}
if(!shown)root.append(el('p','当前筛选没有条目。'));progress()}
function summary(){const counts=data.rows.reduce((acc,row)=>{const k=rowKinds(row);if(k.unknownSpeaker)acc.unknownSpeaker++;if(k.unknownName)acc.unknownName++;if(k.overlap)acc.overlap++;if(k.noAnchor)acc.noAnchor++;if(k.missingDuration)acc.missingDuration++;return acc},{unknownSpeaker:0,unknownName:0,overlap:0,noAnchor:0,missingDuration:0});$('#summary').textContent=`${data.rows.length} 条对话记录：声音归属待核 ${counts.unknownSpeaker}，已有声学簇但姓名未确认 ${counts.unknownName}，预测交叠/多活动 ${counts.overlap}，无原生时间锚 ${counts.noAnchor}，缺原生区间 ${counts.missingDuration}。分类可重叠；这些数量不是准确率。`}
$('#search').oninput=render;$('#filter').onchange=render;$('#reviewer').oninput=persist;
$('#export').onclick=()=>{let result;try{result=ReviewAnnotations.exportReview(data,saved,$('#reviewer').value,new Date().toISOString())}catch(e){alert(e.message);return}persist();const blob=new Blob([JSON.stringify(result,null,2)],{type:'application/json'}),a=el('a');a.href=URL.createObjectURL(blob);a.download='nextecho-identity-review.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
summary();render();
</script></body></html>'''
