const form = document.querySelector('#transcribe-form');
const statusEl = document.querySelector('#status');
const resultsEl = document.querySelector('#results');
const runsEl = document.querySelector('#runs');
const progressShellEl = document.querySelector('#progress-shell');
const progressBarEl = document.querySelector('#progress-bar');
const progressCopyEl = document.querySelector('#progress-copy');
const filesInputEl = document.querySelector('#files');
const fileSummaryEl = document.querySelector('#file-summary');
const resolveButtonEl = document.querySelector('#resolve-button');
const sourcePreviewEl = document.querySelector('#source-preview');
const urlsEl = document.querySelector('#urls');
const progressNoteEl = document.querySelector('#progress-note');
const modelStatusEl = document.querySelector('#model-status');
const modelStatusLabelEl = document.querySelector('#model-status-label');
let activeTaskStartedAt = 0;

// All values from URLs, crawled pages, manifests and model errors enter the
// DOM through textContent. Only links created by the server or explicit
// http(s) URLs are made clickable.
function makeNode(tag, { className = '', text = null, attrs = {}, dataset = {} } = {}, children = []) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== null && text !== undefined) element.textContent = String(text);
  Object.entries(attrs).forEach(([name, value]) => {
    if (value !== null && value !== undefined) element.setAttribute(name, String(value));
  });
  Object.entries(dataset).forEach(([name, value]) => {
    element.dataset[name] = String(value);
  });
  children.filter(Boolean).forEach((child) => element.append(child));
  return element;
}

function clearNode(element) {
  while (element.firstChild) element.removeChild(element.firstChild);
}

function safeHttpUrl(value) {
  try {
    const parsed = new URL(String(value), window.location.origin);
    if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return '';
    return parsed.href;
  } catch (error) {
    return '';
  }
}

function artifactLink(runId, absolutePath, query = '') {
  if (!absolutePath) return '';
  const marker = `/outputs/transcriptions/${runId}/`;
  const normalized = String(absolutePath).replaceAll('\\', '/');
  const idx = normalized.indexOf(marker);
  if (idx === -1) return '';
  const relative = normalized.slice(idx + marker.length).split('/').map(encodeURIComponent).join('/');
  return `/artifacts/${encodeURIComponent(runId)}/${relative}${query}`;
}

function safeArtifactOrHttpUrl(value) {
  const candidate = String(value || '');
  if (candidate.startsWith('/artifacts/')) return candidate;
  return safeHttpUrl(candidate);
}

function makeLink(href, label, { download = false, external = false } = {}) {
  const safeHref = external ? safeHttpUrl(href) : safeArtifactOrHttpUrl(href);
  if (!safeHref) return null;
  const anchor = makeNode('a', { text: label, attrs: { href: safeHref } });
  if (external) {
    anchor.target = '_blank';
    anchor.rel = 'noopener noreferrer';
  }
  if (download) anchor.setAttribute('download', '');
  return anchor;
}

function makeActionButton(label, action, data = {}) {
  return makeNode('button', {
    className: 'secondary-button',
    text: label,
    attrs: { type: 'button' },
    dataset: { action, ...data },
  });
}

function formatElapsed(seconds) {
  const safe = Math.max(0, Number(seconds || 0));
  const minutes = Math.floor(safe / 60);
  const remainder = Math.floor(safe % 60);
  return `总耗时 ${String(minutes).padStart(2, '0')}分${String(remainder).padStart(2, '0')}秒`;
}

function formatCoverage(coverage) {
  if (!coverage || !coverage.known) return '覆盖范围未知';
  const start = Number(coverage.start_ms || 0) / 1000;
  const end = Number(coverage.end_ms || 0) / 1000;
  const clock = (seconds) => {
    const whole = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(whole / 3600)).padStart(2, '0')}:${String(Math.floor((whole % 3600) / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`;
  };
  return `覆盖 ${clock(start)}–${clock(end)}`;
}

function progressPresentation(job) {
  const status = job.status || '';
  const stage = job.stage || 'queued';
  const map = {
    queued: ['步骤 1/6：任务已创建', '正在准备转录任务。', 'determinate'],
    waiting_for_model: ['等待本地模型资源', '另一个 ASR、说话人或 LLM 任务正在使用统一重型任务锁。', 'indeterminate'],
    downloading: ['准备模型', '正在下载或准备本地模型。', 'indeterminate'],
    transcribing: ['语音识别进行中', '正在获取媒体、抽取音频并生成原始转录。', 'indeterminate'],
    diarizing: ['区分说话人', '正在分析多人对话中的说话人切换。', 'indeterminate'],
    aligning: ['对齐转写内容', '正在把原始片段对齐到说话人时间线。', 'indeterminate'],
    identifying_speakers: ['识别说话人身份', '正在应用元数据、声纹和人工映射证据。', 'indeterminate'],
    polishing: ['保真整理中', '本地模型正在修正标点、断句和术语；原始转录不会被改写。', 'indeterminate'],
    summarizing: ['生成知识稿', '正在生成带证据引用的摘要、章节和行动项。', 'indeterminate'],
    packaging: ['封装结果', '正在写出可下载的文本、字幕和结构化产物。', 'determinate'],
    ready: ['任务完成', '结果已经准备好。', 'determinate'],
    partial: ['部分完成', '部分阶段失败，但原始转录仍可使用。', 'determinate'],
    completed: ['任务完成', '结果已经准备好。', 'determinate'],
    failed: ['任务失败', '请查看错误信息，决定是否重试。', 'determinate'],
  };
  const selected = map[status] || map[stage] || ['处理中', '正在处理，请稍候。', 'indeterminate'];
  return { title: selected[0], note: selected[1], mode: selected[2] };
}

function sourceSummary(item) {
  return [
    item.platform ? String(item.platform).toUpperCase() : '',
    item.resolver || '',
    item.duration_seconds ? `${Math.round(Number(item.duration_seconds) / 60)} 分钟` : '',
    item.author || '',
  ].filter(Boolean).join(' · ');
}

function renderResolutionMeta(item) {
  const fragment = document.createDocumentFragment();
  if (item.show_website) {
    const row = makeNode('div', { className: 'source-meta-row' }, [makeNode('span', { text: '节目官网：' })]);
    const link = makeLink(item.show_website, item.show_website, { external: true });
    if (link) row.append(link);
    fragment.append(row);
  }
  if (Array.isArray(item.rss_candidates) && item.rss_candidates.length) {
    const row = makeNode('div', { className: 'source-meta-row' }, [makeNode('span', { text: 'RSS 候选：' })]);
    item.rss_candidates.slice(0, 3).forEach((url, index) => {
      const link = makeLink(url, url, { external: true });
      if (link) {
        if (index) row.append(makeNode('span', { text: '、' }));
        row.append(link);
      }
    });
    fragment.append(row);
  }
  if (Array.isArray(item.resolution_notes) && item.resolution_notes.length) {
    fragment.append(makeNode('div', { className: 'source-note', text: item.resolution_notes.join(' ') }));
  }
  return fragment;
}

function renderSourcePreview(sources) {
  clearNode(sourcePreviewEl);
  if (!sources.length) {
    sourcePreviewEl.classList.add('empty');
    sourcePreviewEl.textContent = '这里会显示链接识别结果。';
    return;
  }
  sourcePreviewEl.classList.remove('empty');
  sources.forEach((item) => {
    const title = item.title || item.canonical_url || item.input || '未命名来源';
    const article = makeNode('article', { className: 'source-row' }, [
      makeNode('strong', { text: title }),
      makeNode('div', { className: 'source-meta' }, [makeNode('span', { text: sourceSummary(item) || '等待补充来源信息' })]),
      renderResolutionMeta(item),
    ]);
    if (item.error) article.append(makeNode('div', { className: 'source-error', text: item.error }));
    sourcePreviewEl.append(article);
  });
}

function appendArtifactLinks(nav, runId, item) {
  const links = [
    ['text_path', '查看全文', {}],
    ['text_path', '下载 TXT', { download: true, query: '?download=1' }],
    ['enhanced_text_path', '查看增强稿', {}],
    ['enhanced_text_path', '下载增强 TXT', { download: true, query: '?download=1' }],
    ['enhanced_srt_path', '下载增强字幕', { download: true, query: '?download=1' }],
    ['enhanced_vtt_path', '下载增强 VTT', { download: true, query: '?download=1' }],
    ['srt_path', '下载字幕', { download: true, query: '?download=1' }],
    ['json_path', '下载 JSON', { download: true, query: '?download=1' }],
    ['enhanced_json_path', '下载增强 JSON', { download: true, query: '?download=1' }],
    ['summary_json_path', '查看知识摘要', {}],
    ['knowledge_markdown_path', '打开 Obsidian 知识稿', {}],
    ['media_path', '下载原视频', { download: true, query: '?download=1' }],
    ['audio_path', '下载音频', { download: true, query: '?download=1' }],
  ];
  links.forEach(([key, label, options]) => {
    const href = artifactLink(runId, item[key], options.query || '');
    const link = makeLink(href, label, { download: Boolean(options.download) });
    if (link) nav.append(link);
  });
}

function renderSpeakerPanel(payload) {
  const runId = payload.run_id;
  const transcript = payload.speaker_transcript;
  const article = makeNode('article', { className: 'speaker-panel' });
  const copy = makeNode('div');
  copy.append(makeNode('strong', { text: '访谈逐字稿' }));
  if (transcript) {
    const speakers = Array.isArray(transcript.speakers) ? transcript.speakers : [];
    copy.append(makeNode('p', {
      text: `${transcript.quality_note || '已生成结构化逐字稿。'} · ${formatCoverage(transcript.coverage)}${transcript.fallback_used ? ' · fallback，建议人工复核' : ''}`,
    }));
    copy.append(makeNode('p', { text: transcript.preview_text || '已生成 Speaker 结构化逐字稿。' }));
    if (speakers.length) {
      const mapping = makeNode('div', { className: 'speaker-mapping' });
      mapping.append(makeNode('span', { className: 'source-meta-row', text: '身份映射（只修改身份层，不覆盖原始分离）：' }));
      speakers.forEach((speaker) => {
        const speakerId = speaker.speaker_id || speaker.raw_labels?.[0] || speaker.display_name || 'Speaker 1';
        const label = makeNode('label', { className: 'speaker-map-row' });
        label.append(makeNode('span', { text: speakerId }));
        label.append(makeNode('input', {
          attrs: { type: 'text', value: speaker.display_name && speaker.display_name !== speakerId ? speaker.display_name : '' },
          dataset: { speakerId },
        }));
        mapping.append(label);
      });
      mapping.append(makeActionButton('保存人工 speaker mapping', 'save-speaker-map', { runId }));
      copy.append(mapping);
    }
  } else {
    copy.append(makeNode('p', { text: '适合播客、访谈和多人对谈。先完成转写，再生成 Speaker 版本。' }));
  }
  article.append(copy);
  const nav = makeNode('nav');
  if (transcript) {
    const paths = transcript.paths || {};
    const md = makeLink(artifactLink(runId, paths.md), '查看 Markdown');
    const txt = makeLink(artifactLink(runId, paths.txt, '?download=1'), '下载 TXT', { download: true });
    const json = makeLink(artifactLink(runId, paths.json, '?download=1'), '下载 JSON', { download: true });
    [md, txt, json].filter(Boolean).forEach((link) => nav.append(link));
  } else {
    nav.append(makeActionButton('生成访谈逐字稿', 'speaker-transcript', { runId }));
  }
  article.append(nav);
  return article;
}

function renderResults(payload) {
  const runId = payload.run_id;
  const manifest = payload.manifest || {};
  clearNode(resultsEl);
  resultsEl.classList.remove('empty');
  resultsEl.append(makeNode('article', { className: 'summary-banner' }, [
    makeNode('strong', { text: '本次转录已完成' }),
    makeNode('span', { text: `${formatElapsed(payload.total_elapsed_seconds)}${manifest.partial_count ? ` · ${manifest.partial_count} 条智能加工部分完成` : ''}` }),
  ]));
  if (manifest.error_count) {
    const errorArticle = makeNode('article', { className: 'speaker-panel' });
    errorArticle.append(makeNode('div', {}, [
      makeNode('strong', { text: '错误摘要' }),
      makeNode('p', { text: `${manifest.error_count} 条素材没有成功完成，原始转录仍保留。` }),
    ]));
    const nav = makeNode('nav');
    const errorsLink = artifactLink(runId, payload.errors_path);
    const view = makeLink(errorsLink, '查看错误 JSON');
    const download = makeLink(`${errorsLink}?download=1`, '下载错误包', { download: true });
    [view, download].filter(Boolean).forEach((link) => nav.append(link));
    errorArticle.append(nav);
    resultsEl.append(errorArticle);
  }
  resultsEl.append(renderSpeakerPanel(payload));
  (manifest.results || []).forEach((item) => {
    const card = makeNode('article', { className: 'result-card' });
    const info = makeNode('div');
    info.append(makeNode('span', { className: 'platform-pill', text: item.platform || 'local' }));
    info.append(makeNode('strong', { text: item.title || item.source_label || '未命名素材' }));
    const details = [
      item.status,
      `${item.char_count ?? 0} 字`,
      item.author || '',
      item.duration_seconds ? `${Math.round(Number(item.duration_seconds) / 60)} 分钟` : '',
      item.speaker_status ? `speaker ${item.speaker_status}` : '',
    ].filter(Boolean).join(' · ');
    info.append(makeNode('p', { text: details }));
    const asr = item.asr || {};
    if (asr.actual_model || asr.actual_engine) {
      const fallback = Array.isArray(asr.fallbacks) && asr.fallbacks.length ? ` · ${asr.fallbacks.length} 次 fallback` : '';
      info.append(makeNode('p', { className: 'source-meta-row', text: `ASR：${asr.actual_engine || 'local'} / ${asr.actual_model || 'unknown'}${asr.revision ? ` @ ${asr.revision}` : ''}${fallback}` }));
    }
    const diarization = item.diarization || {};
    if (diarization.backend) {
      info.append(makeNode('p', { className: 'source-meta-row', text: `说话人：${diarization.backend}${diarization.fallback_used ? ` · fallback${diarization.fallback_reason ? `：${diarization.fallback_reason}` : ''}` : ''} · ${formatCoverage(diarization.coverage)}` }));
    }
    const enhancement = item.enhancement || {};
    if (enhancement.enabled) {
      const models = Array.isArray(enhancement.actual_models) && enhancement.actual_models.length ? enhancement.actual_models.join('、') : '确定性整理';
      info.append(makeNode('p', { className: 'source-meta-row', text: `知识加工：${models}${enhancement.status === 'partial' ? ' · 部分完成，已保留原始稿' : ''}${enhancement.timings?.total_seconds ? ` · ${formatElapsed(enhancement.timings.total_seconds)}` : ''}` }));
    }
    if (item.error) info.append(makeNode('p', { className: 'source-error', text: item.error }));
    card.append(info);
    const nav = makeNode('nav');
    appendArtifactLinks(nav, runId, item);
    nav.append(makeActionButton('重新整理与摘要', 'enhance', { runId }));
    card.append(nav);
    resultsEl.append(card);
  });
}

async function loadRuns() {
  const response = await fetch('/api/runs');
  const payload = await response.json();
  clearNode(runsEl);
  if (!response.ok || !Array.isArray(payload.runs) || !payload.runs.length) {
    runsEl.append(makeNode('p', { className: 'muted', text: '暂无历史运行。' }));
    return;
  }
  payload.runs.forEach((run) => {
    runsEl.append(makeNode('button', {
      className: 'run-row',
      attrs: { type: 'button' },
      dataset: { runId: run.run_id || '' },
    }, [
      makeNode('strong', { text: run.display_title || run.run_id || '未命名运行' }),
      makeNode('span', { text: `${run.item_count || 0} 条 · ${run.quality || '未知质量'}${run.error_count ? ` · ${run.error_count} 失败` : ''} · ${run.run_id || ''}` }),
    ]));
  });
}

function renderModelStatus(payload) {
  const ollama = payload.ollama || {};
  clearNode(modelStatusEl);
  modelStatusEl.classList.remove('empty');
  const rows = [
    `Apple Silicon：${payload.apple_silicon ? '是' : '否'}`,
    `Qwen ASR 运行时：${payload.qwen_mlx_available ? '已安装' : '未安装，将回退 Whisper'}`,
    `内存门禁：${payload.qwen_memory_safe ? '通过' : '未通过'}`,
    `Ollama：${ollama.ready ? '已连接' : '未启动'}`,
    `快速模型：${ollama.fast_model_ready ? (payload.llm_models?.fast || '4B') + ' 已就绪' : (payload.llm_models?.fast || '4B') + ' 未下载'}`,
    `高质量模型：${ollama.quality_model_ready ? (payload.llm_models?.quality || '9B') + ' 已就绪' : (payload.llm_models?.quality || '9B') + ' 未下载'}`,
    `说话人后端：${payload.diarization?.selected_backend || '不可用'}${payload.diarization?.fallback ? ` · fallback ${payload.diarization.fallback}` : ''}`,
  ];
  modelStatusEl.append(makeNode('div', { className: 'source-meta-row', text: rows.join(' · ') }));
  const selected = payload.selected || {};
  modelStatusEl.append(makeNode('div', { className: 'source-meta-row', text: `实际选择：ASR ${selected.asr?.engine || 'unknown'} / ${selected.asr?.model || 'unknown'} · speaker ${selected.diarization?.backend || 'unknown'} · LLM ${selected.llm?.model || 'unknown'}` }));
  const actions = makeNode('nav');
  if (!ollama.fast_model_ready) actions.append(makeActionButton('下载快速模型', 'download-model', { kind: 'llm-fast' }));
  if (!ollama.quality_model_ready) actions.append(makeActionButton('下载高质量模型', 'download-model', { kind: 'llm-quality' }));
  if (payload.qwen_mlx_available && !payload.qwen_asr_model_cache?.accurate) actions.append(makeActionButton('准备 Qwen ASR 1.7B', 'download-model', { kind: 'asr-accurate' }));
  modelStatusEl.append(actions);
  modelStatusLabelEl.textContent = ollama.ready ? '本地服务已连接' : '需要启动 Ollama';
}

async function loadModelStatus() {
  try {
    const response = await fetch('/api/models/status');
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '无法读取模型状态');
    renderModelStatus(payload);
  } catch (error) {
    modelStatusLabelEl.textContent = '读取失败';
    clearNode(modelStatusEl);
    modelStatusEl.append(makeNode('div', { className: 'source-error', text: error.message }));
  }
}

function updateProgress(job) {
  progressShellEl.classList.remove('hidden');
  progressCopyEl.classList.remove('hidden');
  progressNoteEl.classList.remove('hidden');
  const elapsedSeconds = job.elapsed_seconds
    ?? (activeTaskStartedAt ? Math.max(0, Math.floor((Date.now() - activeTaskStartedAt) / 1000)) : 0);
  const presentation = progressPresentation(job);
  progressShellEl.classList.toggle('progress-shell-indeterminate', presentation.mode === 'indeterminate');
  progressBarEl.classList.toggle('progress-bar-indeterminate', presentation.mode === 'indeterminate');
  progressBarEl.style.width = presentation.mode === 'indeterminate' ? '42%' : `${Math.max(0, Math.min(100, Number(job.progress || 0)))}%`;
  progressCopyEl.textContent = `${presentation.title} · ${formatElapsed(elapsedSeconds)}`;
  progressNoteEl.textContent = job.message ? `${job.message} · ${presentation.note}` : presentation.note;
}

async function waitForJob(jobId) {
  while (true) {
    const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || '无法读取任务状态');
    updateProgress(job);
    if (job.status === 'ready' || job.status === 'partial' || job.status === 'completed') return job.result;
    if (job.status === 'failed') throw new Error(job.error || job.message || '任务失败');
    await new Promise((resolve) => setTimeout(resolve, 800));
  }
}

async function resolveSources() {
  const urls = urlsEl.value.trim();
  if (!urls) {
    renderSourcePreview([]);
    return [];
  }
  const response = await fetch('/api/resolve', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ urls }),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || '无法识别链接');
  renderSourcePreview(payload.sources || []);
  return payload.sources || [];
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const submitButton = form.querySelector('button[type="submit"]');
  submitButton.disabled = true;
  resolveButtonEl.disabled = true;
  activeTaskStartedAt = Date.now();
  statusEl.textContent = '转写中…';
  progressShellEl.classList.remove('hidden');
  progressCopyEl.classList.remove('hidden');
  progressNoteEl.classList.remove('hidden');
  progressShellEl.classList.remove('progress-shell-indeterminate');
  progressBarEl.classList.remove('progress-bar-indeterminate');
  progressBarEl.style.width = '0%';
  progressCopyEl.textContent = `任务已创建 · ${formatElapsed(0)}`;
  progressNoteEl.textContent = '系统正在准备转录任务。';
  resultsEl.classList.add('empty');
  resultsEl.textContent = '任务正在运行。';
  try {
    await resolveSources();
    const response = await fetch('/api/transcribe', { method: 'POST', body: new FormData(form) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '转写失败');
    const result = await waitForJob(payload.job_id);
    statusEl.textContent = result?.manifest?.partial_count ? '部分完成' : '已完成';
    renderResults(result);
    await loadRuns();
  } catch (error) {
    statusEl.textContent = '失败';
    resultsEl.classList.add('empty');
    resultsEl.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    resolveButtonEl.disabled = false;
  }
});

resolveButtonEl.addEventListener('click', async () => {
  resolveButtonEl.disabled = true;
  try {
    statusEl.textContent = '识别中';
    await resolveSources();
    statusEl.textContent = '已识别';
  } catch (error) {
    statusEl.textContent = '识别失败';
    renderSourcePreview([{ title: '', canonical_url: '', input: '', platform: '', resolver: '', duration_seconds: 0, author: '', error: error.message }]);
  } finally {
    resolveButtonEl.disabled = false;
  }
});

filesInputEl.addEventListener('change', () => {
  const files = [...filesInputEl.files];
  fileSummaryEl.textContent = files.length
    ? `已选择 ${files.length} 个文件：${files.map((file) => file.name).join('、')}`
    : '支持 mp3 / mp4 / m4a / wav / flac / aac / mov / webm';
});

runsEl.addEventListener('click', async (event) => {
  const row = event.target.closest('[data-run-id]');
  if (!row) return;
  const response = await fetch(`/api/runs/${encodeURIComponent(row.dataset.runId)}`);
  const payload = await response.json();
  if (!response.ok) {
    statusEl.textContent = '加载失败';
    resultsEl.textContent = payload.error || '无法打开历史运行';
    return;
  }
  statusEl.textContent = '历史结果';
  progressShellEl.classList.add('hidden');
  progressCopyEl.classList.add('hidden');
  progressNoteEl.classList.add('hidden');
  renderResults(payload);
});

resultsEl.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-action]');
  if (!button) return;
  event.preventDefault();
  event.stopPropagation();
  button.disabled = true;
  const action = button.dataset.action;
  if (action === 'save-speaker-map') {
    const mapping = {};
    button.closest('.speaker-mapping').querySelectorAll('[data-speaker-id]').forEach((input) => {
      mapping[input.dataset.speakerId] = input.value.trim();
    });
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(button.dataset.runId)}/speaker-map`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mapping }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || '无法保存 speaker mapping');
      statusEl.textContent = 'speaker mapping 已保存';
      const refreshed = await fetch(`/api/runs/${encodeURIComponent(button.dataset.runId)}`);
      renderResults(await refreshed.json());
    } catch (error) {
      statusEl.textContent = '保存失败';
      resultsEl.append(makeNode('p', { className: 'source-error', text: error.message }));
    } finally {
      button.disabled = false;
    }
    return;
  }
  activeTaskStartedAt = Date.now();
  const isEnhance = action === 'enhance';
  statusEl.textContent = isEnhance ? '智能整理与摘要生成中…' : '访谈逐字稿生成中…';
  progressShellEl.classList.remove('hidden');
  progressCopyEl.classList.remove('hidden');
  progressNoteEl.classList.remove('hidden');
  progressBarEl.style.width = '0%';
  progressCopyEl.textContent = `任务已创建 · ${formatElapsed(0)}`;
  progressNoteEl.textContent = isEnhance ? '系统将只读取原始结构化转录，不会重新下载或转写音频。' : '系统正在准备访谈逐字稿任务。';
  try {
    const endpoint = isEnhance
      ? `/api/runs/${encodeURIComponent(button.dataset.runId)}/enhance`
      : `/api/runs/${encodeURIComponent(button.dataset.runId)}/speaker-transcript`;
    const response = await fetch(endpoint, {
      method: 'POST',
      ...(isEnhance ? { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ profile: 'fast' }) } : {}),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || (isEnhance ? '无法启动智能整理任务' : '无法启动访谈逐字稿任务'));
    const result = await waitForJob(payload.job_id);
    statusEl.textContent = isEnhance ? '智能整理与摘要已完成' : '访谈逐字稿已完成';
    renderResults(result);
    await loadRuns();
  } catch (error) {
    statusEl.textContent = '失败';
    resultsEl.classList.add('empty');
    resultsEl.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

modelStatusEl.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-action="download-model"]');
  if (!button) return;
  const kind = button.dataset.kind;
  if (!window.confirm('模型下载可能占用数 GB 磁盘空间，确认现在下载吗？')) return;
  button.disabled = true;
  modelStatusLabelEl.textContent = '下载中…';
  try {
    const response = await fetch('/api/models/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '无法开始下载');
    await waitForJob(payload.job_id);
    await loadModelStatus();
  } catch (error) {
    modelStatusLabelEl.textContent = '下载失败';
    modelStatusEl.append(makeNode('div', { className: 'source-error', text: error.message }));
  } finally {
    button.disabled = false;
  }
});

loadRuns();
loadModelStatus();
