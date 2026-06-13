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

function artifactLink(runId, absolutePath) {
  if (!absolutePath) return null;
  const marker = `/outputs/transcriptions/${runId}/`;
  const normalized = absolutePath.replaceAll('\\\\', '/');
  const idx = normalized.indexOf(marker);
  if (idx === -1) return null;
  const relative = normalized.slice(idx + marker.length);
  return `/artifacts/${runId}/${relative}`;
}

function sourceSummary(item) {
  const bits = [
    item.platform ? item.platform.toUpperCase() : '',
    item.resolver || '',
    item.duration_seconds ? `${Math.round(item.duration_seconds / 60)} 分钟` : '',
    item.author || '',
  ].filter(Boolean);
  return bits.join(' · ');
}

function renderSourcePreview(sources) {
  if (!sources.length) {
    sourcePreviewEl.classList.add('empty');
    sourcePreviewEl.textContent = '这里会显示链接识别结果。';
    return;
  }
  sourcePreviewEl.classList.remove('empty');
  sourcePreviewEl.innerHTML = sources.map((item) => `
    <article class="source-row">
      <strong>${item.title || item.canonical_url || item.input}</strong>
      <div class="source-meta">
        <span>${sourceSummary(item) || '等待补充来源信息'}</span>
      </div>
      ${item.error ? `<div class="source-error">${item.error}</div>` : ''}
    </article>
  `).join('');
}

function renderResults(payload) {
  const { run_id: runId, manifest, errors_path: errorsPath } = payload;
  const speakerPanel = renderSpeakerPanel(payload);
  const errorsLink = artifactLink(runId, errorsPath);
  const errorsBanner = manifest.error_count
    ? `
      <article class="speaker-panel">
        <div>
          <strong>错误摘要</strong>
          <p>${manifest.error_count} 条素材没有成功完成，建议先看错误包再决定是否重试。</p>
        </div>
        <nav>
          ${errorsLink ? `<a href="${errorsLink}" target="_blank">查看错误 JSON</a>` : ''}
          ${errorsLink ? `<a href="${errorsLink}?download=1">下载错误包</a>` : ''}
        </nav>
      </article>
    `
    : '';
  const cards = manifest.results.map((item) => {
    const txt = artifactLink(runId, item.text_path);
    const srt = artifactLink(runId, item.srt_path);
    const json = artifactLink(runId, item.json_path);
    const media = artifactLink(runId, item.media_path);
    const audio = artifactLink(runId, item.audio_path);
    const details = [item.status, `${item.char_count ?? 0} 字`, item.author || '', item.duration_seconds ? `${Math.round(item.duration_seconds / 60)} 分钟` : '']
      .filter(Boolean)
      .join(' · ');
    return `
      <article class="result-card">
        <div>
          <span class="platform-pill">${item.platform || 'local'}</span>
          <strong>${item.title || item.source_label}</strong>
          <p>${details}</p>
          ${item.error ? `<p class="source-error">${item.error}</p>` : ''}
        </div>
        <nav>
          ${txt ? `<a href="${txt}" target="_blank">查看全文</a>` : ''}
          ${txt ? `<a href="${txt}?download=1">下载 TXT</a>` : ''}
          ${srt ? `<a href="${srt}?download=1">下载字幕</a>` : ''}
          ${json ? `<a href="${json}?download=1">下载 JSON</a>` : ''}
          ${media ? `<a href="${media}?download=1">下载原视频</a>` : ''}
          ${audio ? `<a href="${audio}?download=1">下载音频</a>` : ''}
        </nav>
      </article>
    `;
  }).join('');
  resultsEl.classList.remove('empty');
  resultsEl.innerHTML = `${errorsBanner}${speakerPanel}${cards}`;
}

function renderSpeakerPanel(payload) {
  const { run_id: runId, speaker_transcript: speakerTranscript } = payload;
  if (speakerTranscript) {
    const md = artifactLink(runId, speakerTranscript.paths.md);
    const txt = artifactLink(runId, speakerTranscript.paths.txt);
    const json = artifactLink(runId, speakerTranscript.paths.json);
    return `
      <article class="speaker-panel">
        <div>
          <strong>访谈逐字稿</strong>
          <p>${speakerTranscript.preview_text || '已生成 Speaker 1 / Speaker 2 结构化逐字稿。'}</p>
        </div>
        <nav>
          ${md ? `<a href="${md}" target="_blank">查看 Markdown</a>` : ''}
          ${txt ? `<a href="${txt}?download=1">下载 TXT</a>` : ''}
          ${json ? `<a href="${json}?download=1">下载 JSON</a>` : ''}
        </nav>
      </article>
    `;
  }
  return `
    <article class="speaker-panel">
      <div>
        <strong>访谈逐字稿</strong>
        <p>适合播客、访谈和多人对谈。先完成转写，再生成 Speaker 1 / Speaker 2 版本。</p>
      </div>
      <nav>
        <button type="button" class="secondary-button" data-action="speaker-transcript" data-run-id="${runId}">生成访谈逐字稿</button>
      </nav>
    </article>
  `;
}

async function loadRuns() {
  const response = await fetch('/api/runs');
  const payload = await response.json();
  runsEl.innerHTML = payload.runs.map((run) => `
    <button class="run-row" data-run-id="${run.run_id}" type="button">
      <strong>${run.run_id}</strong>
      <span>${run.item_count} 条 · ${run.quality}${run.error_count ? ` · ${run.error_count} 失败` : ''}</span>
    </button>
  `).join('') || '<p class="muted">暂无历史运行。</p>';
}

function updateProgress(job) {
  progressShellEl.classList.remove('hidden');
  progressCopyEl.classList.remove('hidden');
  progressBarEl.style.width = `${job.progress ?? 0}%`;
  progressCopyEl.textContent = `${job.message || '处理中'} · ${job.progress ?? 0}%`;
}

async function waitForJob(jobId) {
  while (true) {
    const response = await fetch(`/api/jobs/${jobId}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.error || '无法读取任务状态');
    updateProgress(job);
    if (job.status === 'completed') return job.result;
    if (job.status === 'failed') throw new Error(job.error || job.message || '转写失败');
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
  statusEl.textContent = '转写中…';
  progressShellEl.classList.remove('hidden');
  progressCopyEl.classList.remove('hidden');
  progressBarEl.style.width = '0%';
  progressCopyEl.textContent = '任务已创建 · 0%';
  resultsEl.classList.add('empty');
  resultsEl.textContent = '任务正在运行。';

  try {
    await resolveSources();
    const response = await fetch('/api/transcribe', {
      method: 'POST',
      body: new FormData(form),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '转写失败');
    const result = await waitForJob(payload.job_id);
    statusEl.textContent = '已完成';
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
  const response = await fetch(`/api/runs/${row.dataset.runId}`);
  const payload = await response.json();
  if (!response.ok) {
    statusEl.textContent = '加载失败';
    resultsEl.textContent = payload.error || '无法打开历史运行';
    return;
  }
  statusEl.textContent = '历史结果';
  progressShellEl.classList.add('hidden');
  progressCopyEl.classList.add('hidden');
  renderResults(payload);
});

resultsEl.addEventListener('click', async (event) => {
  const button = event.target.closest('[data-action="speaker-transcript"]');
  if (!button) return;
  button.disabled = true;
  statusEl.textContent = '访谈逐字稿生成中…';
  progressShellEl.classList.remove('hidden');
  progressCopyEl.classList.remove('hidden');
  progressBarEl.style.width = '0%';
  progressCopyEl.textContent = '访谈逐字稿任务已创建 · 0%';
  try {
    const response = await fetch(`/api/runs/${button.dataset.runId}/speaker-transcript`, {
      method: 'POST',
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '无法启动访谈逐字稿任务');
    const result = await waitForJob(payload.job_id);
    statusEl.textContent = '访谈逐字稿已完成';
    renderResults(result);
    await loadRuns();
  } catch (error) {
    statusEl.textContent = '失败';
    resultsEl.classList.add('empty');
    resultsEl.textContent = error.message;
  }
});

loadRuns();
