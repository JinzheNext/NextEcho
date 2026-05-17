const form = document.querySelector('#transcribe-form');
const statusEl = document.querySelector('#status');
const resultsEl = document.querySelector('#results');
const runsEl = document.querySelector('#runs');
const progressShellEl = document.querySelector('#progress-shell');
const progressBarEl = document.querySelector('#progress-bar');
const progressCopyEl = document.querySelector('#progress-copy');

function artifactLink(runId, absolutePath) {
  const marker = `/outputs/transcriptions/${runId}/`;
  const normalized = absolutePath.replaceAll('\\\\', '/');
  const idx = normalized.indexOf(marker);
  if (idx === -1) return null;
  const relative = normalized.slice(idx + marker.length);
  return `/artifacts/${runId}/${relative}`;
}

function renderResults(payload) {
  const { run_id: runId, manifest } = payload;
  const cards = manifest.results.map((item) => {
    const txt = artifactLink(runId, item.text_path);
    const srt = artifactLink(runId, item.srt_path);
    const json = artifactLink(runId, item.json_path);
    return `
      <article class="result-card">
        <div>
          <strong>${item.source_label}</strong>
          <p>${item.status} · ${item.char_count ?? 0} 字</p>
        </div>
        <nav>
          ${txt ? `<a href="${txt}" target="_blank">全文</a>` : ''}
          ${srt ? `<a href="${srt}" target="_blank">字幕</a>` : ''}
          ${json ? `<a href="${json}" target="_blank">JSON</a>` : ''}
        </nav>
      </article>
    `;
  }).join('');
  resultsEl.classList.remove('empty');
  resultsEl.innerHTML = cards;
}

async function loadRuns() {
  const response = await fetch('/api/runs');
  const payload = await response.json();
  runsEl.innerHTML = payload.runs.map((run) => `
    <article class="run-row">
      <strong>${run.run_id}</strong>
      <span>${run.item_count} 条 · ${run.model_name}</span>
    </article>
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

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const submitButton = form.querySelector('button');
  submitButton.disabled = true;
  statusEl.textContent = '转写中…';
  progressShellEl.classList.remove('hidden');
  progressCopyEl.classList.remove('hidden');
  progressBarEl.style.width = '0%';
  progressCopyEl.textContent = '任务已创建 · 0%';
  resultsEl.classList.add('empty');
  resultsEl.textContent = '任务正在运行。';

  try {
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
  }
});

loadRuns();
