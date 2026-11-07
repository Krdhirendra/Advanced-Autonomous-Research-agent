// Backend: local server when developing locally, hosted Space otherwise.
const IS_LOCAL = ["localhost", "127.0.0.1", ""].includes(window.location.hostname);
const BACKEND_HOST = IS_LOCAL ? "http://localhost:7860" : "https://player12026-aara-backend.hf.space";

const EXAMPLE_QUERY = "Evaluate the architectural differences and latency trade-offs between standard Retrieval-Augmented Generation (RAG) and Agentic RAG workflows utilizing state machine routing.";

// Tab Switching Logic
function switchTab(viewId, element) {
    document.querySelectorAll('.view-content').forEach(view => { view.style.display = 'none'; });
    document.querySelectorAll('.nav-btn').forEach(btn => { btn.classList.remove('active'); });
    document.getElementById(viewId).style.display = 'block';
    element.classList.add('active');
}

function useExample() {
    const input = document.getElementById('query-input');
    input.value = EXAMPLE_QUERY;
    input.focus();
}

// Update Status Cell
function setAgentStatus(status, desc, color) {
    const titleEl = document.querySelector('.status-title');
    const descEl = document.querySelector('.status-desc');
    titleEl.textContent = status;
    titleEl.style.color = color;
    descEl.textContent = desc;
}

// Terminal Logging
function addLog(message, type = 'normal') {
    const terminal = document.getElementById('terminal-output');
    const line = document.createElement('div');

    let className = 'log-line';
    if (type === 'system') className += ' sys-log';
    if (type === 'error') className += ' err-log';

    line.className = className;
    line.textContent = message;

    terminal.appendChild(line);
    terminal.scrollTop = terminal.scrollHeight;
}

function handleEvent(data, els) {
    if (data.type === "log") {
        addLog(data.message, "system");
    } else if (data.type === "error") {
        setAgentStatus('ERROR', 'Pipeline failure.', 'var(--err)');
        addLog(`[PIPELINE_ERROR] ${data.message}`, "error");
    } else if (data.type === "result") {
        els.resultsContainer.style.display = 'block';

        if (data.response_type === "text") {
            els.resultsTitle.innerHTML = '<i class="fa-solid fa-comment"></i> AGENT RESPONSE';
            els.textViewer.textContent = data.content;
            els.textViewer.style.display = 'block';
            els.pdfContainer.style.display = 'none';
            els.downloadBtn.style.display = 'none';
            addLog(`[SYSTEM] Bypass successful. Processed in ${data.time}s.`, "system");
        } else if (data.response_type === "pdf") {
            els.resultsTitle.innerHTML = '<i class="fa-solid fa-file-pdf"></i> SYNTHESIS OUTPUT';
            const fullPdfUrl = BACKEND_HOST + data.pdf_url;

            els.pdfViewer.src = fullPdfUrl;
            els.downloadBtn.href = fullPdfUrl;

            els.textViewer.style.display = 'none';
            els.pdfContainer.style.display = 'block';
            els.downloadBtn.style.display = 'inline-flex';
            addLog(`[SYSTEM] Operation complete. PDF compiled in ${data.time}s.`, "system");
        }
        setAgentStatus('COMPLETE', `Research synthesized in ${data.time} seconds.`, 'var(--green)');
        els.resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

async function startResearch() {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;

    const btn = document.getElementById('start-btn');
    const els = {
        resultsContainer: document.getElementById('results-container'),
        pdfViewer: document.getElementById('pdf-viewer'),
        pdfContainer: document.getElementById('pdf-viewer-container'),
        textViewer: document.getElementById('text-viewer'),
        downloadBtn: document.getElementById('download-btn'),
        resultsTitle: document.getElementById('results-title'),
    };
    const researchProgress = document.getElementById('research-progress');

    // UI Reset
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> PROCESSING';
    researchProgress.style.display = 'block';
    els.resultsContainer.style.display = 'none';
    els.pdfViewer.src = '';

    setAgentStatus('RUNNING', 'Executing research loop...', 'var(--red)');
    addLog(`[USER_INPUT] ${query}`);
    addLog("[SYSTEM] Connection established. Awaiting node telemetry...", "system");

    try {
        const response = await fetch(`${BACKEND_HOST}/api/research`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";

        // NDJSON: a network chunk may hold several lines or end mid-line, so buffer until a newline.
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (line.trim() === '') continue;
                handleEvent(JSON.parse(line), els);
            }
        }
        if (buffer.trim() !== '') handleEvent(JSON.parse(buffer), els);
    } catch (error) {
        setAgentStatus('ERROR', 'Pipeline failure.', 'var(--err)');
        addLog(`[FETCH_ERROR] Connection lost or failed.`, "error");
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.innerHTML = 'EXECUTE <i class="fa-solid fa-arrow-right"></i>';
        researchProgress.style.display = 'none';
    }
}
