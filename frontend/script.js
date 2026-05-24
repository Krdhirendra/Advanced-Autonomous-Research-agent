// Tab Switching Logic
function switchTab(viewId, element) {
    // Hide all views
    document.querySelectorAll('.view-content').forEach(view => {
        view.style.display = 'none';
    });
    // Remove active class from buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show view and set active button
    document.getElementById(viewId).style.display = 'block';
    element.classList.add('active');
    closeSidebar();
}

function toggleSidebar() {
    document.body.classList.toggle('sidebar-open');
}

function closeSidebar() {
    document.body.classList.remove('sidebar-open');
}

// Update Status Box
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


async function startResearch() {
    const query = document.getElementById('query-input').value.trim();
    if (!query) return;

    const btn = document.getElementById('start-btn');
    const resultsContainer = document.getElementById('results-container');
    const pdfViewer = document.getElementById('pdf-viewer');
    const pdfContainer = document.getElementById('pdf-viewer-container');
    const textViewer = document.getElementById('text-viewer');
    const downloadBtn = document.getElementById('download-btn');
    const resultsTitle = document.getElementById('results-title');
    const researchProgress = document.getElementById('research-progress');

    // UI Reset
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing';
    researchProgress.style.display = 'block';
    resultsContainer.style.display = 'none';
    pdfViewer.src = ''; 
    
    setAgentStatus('RUNNING', 'Executing research loop...', 'var(--accent-blue)');
    addLog(`[USER_INPUT] ${query}`);
    addLog("[SYSTEM] Connection established. Awaiting node telemetry...", "system");

    // DELETE ALL FAKE TIMEOUTS HERE

    try {
        const BACKEND_URL = "https://player12026-aara-backend.hf.space/api/research"; 

        const response = await fetch(BACKEND_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) throw new Error(`HTTP Error: ${response.status}`);

        // 1. ATTACH A STREAM READER
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");

        // 2. READ THE STREAM IN REAL-TIME
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // Decode the chunk and split it by newlines (in case multiple arrive at once)
            const chunk = decoder.decode(value, { stream: true });
            const lines = chunk.split('\n').filter(line => line.trim() !== '');

            for (const line of lines) {
                const data = JSON.parse(line);

                // Handle live logs from LangGraph nodes
                if (data.type === "log") {
                    addLog(data.message, "system");
                } 
                // Handle the final result delivery
                else if (data.type === "result") {
                    resultsContainer.style.display = 'flex';

                    if (data.response_type === "text") {
                        resultsTitle.innerHTML = '<i class="fa-solid fa-comment"></i> Agent Response';
                        textViewer.textContent = data.content;
                        textViewer.style.display = 'block';
                        pdfContainer.style.display = 'none';
                        downloadBtn.style.display = 'none';

                        addLog(`[SYSTEM] Bypass successful. Processed in ${data.time}s.`, "system");

                    } else if (data.response_type === "pdf") {
                        resultsTitle.innerHTML = '<i class="fa-solid fa-file-pdf"></i> Synthesis Output';
                        const backendHost = "https://player12026-aara-backend.hf.space";
                        const fullPdfUrl = backendHost + data.pdf_url;
                        
                        pdfViewer.src = fullPdfUrl;
                        downloadBtn.href = fullPdfUrl;
                        
                        textViewer.style.display = 'none';
                        pdfContainer.style.display = 'block';
                        downloadBtn.style.display = 'inline-block';
                        addLog(`[SYSTEM] Operation complete. PDF compiled in ${data.time}s.`, "system");
                    }
                    setAgentStatus('COMPLETE', `Research synthesized in ${data.time} seconds.`, 'var(--term-green)');
                    setAgentStatus('COMPLETE', 'Awaiting next command.', 'var(--term-green)');
                }
            }
        }
    } catch (error) {
        setAgentStatus('ERROR', 'Pipeline failure.', 'var(--term-error)');
        addLog(`[FETCH_ERROR] Connection lost or failed.`, "error");
        console.error(error);
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fa-solid fa-play"></i> Execute';
        researchProgress.style.display = 'none';
    }
}