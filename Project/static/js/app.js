/**
 * app.js — Mini Photoshop Frontend Logic
 * Handles: API communication, UI interactions, histogram rendering, crop tool
 */

// =============================================================================
// STATE
// =============================================================================
const state = {
    imageLoaded: false,
    processing: false,
    cropMode: false,
    cropStart: null,
    cropRect: null,
};

// =============================================================================
// HELPERS
// =============================================================================

async function apiCall(endpoint, options = {}) {
    showLoading(true);
    try {
        const res = await fetch(endpoint, options);
        const data = await res.json();
        if (!data.success) {
            showToast(data.error || 'Operation failed', 'error');
            return null;
        }
        return data;
    } catch (err) {
        showToast('Connection error: ' + err.message, 'error');
        return null;
    } finally {
        showLoading(false);
    }
}

async function apiPost(endpoint, body = {}) {
    return apiCall(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body)
    });
}

function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    const dot = document.getElementById('statusDot');
    if (show) {
        overlay.classList.add('show');
        dot.className = 'status-dot processing';
        state.processing = true;
    } else {
        overlay.classList.remove('show');
        dot.className = 'status-dot';
        state.processing = false;
    }
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast-item ${type}`;
    const icons = { success: '✓', error: '✕', info: 'ℹ' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${message}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('removing');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function updateImageInfo(info) {
    if (!info) return;
    document.getElementById('statusInfo').textContent =
        `${info.width} × ${info.height} px | ${info.channels === 1 ? 'Grayscale' : 'RGB'}`;
}

function updateAfterImage(base64) {
    const img = document.getElementById('afterImage');
    img.src = 'data:image/png;base64,' + base64;
    img.style.display = 'block';
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('previewWrapper').style.display = 'flex';
}

function updateBothImages(base64) {
    const before = document.getElementById('beforeImage');
    const after = document.getElementById('afterImage');
    before.src = 'data:image/png;base64,' + base64;
    after.src = 'data:image/png;base64,' + base64;
    before.style.display = 'block';
    after.style.display = 'block';
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('previewWrapper').style.display = 'flex';
    state.imageLoaded = true;
}

// =============================================================================
// IMAGE MANAGEMENT
// =============================================================================

async function uploadImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    const data = await apiCall('/api/upload', { method: 'POST', body: formData });
    if (data) {
        updateBothImages(data.image);
        updateImageInfo(data.info);
        updateHistogram();
        showToast('Image loaded successfully!', 'success');
    }
}

async function resetImage() {
    const data = await apiPost('/api/reset');
    if (data) {
        updateAfterImage(data.image);
        updateImageInfo(data.info);
        updateHistogram();
        showToast('Image reset to original', 'success');
    }
}

async function saveImage() {
    const filename = document.getElementById('saveFilename').value || 'output';
    const format = document.getElementById('saveFormat').value;
    const quality = document.getElementById('saveQuality').value;

    const data = await apiPost('/api/save', { filename, format, quality: parseInt(quality) });
    if (data) {
        showToast(`Saved as ${data.filename}`, 'success');
        const modal = bootstrap.Modal.getInstance(document.getElementById('saveModal'));
        if (modal) modal.hide();
    }
}

function downloadImage() {
    const format = document.getElementById('saveFormat')?.value || 'png';
    window.open(`/api/download?format=${format}`, '_blank');
}

// =============================================================================
// IMAGE ENHANCEMENT
// =============================================================================

async function applyBrightness() {
    const val = document.getElementById('brightnessSlider').value;
    const data = await apiPost('/api/enhance', { operation: 'brightness', value: parseInt(val) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); }
}

async function applyContrast() {
    const val = document.getElementById('contrastSlider').value;
    const data = await apiPost('/api/enhance', { operation: 'contrast', alpha: parseFloat(val) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); }
}

async function applyHistogramEq() {
    const data = await apiPost('/api/enhance', { operation: 'histogram_eq' });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); showToast('Histogram equalization applied', 'success'); }
}

async function applySharpen() {
    const val = document.getElementById('sharpenSlider').value;
    const data = await apiPost('/api/enhance', { operation: 'sharpen', strength: parseFloat(val) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); }
}

async function applySmooth() {
    const method = document.getElementById('smoothMethod').value;
    const ksize = document.getElementById('smoothKsize').value;
    const data = await apiPost('/api/enhance', { operation: 'smooth', method, ksize: parseInt(ksize) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); }
}

// =============================================================================
// GEOMETRIC TRANSFORMATION
// =============================================================================

async function applyRotate() {
    const angle = document.getElementById('rotateSlider').value;
    const interp = document.getElementById('interpMethod').value;
    const data = await apiPost('/api/transform', { operation: 'rotate', angle: parseFloat(angle), interpolation: interp });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); }
}

async function applyFlip(direction) {
    const data = await apiPost('/api/transform', { operation: 'flip', direction });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); showToast(`Flipped ${direction}`, 'success'); }
}

async function applyResize() {
    const w = document.getElementById('resizeWidth').value;
    const h = document.getElementById('resizeHeight').value;
    const interp = document.getElementById('interpMethod').value;
    const data = await apiPost('/api/transform', {
        operation: 'resize',
        width: w ? parseInt(w) : null,
        height: h ? parseInt(h) : null,
        interpolation: interp
    });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); showToast('Image resized', 'success'); }
}

async function applyTranslate() {
    const tx = document.getElementById('translateX').value || 0;
    const ty = document.getElementById('translateY').value || 0;
    const data = await apiPost('/api/transform', { operation: 'translate', tx: parseInt(tx), ty: parseInt(ty) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); }
}

async function applyCrop() {
    if (!state.cropRect) {
        showToast('Please draw a crop area on the image first', 'info');
        return;
    }
    const data = await apiPost('/api/transform', {
        operation: 'crop',
        x: state.cropRect.x,
        y: state.cropRect.y,
        w: state.cropRect.w,
        h: state.cropRect.h
    });
    if (data) {
        updateAfterImage(data.image);
        updateImageInfo(data.info);
        updateHistogram();
        toggleCropMode(false);
        showToast('Image cropped', 'success');
    }
}

// =============================================================================
// IMAGE RESTORATION
// =============================================================================

async function applyRestore(operation) {
    const ksize = document.getElementById('restoreKsize').value;
    const data = await apiPost('/api/restore', { operation, ksize: parseInt(ksize) });
    if (data) {
        updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram();
        const names = { gaussian: 'Gaussian Blur', median: 'Median Filter', salt_pepper: 'Salt & Pepper Removal' };
        showToast(`${names[operation]} applied`, 'success');
    }
}

// =============================================================================
// BINARY & EDGE PROCESSING
// =============================================================================

async function applyThreshold() {
    const thresh = document.getElementById('thresholdSlider').value;
    const data = await apiPost('/api/edge', { operation: 'threshold', thresh: parseInt(thresh) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); }
}

async function applyEdge(method) {
    const params = { operation: method };
    if (method === 'canny') {
        params.t1 = parseInt(document.getElementById('cannyT1').value || 50);
        params.t2 = parseInt(document.getElementById('cannyT2').value || 150);
    }
    if (method === 'log') {
        params.ksize = parseInt(document.getElementById('logKsize').value || 5);
    }
    const data = await apiPost('/api/edge', params);
    if (data) {
        updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram();
        showToast(`${method.charAt(0).toUpperCase() + method.slice(1)} edge detection applied`, 'success');
    }
}

async function applyMorphology(operation) {
    const ksize = document.getElementById('morphKsize').value;
    const iter = document.getElementById('morphIter').value;
    const data = await apiPost('/api/edge', { operation, ksize: parseInt(ksize), iterations: parseInt(iter) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); showToast(`${operation} applied`, 'success'); }
}

// =============================================================================
// COLOR PROCESSING
// =============================================================================

async function applyGrayscale() {
    const data = await apiPost('/api/color', { operation: 'grayscale' });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); showToast('Converted to grayscale', 'success'); }
}

async function applySplitChannel(channel) {
    const data = await apiPost('/api/color', { operation: 'split', channel });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); showToast(`${channel} channel extracted`, 'success'); }
}

async function applyHueSaturation() {
    const hue = document.getElementById('hueSlider').value;
    const sat = document.getElementById('satSlider').value;
    const data = await apiPost('/api/color', { operation: 'hue_saturation', hue: parseInt(hue), saturation: parseFloat(sat) });
    if (data) { updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram(); }
}

// =============================================================================
// SEGMENTATION
// =============================================================================

async function applySegment(operation) {
    const params = { operation };
    if (operation === 'threshold') {
        params.thresh = parseInt(document.getElementById('segThreshold').value || 128);
    } else if (operation === 'region') {
        params.regions = parseInt(document.getElementById('segRegions').value || 4);
    }
    const data = await apiPost('/api/segment', params);
    if (data) {
        updateAfterImage(data.image); updateImageInfo(data.info); updateHistogram();
        showToast(`${operation} segmentation applied`, 'success');
    }
}

// =============================================================================
// COMPRESSION
// =============================================================================

async function applyCompression() {
    const method = document.getElementById('compressMethod').value;
    const params = { method };
    if (method === 'quantize') {
        params.levels = parseInt(document.getElementById('quantLevels').value || 16);
    }
    const data = await apiPost('/api/compress', params);
    if (data) {
        updateAfterImage(data.image);
        updateImageInfo(data.info);
        updateHistogram();
        if (data.stats) {
            displayCompressionStats(data.stats);
        }
        showToast(`${data.stats?.method || method} compression done`, 'success');
    }
}

function displayCompressionStats(stats) {
    const container = document.getElementById('compressStats');
    container.innerHTML = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${stats.ratio}x</div>
                <div class="stat-label">Compression Ratio</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.savings}%</div>
                <div class="stat-label">Space Saved</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${(stats.original_bits / 8 / 1024).toFixed(1)}KB</div>
                <div class="stat-label">Original Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${stats.time}s</div>
                <div class="stat-label">Time Taken</div>
            </div>
        </div>
    `;
    container.style.display = 'block';
}

// =============================================================================
// CNN DETECTION
// =============================================================================

async function detectObjects() {
    const data = await apiPost('/api/detect');
    if (data) {
        updateAfterImage(data.image);
        if (data.predictions) {
            displayPredictions(data.predictions);
        }
        showToast('Object detection complete!', 'success');
    }
}

function displayPredictions(predictions) {
    const container = document.getElementById('cnnResults');
    let html = '<ul class="prediction-list">';
    predictions.forEach((pred, i) => {
        html += `
            <li class="prediction-item">
                <div class="prediction-rank">${i + 1}</div>
                <div style="flex:1">
                    <div class="prediction-name">${pred.description}</div>
                    <div class="prediction-bar">
                        <div class="prediction-bar-fill" style="width:${pred.confidence}%"></div>
                    </div>
                </div>
                <div class="prediction-conf">${pred.confidence}%</div>
            </li>
        `;
    });
    html += '</ul>';
    container.innerHTML = html;
    container.style.display = 'block';
}

// =============================================================================
// HISTOGRAM
// =============================================================================

let histBeforeChart = null;
let histAfterChart = null;

async function updateHistogram() {
    if (!state.imageLoaded) return;
    const data = await apiCall('/api/histogram');
    if (!data || !data.histograms) return;

    const labels = Array.from({ length: 256 }, (_, i) => i);

    if (data.histograms.before) {
        renderHistogram('histBefore', data.histograms.before, labels, 'before');
    }
    if (data.histograms.after) {
        renderHistogram('histAfter', data.histograms.after, labels, 'after');
    }
}

function renderHistogram(canvasId, histData, labels, which) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Destroy existing chart
    if (which === 'before' && histBeforeChart) histBeforeChart.destroy();
    if (which === 'after' && histAfterChart) histAfterChart.destroy();

    const datasets = [];
    if (histData.red) {
        datasets.push({ label: 'R', data: histData.red, borderColor: 'rgba(239,68,68,0.8)', backgroundColor: 'rgba(239,68,68,0.15)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.3 });
        datasets.push({ label: 'G', data: histData.green, borderColor: 'rgba(34,197,94,0.8)', backgroundColor: 'rgba(34,197,94,0.15)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.3 });
        datasets.push({ label: 'B', data: histData.blue, borderColor: 'rgba(59,130,246,0.8)', backgroundColor: 'rgba(59,130,246,0.15)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.3 });
    }
    if (histData.gray) {
        datasets.push({ label: 'Gray', data: histData.gray, borderColor: 'rgba(200,200,220,0.8)', backgroundColor: 'rgba(200,200,220,0.1)', borderWidth: 1, pointRadius: 0, fill: true, tension: 0.3 });
    }

    const chart = new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            animation: { duration: 400 }
        }
    });

    if (which === 'before') histBeforeChart = chart;
    else histAfterChart = chart;
}

// =============================================================================
// CROP TOOL
// =============================================================================

function toggleCropMode(enable) {
    const overlay = document.getElementById('cropOverlay');
    const btn = document.getElementById('cropToggleBtn');
    if (enable === undefined) enable = !state.cropMode;
    state.cropMode = enable;

    if (enable) {
        overlay.classList.add('active');
        btn.classList.add('primary');
        showToast('Draw a rectangle on the image to crop', 'info');
    } else {
        overlay.classList.remove('active');
        btn.classList.remove('primary');
        const rect = document.getElementById('cropRect');
        if (rect) rect.style.display = 'none';
        state.cropRect = null;
    }
}

function initCropTool() {
    const overlay = document.getElementById('cropOverlay');
    const rect = document.getElementById('cropRect');
    const afterImg = document.getElementById('afterImage');

    let dragging = false;
    let startX, startY;

    overlay.addEventListener('mousedown', (e) => {
        dragging = true;
        const bounds = overlay.getBoundingClientRect();
        startX = e.clientX - bounds.left;
        startY = e.clientY - bounds.top;
        rect.style.display = 'block';
        rect.style.left = startX + 'px';
        rect.style.top = startY + 'px';
        rect.style.width = '0';
        rect.style.height = '0';
    });

    overlay.addEventListener('mousemove', (e) => {
        if (!dragging) return;
        const bounds = overlay.getBoundingClientRect();
        const curX = e.clientX - bounds.left;
        const curY = e.clientY - bounds.top;
        const x = Math.min(startX, curX);
        const y = Math.min(startY, curY);
        const w = Math.abs(curX - startX);
        const h = Math.abs(curY - startY);
        rect.style.left = x + 'px';
        rect.style.top = y + 'px';
        rect.style.width = w + 'px';
        rect.style.height = h + 'px';
    });

    overlay.addEventListener('mouseup', (e) => {
        if (!dragging) return;
        dragging = false;

        const bounds = overlay.getBoundingClientRect();
        const curX = e.clientX - bounds.left;
        const curY = e.clientY - bounds.top;

        // Convert screen coords to image coords
        const imgRect = afterImg.getBoundingClientRect();
        const overlayRect = overlay.getBoundingClientRect();
        const imgOffX = imgRect.left - overlayRect.left;
        const imgOffY = imgRect.top - overlayRect.top;
        const scaleX = afterImg.naturalWidth / imgRect.width;
        const scaleY = afterImg.naturalHeight / imgRect.height;

        const sx = (Math.min(startX, curX) - imgOffX) * scaleX;
        const sy = (Math.min(startY, curY) - imgOffY) * scaleY;
        const sw = Math.abs(curX - startX) * scaleX;
        const sh = Math.abs(curY - startY) * scaleY;

        state.cropRect = {
            x: Math.max(0, Math.round(sx)),
            y: Math.max(0, Math.round(sy)),
            w: Math.round(sw),
            h: Math.round(sh)
        };
    });
}

// =============================================================================
// SIDEBAR ACCORDION
// =============================================================================

function initSidebarAccordion() {
    document.querySelectorAll('.sidebar-section-header').forEach(header => {
        header.addEventListener('click', () => {
            const body = header.nextElementSibling;
            const wasActive = header.classList.contains('active');

            // Close all
            document.querySelectorAll('.sidebar-section-header').forEach(h => h.classList.remove('active'));
            document.querySelectorAll('.sidebar-section-body').forEach(b => b.classList.remove('show'));

            // Toggle current
            if (!wasActive) {
                header.classList.add('active');
                body.classList.add('show');
            }
        });
    });
}

// =============================================================================
// SLIDER VALUE DISPLAY
// =============================================================================

function initSliders() {
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const displayId = slider.dataset.display;
        if (displayId) {
            const display = document.getElementById(displayId);
            if (display) {
                display.textContent = slider.value;
                slider.addEventListener('input', () => {
                    display.textContent = slider.value;
                });
            }
        }
    });
}

// =============================================================================
// FILE UPLOAD (drag & drop + file input)
// =============================================================================

function initUpload() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');

    if (zone) {
        zone.addEventListener('click', () => input.click());
        zone.addEventListener('dragover', (e) => { e.preventDefault(); zone.classList.add('dragover'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                uploadImage(e.dataTransfer.files[0]);
            }
        });
    }

    if (input) {
        input.addEventListener('change', () => {
            if (input.files.length > 0) {
                uploadImage(input.files[0]);
            }
        });
    }
}

// =============================================================================
// HISTOGRAM TOGGLE
// =============================================================================

function toggleHistogram() {
    const panel = document.getElementById('histogramPanel');
    panel.classList.toggle('collapsed');
}

// =============================================================================
// INIT
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initSidebarAccordion();
    initSliders();
    initUpload();
    initCropTool();

    // Keyboard shortcut
    document.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'z') { e.preventDefault(); resetImage(); }
        if (e.ctrlKey && e.key === 'o') { e.preventDefault(); document.getElementById('fileInput').click(); }
        if (e.ctrlKey && e.key === 's') { e.preventDefault(); new bootstrap.Modal(document.getElementById('saveModal')).show(); }
    });
});
