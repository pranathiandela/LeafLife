// ─── Detection Page JS ────────────────────────────────────────
let selectedFile = null;

function togglePhotoGuide() {
    const guide = document.getElementById('photoGuide');
    const chevron = document.getElementById('guideChevron');
    const isOpen = guide.style.display !== 'none';
    guide.style.display = isOpen ? 'none' : 'block';
    chevron.style.transform = isOpen ? '' : 'rotate(180deg)';
}


const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');

// Drag & Drop
['dragenter','dragover'].forEach(e => {
    uploadZone?.addEventListener(e, (ev) => {
        ev.preventDefault();
        uploadZone.classList.add('drag-over');
    });
});
['dragleave','drop'].forEach(e => {
    uploadZone?.addEventListener(e, (ev) => {
        ev.preventDefault();
        uploadZone.classList.remove('drag-over');
    });
});
uploadZone?.addEventListener('drop', (ev) => {
    const file = ev.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) handleFile(file);
});
uploadZone?.addEventListener('click', (e) => {
    if (!e.target.closest('label')) fileInput?.click();
});
fileInput?.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('previewImg').src = e.target.result;
        document.getElementById('previewFilename').textContent = file.name;
        document.getElementById('previewSize').textContent = formatBytes(file.size);
        document.getElementById('uploadPreview').style.display = 'flex';
        document.getElementById('uploadActions').style.display = 'block';
        document.getElementById('uploadIconWrap').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

function formatBytes(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / 1048576).toFixed(1) + ' MB';
}

function clearUpload() {
    selectedFile = null;
    fileInput.value = '';
    document.getElementById('uploadPreview').style.display = 'none';
    document.getElementById('uploadActions').style.display = 'none';
    document.getElementById('uploadIconWrap').style.display = 'block';
}

async function analyzeImage() {
    if (!selectedFile) return;

    // Show loading
    document.getElementById('uploadSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultSection').style.display = 'none';

    // Animate loading steps
    const steps = ['ls1','ls2','ls3'];
    let si = 0;
    const stepInterval = setInterval(() => {
        steps.forEach((id,i) => document.getElementById(id)?.classList.toggle('active', i <= si));
        si++;
        if (si >= steps.length) clearInterval(stepInterval);
    }, 900);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
        const res = await fetch('/api/analyze', { method: 'POST', body: formData });
        const data = await res.json();

        clearInterval(stepInterval);

        if (data.error) {
            alert('Error: ' + data.error);
            resetDetection();
            return;
        }

        showResult(data);
    } catch (err) {
        clearInterval(stepInterval);
        alert('Failed to analyze image. Please try again.');
        resetDetection();
    }
}

function showResult(data) {
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'block';

    // Image
    const imgSrc = data.image_path ? `/static/${data.image_path}` : URL.createObjectURL(selectedFile);
    document.getElementById('resultImg').src = imgSrc;

    // Basic info
    document.getElementById('resultPlant').textContent = data.plant;
    document.getElementById('resultDisease').textContent = data.disease;
    document.getElementById('resultDesc').textContent = data.description;

    // Confidence
    const conf = data.confidence;
    document.getElementById('confVal').textContent = conf + '%';
    setTimeout(() => {
        document.getElementById('confFill').style.width = conf + '%';
    }, 100);

    // Show low-confidence warning if below threshold
    const warning = document.getElementById('confidenceWarning');
    if (warning) warning.style.display = conf < 75 ? 'flex' : 'none';

    // Severity badge
    const sev = (data.severity || 'medium').toLowerCase();
    const badge = document.getElementById('resultBadge');
    badge.textContent = (data.severity || 'Medium') + ' Risk';
    badge.className = `result-badge badge-${sev}`;

    const sevEl = document.getElementById('resultSeverity');
    sevEl.className = `result-severity severity-${sev}`;
    const icons = { high: '⚠️', medium: '🔔', none: '✅' };
    sevEl.innerHTML = `${icons[sev] || '🔔'} ${data.severity || 'Medium'} Risk`;

    // Lists
    populateList('symptomsList', data.symptoms || []);
    populateList('causesList', data.causes || []);
    populateList('preventionList', data.prevention || [], true);
    populateList('organicList', data.organic_treatment || []);
    populateList('chemicalList', data.chemical_treatment || []);
    populateList('recoveryList', data.recovery_tips || []);

    // Scroll to result
    document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function populateList(id, items, numbered = false) {
    const el = document.getElementById(id);
    if (!el) return;
    el.innerHTML = '';
    items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        el.appendChild(li);
    });
    if (items.length === 0) {
        el.innerHTML = '<li>No information available</li>';
    }
}

function resetDetection() {
    document.getElementById('uploadSection').style.display = 'block';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultSection').style.display = 'none';
    clearUpload();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
