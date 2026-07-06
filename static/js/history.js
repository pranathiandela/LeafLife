// ─── History Page JS ──────────────────────────────────────────
const ROWS_PER_PAGE = 10;
let currentPage = 1;
let filteredRows = [];

function getRows() {
    return Array.from(document.querySelectorAll('#historyBody .history-row'));
}

function filterHistory() {
    const search = (document.getElementById('searchInput')?.value || '').toLowerCase();
    const plant = (document.getElementById('plantFilter')?.value || '').toLowerCase();
    const severity = (document.getElementById('severityFilter')?.value || '').toLowerCase();
    const sort = document.getElementById('sortFilter')?.value || 'newest';

    let rows = getRows();

    filteredRows = rows.filter(row => {
        const rowPlant = (row.dataset.plant || '').toLowerCase();
        const rowDisease = (row.dataset.disease || '').toLowerCase();
        const rowSev = (row.dataset.severity || '').toLowerCase();

        const matchSearch = !search || rowPlant.includes(search) || rowDisease.includes(search);
        const matchPlant = !plant || rowPlant === plant;
        const matchSev = !severity || rowSev === severity.charAt(0).toUpperCase() + severity.slice(1);

        return matchSearch && matchPlant && matchSev;
    });

    // Sort
    filteredRows.sort((a, b) => {
        if (sort === 'newest') return new Date(b.dataset.date) - new Date(a.dataset.date);
        if (sort === 'oldest') return new Date(a.dataset.date) - new Date(b.dataset.date);
        if (sort === 'confidence') return parseFloat(b.dataset.confidence) - parseFloat(a.dataset.confidence);
        return 0;
    });

    const count = document.getElementById('recordCount');
    if (count) count.textContent = filteredRows.length + ' records';

    currentPage = 1;
    renderPage();
}

function renderPage() {
    const allRows = getRows();
    const hiddenSet = new Set(allRows);

    const start = (currentPage - 1) * ROWS_PER_PAGE;
    const pageRows = filteredRows.slice(start, start + ROWS_PER_PAGE);

    allRows.forEach(r => r.style.display = 'none');
    pageRows.forEach(r => {
        r.style.display = '';
        hiddenSet.delete(r);
    });

    renderPagination();
}

function renderPagination() {
    const totalPages = Math.ceil(filteredRows.length / ROWS_PER_PAGE);
    const container = document.getElementById('pagination');
    if (!container) return;

    container.innerHTML = '';
    if (totalPages <= 1) return;

    const makeBtn = (label, page, active = false) => {
        const btn = document.createElement('button');
        btn.className = 'page-btn' + (active ? ' active' : '');
        btn.textContent = label;
        btn.onclick = () => { currentPage = page; renderPage(); };
        return btn;
    };

    if (currentPage > 1) container.appendChild(makeBtn('«', currentPage - 1));

    for (let i = 1; i <= totalPages; i++) {
        if (i === 1 || i === totalPages || Math.abs(i - currentPage) <= 1) {
            container.appendChild(makeBtn(i, i, i === currentPage));
        } else if (Math.abs(i - currentPage) === 2) {
            const dots = document.createElement('span');
            dots.textContent = '...';
            dots.style.padding = '8px 6px';
            container.appendChild(dots);
        }
    }

    if (currentPage < totalPages) container.appendChild(makeBtn('»', currentPage + 1));
}

async function deleteRecord(id, btn) {
    if (!confirm('Delete this detection record?')) return;
    try {
        const res = await fetch(`/api/history/delete/${id}`, { method: 'DELETE' });
        const data = await res.json();
        if (data.success) {
            const row = btn.closest('tr');
            row.style.transition = 'opacity 0.3s';
            row.style.opacity = '0';
            setTimeout(() => { row.remove(); filterHistory(); }, 300);
        }
    } catch (e) {
        alert('Failed to delete record.');
    }
}

function viewRecord(id, plant, disease, confidence, severity, description, imagePath) {
    document.getElementById('modalPlant').textContent = '🌱 ' + plant;
    document.getElementById('modalDisease').textContent = disease;
    document.getElementById('modalConf').textContent = confidence + '%';
    document.getElementById('modalDesc').textContent = description;

    const sev = (severity || 'medium').toLowerCase();
    const sevEl = document.getElementById('modalSeverity');
    sevEl.className = `result-severity severity-${sev}`;
    sevEl.textContent = severity + ' Risk';

    const img = document.getElementById('modalImg');
    if (imagePath) {
        img.src = `/static/${imagePath}`;
        img.style.display = 'block';
    } else {
        img.style.display = 'none';
    }

    document.getElementById('viewModal').classList.add('open');
}

function closeModal() {
    document.getElementById('viewModal')?.classList.remove('open');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    filteredRows = getRows();
    renderPage();
});
