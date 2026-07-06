// ─── Navbar scroll ────────────────────────────────────────────
window.addEventListener('scroll', () => {
    document.getElementById('navbar')?.classList.toggle('scrolled', window.scrollY > 10);
});

function toggleNav() {
    document.getElementById('navLinks')?.classList.toggle('open');
}

// ─── Language Selector ────────────────────────────────────────
function toggleLangMenu() {
    const dropdown = document.getElementById('langDropdown');
    const chevron  = document.getElementById('langChevron');
    dropdown?.classList.toggle('open');
    if (chevron) chevron.style.transform = dropdown?.classList.contains('open') ? 'rotate(180deg)' : '';
}

function setLang(code, label) {
    // Update button label
    const el = document.getElementById('currentLang');
    if (el) el.textContent = label;

    // Mark active option
    document.querySelectorAll('.lang-option').forEach(o => o.classList.remove('active'));
    event.currentTarget.classList.add('active');

    // Close dropdown
    document.getElementById('langDropdown')?.classList.remove('open');
    const chevron = document.getElementById('langChevron');
    if (chevron) chevron.style.transform = '';

    // Save to localStorage
    localStorage.setItem('ll_lang', code);
    localStorage.setItem('ll_lang_label', label);

    // Trigger Google Translate
    if (code === 'en') {
        // Restore original English
        const frame = document.querySelector('.goog-te-menu-frame') ||
                      document.querySelector('iframe.skiptranslate');
        // Use the cookie method to reset
        document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;';
        document.cookie = 'googtrans=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; domain=' + location.hostname;
        location.reload();
        return;
    }

    // Set Google Translate cookie and reload
    const domain = location.hostname;
    document.cookie = `googtrans=/en/${code}; path=/`;
    document.cookie = `googtrans=/en/${code}; path=/; domain=${domain}`;

    // Try using the Google Translate select element directly
    const tryTranslate = () => {
        const select = document.querySelector('.goog-te-combo');
        if (select) {
            select.value = code;
            select.dispatchEvent(new Event('change'));
        } else {
            // Fallback: reload with cookie set
            location.reload();
        }
    };

    // Give translate widget time to load
    setTimeout(tryTranslate, 500);
}

// Close dropdown on outside click
document.addEventListener('click', (e) => {
    const sel = document.getElementById('langSelector');
    if (sel && !sel.contains(e.target)) {
        document.getElementById('langDropdown')?.classList.remove('open');
        const chevron = document.getElementById('langChevron');
        if (chevron) chevron.style.transform = '';
    }
    // Close nav on outside click
    const navLinks  = document.getElementById('navLinks');
    const hamburger = document.getElementById('hamburger');
    if (navLinks && hamburger && !navLinks.contains(e.target) && !hamburger.contains(e.target)) {
        navLinks.classList.remove('open');
    }
});

// ─── Restore language on page load ───────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    const savedLabel = localStorage.getItem('ll_lang_label');
    const savedCode  = localStorage.getItem('ll_lang');

    if (savedLabel) {
        const el = document.getElementById('currentLang');
        if (el) el.textContent = savedLabel;
    }

    // Mark the correct active option
    if (savedCode) {
        document.querySelectorAll('.lang-option').forEach(o => {
            const onclick = o.getAttribute('onclick') || '';
            o.classList.toggle('active', onclick.includes(`'${savedCode}'`));
        });
    }

    // Apply stored translation via cookie if not English
    if (savedCode && savedCode !== 'en') {
        const existing = document.cookie.includes('googtrans');
        if (!existing) {
            document.cookie = `googtrans=/en/${savedCode}; path=/`;
        }
    }
});
