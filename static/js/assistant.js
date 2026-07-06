// ─── AI Assistant JS ──────────────────────────────────────────
let chatId = null;
let messageHistory = [];
let isTyping = false;

function toggleSidebar() {
    document.getElementById('chatSidebar')?.classList.toggle('open');
}

function newChat() {
    chatId = null;
    messageHistory = [];
    document.getElementById('chatMessages').innerHTML = `
        <div class="welcome-msg" id="welcomeMsg">
            <div class="welcome-icon"><i class="fas fa-robot"></i></div>
            <h2>LeafLife AI Crop Assistant</h2>
            <p>Ask me anything about crop diseases, treatments, prevention methods, farming practices, and more. I'm here to help Indian farmers grow healthier crops!</p>
            <div class="welcome-chips">
                <div class="suggestion-chip" onclick="askSuggestion('What are the common diseases of tomato plants?')">Tomato diseases</div>
                <div class="suggestion-chip" onclick="askSuggestion('How do I identify rice brown spot disease?')">Rice brown spot</div>
                <div class="suggestion-chip" onclick="askSuggestion('Neem oil spray preparation and usage for crops')">Neem oil spray</div>
                <div class="suggestion-chip" onclick="askSuggestion('How can I improve soil health for better yields?')">Soil health tips</div>
            </div>
        </div>`;
    document.getElementById('chatTitle').textContent = 'LeafLife AI Crop Assistant';
    document.querySelectorAll('.chat-list-item').forEach(i => i.classList.remove('active'));
    document.getElementById('chatSidebar')?.classList.remove('open');
}

async function loadChat(id, title) {
    chatId = id;
    messageHistory = [];
    document.getElementById('chatTitle').textContent = title;
    document.querySelectorAll('.chat-list-item').forEach(i => i.classList.remove('active'));
    document.getElementById(`chat-${id}`)?.classList.add('active');

    const messagesEl = document.getElementById('chatMessages');
    messagesEl.innerHTML = '<div style="text-align:center;color:var(--gray-400);padding:2rem"><i class="fas fa-spinner fa-spin"></i> Loading conversation...</div>';

    try {
        const res = await fetch(`/api/chat/${id}/messages`);
        const messages = await res.json();
        messagesEl.innerHTML = '';
        messageHistory = [];

        messages.forEach(msg => {
            messageHistory.push({ role: msg.role, content: msg.content });
            appendMessage(msg.role, msg.content, false);
        });

        messagesEl.scrollTop = messagesEl.scrollHeight;
    } catch (e) {
        messagesEl.innerHTML = '<div style="text-align:center;color:var(--red-500);padding:2rem">Failed to load conversation.</div>';
    }

    document.getElementById('chatSidebar')?.classList.remove('open');
}

function askSuggestion(text) {
    document.getElementById('chatInput').value = text;
    sendMessage();
}

async function sendMessage() {
    if (isTyping) return;

    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    if (!message) return;

    input.value = '';
    autoResize(input);

    // Hide welcome
    document.getElementById('welcomeMsg')?.remove();

    // Append user message
    appendMessage('user', message);
    messageHistory.push({ role: 'user', content: message });

    // Show typing indicator
    showTyping();
    document.getElementById('sendBtn').disabled = true;
    isTyping = true;

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                chat_id: chatId,
                history: messageHistory.slice(-20)
            })
        });

        const data = await res.json();
        removeTyping();

        if (data.error) {
            appendMessage('ai', '⚠️ Sorry, I encountered an error. Please try again.');
        } else {
            chatId = data.chat_id;
            appendMessage('ai', data.reply);
            messageHistory.push({ role: 'assistant', content: data.reply });
            updateChatList(chatId, message);
        }
    } catch (e) {
        removeTyping();
        appendMessage('ai', '⚠️ Connection error. Please check your network and try again.');
    } finally {
        document.getElementById('sendBtn').disabled = false;
        isTyping = false;
    }
}

function appendMessage(role, content, scroll = true) {
    const messagesEl = document.getElementById('chatMessages');
    const isUser = role === 'user';

    const row = document.createElement('div');
    row.className = `msg-row ${isUser ? 'user' : 'ai'}`;

    const avatar = document.createElement('div');
    avatar.className = `msg-avatar ${isUser ? 'user' : 'ai'}`;
    avatar.innerHTML = isUser ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    bubble.innerHTML = formatMarkdown(content);

    const time = document.createElement('div');
    time.className = 'msg-time';
    time.textContent = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });

    const contentWrap = document.createElement('div');
    contentWrap.appendChild(bubble);
    contentWrap.appendChild(time);

    if (isUser) {
        row.appendChild(contentWrap);
        row.appendChild(avatar);
    } else {
        row.appendChild(avatar);
        row.appendChild(contentWrap);
    }

    messagesEl.appendChild(row);
    if (scroll) messagesEl.scrollTop = messagesEl.scrollHeight;
}

function formatMarkdown(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/^### (.*$)/gm, '<h4 style="margin:0.5rem 0 0.25rem;font-weight:700">$1</h4>')
        .replace(/^## (.*$)/gm, '<h3 style="margin:0.5rem 0 0.25rem;font-weight:700">$1</h3>')
        .replace(/^# (.*$)/gm, '<h3 style="margin:0.5rem 0 0.25rem;font-weight:800">$1</h3>')
        .replace(/^- (.*$)/gm, '<li>$1</li>')
        .replace(/^(\d+)\. (.*$)/gm, '<li>$2</li>')
        .replace(/(<li>.*<\/li>)+/gs, (m) => `<ul style="padding-left:1.2rem;margin:0.4rem 0">${m}</ul>`)
        .replace(/\n\n/g, '<br><br>')
        .replace(/\n/g, '<br>');
}

function showTyping() {
    const messagesEl = document.getElementById('chatMessages');
    const row = document.createElement('div');
    row.className = 'msg-row ai';
    row.id = 'typingRow';
    row.innerHTML = `
        <div class="msg-avatar ai"><i class="fas fa-robot"></i></div>
        <div class="msg-bubble" style="background:#fff;border:1px solid var(--gray-100)">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>`;
    messagesEl.appendChild(row);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removeTyping() {
    document.getElementById('typingRow')?.remove();
}

function updateChatList(id, title) {
    const list = document.getElementById('chatList');
    let existing = document.getElementById(`chat-${id}`);
    if (!existing) {
        existing = document.createElement('div');
        existing.className = 'chat-list-item';
        existing.id = `chat-${id}`;
        existing.onclick = () => loadChat(id, title.slice(0, 50));
        existing.innerHTML = `
            <i class="fas fa-comment-dots"></i>
            <div class="chat-list-title">${title.slice(0, 45)}${title.length > 45 ? '...' : ''}</div>
            <span class="chat-list-date">${new Date().toISOString().split('T')[0]}</span>`;
        list.prepend(existing);
    }
    document.querySelectorAll('.chat-list-item').forEach(i => i.classList.remove('active'));
    existing.classList.add('active');
}

function clearChat() {
    if (confirm('Clear current chat? This will start a new conversation.')) newChat();
}

function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
