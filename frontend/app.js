// Configuration
const API_BASE = 'https://rag-chatbot-api-2x9v.onrender.com';

// DOM Elements
const trainForm = document.getElementById('train-form');
const urlInput = document.getElementById('url-input');
const trainBtn = document.getElementById('train-btn');
const trainStatus = document.getElementById('train-status');
const progressContainer = document.getElementById('progress-container');
const progressBar = document.getElementById('progress-bar');
const counterBadge = document.getElementById('counter-badge');

const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const chatBtn = document.getElementById('chat-btn');
const chatWindow = document.getElementById('chat-window');
const emptyState = document.getElementById('empty-state');

// State: Load from browser memory so it survives a refresh
let trainedWebsitesCount = parseInt(localStorage.getItem('rag_source_count')) || 0;
counterBadge.textContent = `${trainedWebsitesCount} Sources`;

// --- Train Logic ---
trainForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const url = urlInput.value;
    if (!url) return;

    trainBtn.disabled = true;
    trainBtn.textContent = 'Training...';
    trainStatus.classList.add('hidden');
    progressContainer.classList.remove('hidden');
    progressBar.style.width = '0%';
    
    let currentProgress = 0;
    const progressInterval = setInterval(() => {
        if (currentProgress < 90) {
            currentProgress += Math.random() * 10;
            if (currentProgress > 90) currentProgress = 90;
            progressBar.style.width = `${currentProgress}%`;
        }
    }, 500);

    try {
        console.log(`Sending URL to backend: ${url}`);
        const response = await fetch(`${API_BASE}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        const data = await response.json();
        console.log("Backend response:", data);

        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        trainStatus.classList.remove('hidden');

        if (data.status === 'success') {
            trainStatus.textContent = `✅ Success! Stored ${data.chunks} chunks.`;
            trainStatus.style.color = '#117a65';
            trainStatus.style.backgroundColor = '#e8f8f5';
            
            // Update Counter & Save to Memory
            trainedWebsitesCount++;
            localStorage.setItem('rag_source_count', trainedWebsitesCount);
            counterBadge.textContent = `${trainedWebsitesCount} Sources`;
        } else {
            trainStatus.textContent = `❌ Error: ${data.message}`;
            trainStatus.style.color = '#c0392b';
            trainStatus.style.backgroundColor = '#fadbd8';
        }
    } catch (error) {
        console.error("Network or Fetch Error:", error);
        clearInterval(progressInterval);
        trainStatus.classList.remove('hidden');
        trainStatus.textContent = `⚠️ Connection error. Is your Python backend running? Check console.`;
        trainStatus.style.color = '#c0392b';
        trainStatus.style.backgroundColor = '#fadbd8';
    }

    setTimeout(() => {
        progressContainer.classList.add('hidden');
        progressBar.style.width = '0%';
        trainBtn.disabled = false;
        trainBtn.textContent = 'Train';
        urlInput.value = '';
    }, 2000);
});

// --- Chat Logic ---
chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = chatInput.value.trim();
    if (!question) return;

    if (emptyState) emptyState.remove();
    appendMessage(question, 'user');
    chatInput.value = '';
    
    const thinkingId = appendMessage('Thinking...', 'bot', true);
    chatBtn.disabled = true;

    try {
        console.log(`Sending question: ${question}`);
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        const data = await response.json();
        console.log("Chat response:", data);

        const thinkingBubble = document.getElementById(thinkingId);
        if (thinkingBubble) thinkingBubble.remove();

        if (data.status === 'success') {
            appendMessage(data.answer, 'bot');
        } else {
            appendMessage(`❌ Backend Error: ${data.message}`, 'bot');
        }
    } catch (error) {
        console.error("Chat Fetch Error:", error);
        const thinkingBubble = document.getElementById(thinkingId);
        if (thinkingBubble) thinkingBubble.remove();
        appendMessage(`⚠️ Connection error. Check your Python terminal for crashes.`, 'bot');
    }

    chatBtn.disabled = false;
});

function appendMessage(text, sender, isThinking = false) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message-bubble', sender);
    if (isThinking) messageDiv.classList.add('thinking');
    
    const uniqueId = 'msg-' + Math.random().toString(36).substr(2, 9);
    messageDiv.id = uniqueId;
    
    const textNode = document.createElement('p');
    textNode.textContent = text;
    messageDiv.appendChild(textNode);
    
    chatWindow.appendChild(messageDiv);
    chatWindow.scrollTop = chatWindow.scrollHeight;
    
    return uniqueId;
}