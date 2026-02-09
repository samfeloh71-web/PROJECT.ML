/**
 * Smart Segmentation Support Assistant - Chat Client
 * Handles communication with /chat endpoint and displays messages/charts
 */

// Toggle chat window visibility
function toggleChat() {
    const chatWindow = document.getElementById('chat-window');
    if (chatWindow.style.display === 'none') {
        chatWindow.style.display = 'flex';
        document.getElementById('chat-input').focus();
    } else {
        chatWindow.style.display = 'none';
    }
}

// Send chat message to /chat endpoint
function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    // Add user message to chat
    addMessageToChat(message, 'user');
    input.value = '';

    // Show typing indicator
    showTypingIndicator();

    // Send to backend
    fetch('/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: message })
    })
    .then(response => response.json())
    .then(data => {
        removeTypingIndicator();

        if (data.error) {
            addMessageToChat('Error: ' + data.error, 'ai');
        } else {
            // Add AI response
            addMessageToChat(data.response, 'ai');

            // Add chart if present
            if (data.chart && data.chart_type) {
                addChartToChat(data.chart, data.chart_type);
            }
        }
    })
    .catch(error => {
        removeTypingIndicator();
        console.error('Chat error:', error);
        addMessageToChat('Error: Could not reach the server. Please try again.', 'ai');
    });
}

// Add message to chat window
function addMessageToChat(text, sender) {
    const messagesDiv = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = escapeHtml(text).replace(/\n/g, '<br>');

    messageDiv.appendChild(contentDiv);
    messagesDiv.appendChild(messageDiv);

    // Auto-scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Add chart to chat window
function addChartToChat(chartJson, chartType) {
    const messagesDiv = document.getElementById('chat-messages');
    
    const chartWrapper = document.createElement('div');
    chartWrapper.className = 'chart-wrapper';

    const chartContainer = document.createElement('div');
    chartContainer.className = 'chart-container';
    chartContainer.id = 'chart-' + Date.now();

    chartWrapper.appendChild(chartContainer);
    messagesDiv.appendChild(chartWrapper);

    try {
        const chartData = JSON.parse(chartJson);
        Plotly.newPlot(chartContainer.id, chartData.data, chartData.layout, {
            responsive: true,
            displayModeBar: false
        });
    } catch (error) {
        console.error('Error rendering chart:', error);
        chartContainer.innerHTML = '<p style="color: red; padding: 10px;">Error displaying chart</p>';
    }

    // Auto-scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const messagesDiv = document.getElementById('chat-messages');
    
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typing-indicator';
    typingDiv.className = 'chat-message ai-message';
    
    const typingContent = document.createElement('div');
    typingContent.className = 'typing-indicator';
    typingContent.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    
    typingDiv.appendChild(typingContent);
    messagesDiv.appendChild(typingDiv);

    // Auto-scroll to bottom
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Handle Enter key in input
function handleChatKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}
