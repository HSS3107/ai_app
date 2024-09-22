document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    const analysisForm = document.getElementById('analysis-form');
    const queryForm = document.getElementById('query-form');

    // Dark mode toggle
    darkModeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    });

    // Check for saved dark mode preference
    if (localStorage.getItem('darkMode') === 'true') {
        document.body.classList.add('dark-mode');
    }

    if (chatForm) {
        chatForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const message = userInput.value.trim();
            if (message) {
                addMessage('user', message);
                processMessage(message);
                userInput.value = '';
            }
        });
    }

    if (analysisForm) {
        analysisForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const videoUrl = document.getElementById('video-url').value.trim();
            if (videoUrl) {
                analyzeTranscript(videoUrl);
            }
        });
    }

    if (queryForm) {
        queryForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const videoId = document.getElementById('video-id').value.trim();
            const query = document.getElementById('query').value.trim();
            if (videoId && query) {
                queryTranscript(videoId, query);
            }
        });
    }

    function formatMessage(content) {
        // Convert line breaks to <br> tags
        content = content.replace(/\n/g, '<br>');
    
        // Convert numbered lists
        content = content.replace(/(\d+\.)\s(.*?)(?=(?:\n\d+\.|\n\n|$))/gs, '<p class="numbered-item">$1 $2</p>');
    
        // Convert bullet points
        content = content.replace(/(-|\*)\s(.*?)(?=(?:\n(?:-|\*)|\n\n|$))/gs, '<li>$2</li>');
        content = content.replace(/<li>.*?<\/li>/gs, function(match) {
            return '<ul>' + match + '</ul>';
        });
    
        // Wrap paragraphs
        content = content.replace(/(?:<br>){2,}/g, '</p><p>');
        content = '<p>' + content + '</p>';
    
        return content;
    }
    
    function addMessage(sender, content) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', `${sender}-message`);
        
        const contentDiv = document.createElement('div');
        contentDiv.innerHTML = content.replace(/\n/g, '<br>');
        messageDiv.appendChild(contentDiv);
        
        const chatMessages = document.querySelector('.chat-messages');
        if (chatMessages) {
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    

    function processMessage(message) {
        if (isYouTubeURL(message)) {
            analyzeTranscript(message);
        } else {
            queryTranscript(localStorage.getItem('lastVideoId'), message);
        }
    }

    async function analyzeTranscript(url) {
        try {
            const response = await fetch('/api/analyze-transcript/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({video_url: url})
            });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            const result = `${data.message}\nVideo ID: ${data.video_id}`;
            if (chatMessages) {
                addMessage('assistant', result);
            } else {
                document.getElementById('transcript-result').textContent = result;
            }
            localStorage.setItem('lastVideoId', data.video_id);
        } catch (error) {
            console.error('Error:', error);
            const errorMessage = 'An error occurred while analyzing the transcript. Please try again.';
            if (chatMessages) {
                addMessage('assistant', errorMessage);
            } else {
                document.getElementById('transcript-result').textContent = errorMessage;
            }
        }
    }

    async function queryTranscript(videoId, query) {
        if (!videoId) {
            const errorMessage = 'Please analyze a YouTube video first before querying.';
            if (chatMessages) {
                addMessage('assistant', errorMessage);
            } else {
                document.getElementById('query-result').textContent = errorMessage;
            }
            return;
        }
        try {
            const response = await fetch('/api/query/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({video_id: videoId, query: query})
            });
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            const data = await response.json();
            if (chatMessages) {
                addMessage('assistant', data.result);
            } else {
                document.getElementById('query-result').textContent = data.result;
            }
        } catch (error) {
            console.error('Error:', error);
            const errorMessage = 'An error occurred while processing your query. Please try again.';
            if (chatMessages) {
                addMessage('assistant', errorMessage);
            } else {
                document.getElementById('query-result').textContent = errorMessage;
            }
        }
    }

    function isYouTubeURL(url) {
        return url.includes('youtube.com') || url.includes('youtu.be');
    }
});