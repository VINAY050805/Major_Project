// Enhanced JavaScript for PlantDetect

// DOM Content Loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavbar();
    initAnimations();
    initFileUpload();
    initChatSystem();
    initHistoryFilters();
    
    // Add scroll effects
    window.addEventListener('scroll', handleScroll);
});

// Navbar Scroll Effect
function initNavbar() {
    const navbar = document.querySelector('.navbar');
    if (navbar) {
        window.addEventListener('scroll', () => {
            if (window.scrollY > 50) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }
}

// Scroll Animation Handler
function handleScroll() {
    const elements = document.querySelectorAll('.feature-card, .step, .stat');
    
    elements.forEach(element => {
        const elementTop = element.getBoundingClientRect().top;
        const elementBottom = element.getBoundingClientRect().bottom;
        const isVisible = (elementTop < window.innerHeight - 100) && (elementBottom > 0);
        
        if (isVisible) {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }
    });
}

// Animation Initialization
function initAnimations() {
    // Add animation classes to elements
    const animatedElements = document.querySelectorAll('.feature-card, .step, .stat');
    
    animatedElements.forEach((element, index) => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        element.style.transitionDelay = `${index * 0.1}s`;
    });
    
    // Trigger animations on load
    setTimeout(() => {
        handleScroll();
    }, 500);
}

// File Upload System
function initFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImage = document.getElementById('previewImage');
    const previewText = document.getElementById('previewText');
    const uploadBtn = document.getElementById('uploadBtn');
    const uploadForm = document.querySelector('.upload-form');
    
    if (fileInput && imagePreview) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    previewImage.style.display = 'block';
                    previewText.style.display = 'none';
                    imagePreview.style.borderColor = '#10b981';
                    
                    // Enable upload button
                    if (uploadBtn) {
                        uploadBtn.disabled = false;
                    }
                    
                    // Add success animation
                    imagePreview.classList.add('pulse');
                    setTimeout(() => {
                        imagePreview.classList.remove('pulse');
                    }, 2000);
                };
                
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        imagePreview.addEventListener('dragover', function(e) {
            e.preventDefault();
            this.style.borderColor = '#2563eb';
            this.style.backgroundColor = '#f0f9ff';
        });
        
        imagePreview.addEventListener('dragleave', function(e) {
            e.preventDefault();
            this.style.borderColor = '#e2e8f0';
            this.style.backgroundColor = '#f8fafc';
        });
        
        imagePreview.addEventListener('drop', function(e) {
            e.preventDefault();
            this.style.borderColor = '#e2e8f0';
            this.style.backgroundColor = '#f8fafc';
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                fileInput.files = e.dataTransfer.files;
                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        });
    }
    
    // Form submission handler - REMOVED the fake analysis
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('fileInput');
            if (!fileInput.files.length) {
                e.preventDefault();
                showNotification('Please select an image first!', 'error');
                return;
            }
            
            // Show loading state - but don't prevent actual form submission
            const uploadBtn = document.getElementById('uploadBtn');
            if (uploadBtn) {
                const originalText = uploadBtn.innerHTML;
                uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
                uploadBtn.disabled = true;
                
                // Revert button after 5 seconds if something goes wrong
                setTimeout(() => {
                    uploadBtn.innerHTML = originalText;
                    uploadBtn.disabled = false;
                }, 5000);
            }
            
            // Let the form submit normally to the backend
        });
    }
}

// Show Results Function - Only for demo purposes
function showResults() {
    const resultsSection = document.getElementById('resultsSection');
    const chatContainer = document.getElementById('chatContainer');
    
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 500);
    }
    
    // Show chat container after a delay
    setTimeout(() => {
        if (chatContainer) {
            chatContainer.style.display = 'block';
            chatContainer.classList.add('slide-up');
        }
    }, 1500);
}

// Chat System
function initChatSystem() {
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const chatBox = document.getElementById('chatBox');
    const quickQuestions = document.querySelectorAll('.quick-question');
    const getHelpBtn = document.getElementById('getHelpBtn');
    
    // Send message function
    function sendMessage() {
        const message = chatInput.value.trim();
        if (!message) return;
        
        // Add user message
        addMessage(message, 'user');
        chatInput.value = '';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Get disease from the page
        const diseaseElement = document.getElementById('prediction-text');
        const disease = diseaseElement ? diseaseElement.textContent : 'plant disease';
        const languageSelect = document.getElementById('languageSelect');
        const language = languageSelect ? languageSelect.value : 'english';
        
        // Call backend for AI response
        fetch('/ask_gemini', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            },
            body: JSON.stringify({ 
                question: message, 
                disease: disease, 
                language: language
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            hideTypingIndicator();
            addMessage(data.response, 'bot');
        })
        .catch(error => {
            console.error('Error:', error);
            hideTypingIndicator();
            addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot');
        });
    }
    
    // Send button click
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    
    // Enter key press
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
    
    // Quick questions
    quickQuestions.forEach(question => {
        question.addEventListener('click', function() {
            const questionText = this.textContent;
            if (chatInput) {
                chatInput.value = questionText;
                sendMessage();
            }
        });
    });
    
    // Get Help button
    if (getHelpBtn) {
        getHelpBtn.addEventListener('click', function() {
            const chatContainer = document.getElementById('chatContainer');
            if (chatContainer) {
                if (chatContainer.style.display === 'block') {
                    chatContainer.style.display = 'none';
                } else {
                    chatContainer.style.display = 'block';
                    chatContainer.classList.add('slide-up');
                    
                    // Scroll to chat
                    setTimeout(() => {
                        chatContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
                    }, 500);
                }
            }
        });
    }
}

// Add message to chat
function addMessage(text, sender) {
    const chatBox = document.getElementById('chatBox');
    if (!chatBox) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    messageDiv.textContent = text;
    
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Show typing indicator
function showTypingIndicator() {
    const chatBox = document.getElementById('chatBox');
    const typingIndicator = document.getElementById('typingIndicator');
    
    if (typingIndicator && chatBox) {
        typingIndicator.style.display = 'flex';
        chatBox.appendChild(typingIndicator);
        chatBox.scrollTop = chatBox.scrollHeight;
    }
}

// Hide typing indicator
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.style.display = 'none';
    }
}

// History Filters
function initHistoryFilters() {
    const filterOptions = document.querySelectorAll('.filter-option');
    const historyCards = document.querySelectorAll('.history-card');
    const searchInput = document.getElementById('searchInput');
    
    // Filter by status
    if (filterOptions.length > 0) {
        filterOptions.forEach(option => {
            option.addEventListener('click', function() {
                const filter = this.getAttribute('data-filter');
                
                // Update active state
                filterOptions.forEach(opt => opt.classList.remove('active'));
                this.classList.add('active');
                
                // Filter cards
                historyCards.forEach(card => {
                    if (filter === 'all' || card.getAttribute('data-status') === filter) {
                        card.style.display = 'block';
                        setTimeout(() => {
                            card.style.opacity = '1';
                            card.style.transform = 'translateY(0)';
                        }, 100);
                    } else {
                        card.style.opacity = '0';
                        card.style.transform = 'translateY(20px)';
                        setTimeout(() => {
                            card.style.display = 'none';
                        }, 300);
                    }
                });
            });
        });
    }
    
    // Search functionality
    if (searchInput) {
        searchInput.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase();
            
            historyCards.forEach(card => {
                const title = card.querySelector('.card-title').textContent.toLowerCase();
                const status = card.querySelector('.card-status').textContent.toLowerCase();
                
                if (title.includes(searchTerm) || status.includes(searchTerm)) {
                    card.style.display = 'block';
                    setTimeout(() => {
                        card.style.opacity = '1';
                        card.style.transform = 'translateY(0)';
                    }, 100);
                } else {
                    card.style.opacity = '0';
                    card.style.transform = 'translateY(20px)';
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                }
            });
        });
    }
}

// Notification System
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    // Add styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${getNotificationColor(type)};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 10px 15px rgba(0,0,0,0.1);
        z-index: 10000;
        transform: translateX(400px);
        transition: transform 0.3s ease;
        max-width: 350px;
    `;
    
    document.body.appendChild(notification);
    
    // Animate in
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    // Remove after delay
    setTimeout(() => {
        notification.style.transform = 'translateX(400px)';
        setTimeout(() => {
            if (document.body.contains(notification)) {
                document.body.removeChild(notification);
            }
        }, 300);
    }, 4000);
}

// Helper function for notification icons
function getNotificationIcon(type) {
    const icons = {
        'success': 'check-circle',
        'error': 'exclamation-circle',
        'warning': 'exclamation-triangle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Helper function for notification colors
function getNotificationColor(type) {
    const colors = {
        'success': 'linear-gradient(135deg, #10b981, #059669)',
        'error': 'linear-gradient(135deg, #ef4444, #dc2626)',
        'warning': 'linear-gradient(135deg, #f59e0b, #d97706)',
        'info': 'linear-gradient(135deg, #3b82f6, #2563eb)'
    };
    return colors[type] || colors.info;
}

// Language Selector
function initLanguageSelector() {
    const languageSelect = document.getElementById('languageSelect');
    if (languageSelect) {
        languageSelect.addEventListener('change', function() {
            const selectedLanguage = this.value;
            showNotification(`Language changed to ${this.options[this.selectedIndex].text}`, 'info');
        });
    }
}

// Initialize language selector
document.addEventListener('DOMContentLoaded', initLanguageSelector);

// Check if we have results on page load and show them
document.addEventListener('DOMContentLoaded', function() {
    // Check if results section exists and should be visible
    const resultsSection = document.getElementById('resultsSection');
    const predictionText = document.getElementById('prediction-text');
    
    if (resultsSection && predictionText && predictionText.textContent.trim() !== '') {
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
    }
});

// Export functions for global access
window.PlantDetect = {
    showNotification,
    initFileUpload,
    initChatSystem
};