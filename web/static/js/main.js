// MemVid Web Platform - Main JavaScript

// Global configuration
const CONFIG = {
    API_BASE: '/api',
    WS_URL: `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`,
    REFRESH_INTERVAL: 30000, // 30 seconds
    SEARCH_DEBOUNCE: 500, // 500ms
};

// Global state
let globalState = {
    user: { id: 'anonymous', name: 'Anonymous' },
    memories: [],
    websocket: null,
    isConnected: false,
};

// Utility functions
const utils = {
    // Format bytes to human readable format
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Format date to human readable format
    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diffTime = Math.abs(now - date);
        const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 7) return `${diffDays} days ago`;
        if (diffDays < 30) return `${Math.ceil(diffDays / 7)} weeks ago`;
        if (diffDays < 365) return `${Math.ceil(diffDays / 30)} months ago`;
        return `${Math.ceil(diffDays / 365)} years ago`;
    },

    // Debounce function
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Show toast notification
    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <i class="fas fa-${this.getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;

        // Add toast styles if not already present
        if (!document.getElementById('toast-styles')) {
            const styles = document.createElement('style');
            styles.id = 'toast-styles';
            styles.textContent = `
                .toast {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                    padding: 1rem;
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    z-index: 10000;
                    animation: slideIn 0.3s ease-out;
                    max-width: 400px;
                }
                .toast-success { border-left: 4px solid #28a745; }
                .toast-error { border-left: 4px solid #dc3545; }
                .toast-warning { border-left: 4px solid #ffc107; }
                .toast-info { border-left: 4px solid #17a2b8; }
                .toast-content { display: flex; align-items: center; gap: 0.5rem; flex: 1; }
                .toast-close { background: none; border: none; cursor: pointer; color: #666; }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(styles);
        }

        document.body.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    },

    getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || 'info-circle';
    },

    // Copy text to clipboard
    async copyToClipboard(text) {
        try {
            await navigator.clipboard.writeText(text);
            this.showToast('Copied to clipboard!', 'success');
        } catch (err) {
            console.error('Failed to copy text: ', err);
            this.showToast('Failed to copy text', 'error');
        }
    },

    // Generate random ID
    generateId() {
        return Math.random().toString(36).substr(2, 9);
    },

    // Validate email
    isValidEmail(email) {
        const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return re.test(email);
    },

    // Sanitize HTML
    sanitizeHtml(str) {
        const temp = document.createElement('div');
        temp.textContent = str;
        return temp.innerHTML;
    }
};

// API functions
const api = {
    // Generic API call
    async call(endpoint, options = {}) {
        const url = `${CONFIG.API_BASE}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            },
        };

        const finalOptions = { ...defaultOptions, ...options };

        try {
            const response = await fetch(url, finalOptions);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                return await response.text();
            }
        } catch (error) {
            console.error('API call failed:', error);
            utils.showToast(`API Error: ${error.message}`, 'error');
            throw error;
        }
    },

    // Get memories
    async getMemories(userId = 'anonymous') {
        return this.call(`/memories?user_id=${userId}`);
    },

    // Get memory by ID
    async getMemory(memoryId) {
        return this.call(`/memories/${memoryId}`);
    },

    // Search memories
    async searchMemories(query, options = {}) {
        return this.call('/search', {
            method: 'POST',
            body: JSON.stringify({ query, ...options }),
        });
    },

    // Get analytics
    async getAnalytics(memoryId) {
        return this.call(`/analytics/${memoryId}`);
    },

    // Generate synthesis
    async generateSynthesis(query, memoryIds, synthesisType = 'summary') {
        return this.call('/synthesis', {
            method: 'POST',
            body: JSON.stringify({
                query,
                memory_ids: memoryIds,
                synthesis_type: synthesisType,
            }),
        });
    },

    // Get knowledge graph
    async getKnowledgeGraph(memoryId) {
        return this.call(`/knowledge-graph/${memoryId}`);
    },
};

// WebSocket management
const websocket = {
    connect() {
        if (globalState.websocket) {
            globalState.websocket.close();
        }

        try {
            globalState.websocket = new WebSocket(CONFIG.WS_URL);

            globalState.websocket.onopen = () => {
                console.log('WebSocket connected');
                globalState.isConnected = true;
                this.updateConnectionStatus('Connected');
                utils.showToast('Real-time connection established', 'success');
            };

            globalState.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };

            globalState.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                globalState.isConnected = false;
                this.updateConnectionStatus('Disconnected');
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    if (!globalState.isConnected) {
                        this.connect();
                    }
                }, 3000);
            };

            globalState.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus('Error');
            };
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            utils.showToast('Failed to establish real-time connection', 'error');
        }
    },

    send(message) {
        if (globalState.websocket && globalState.websocket.readyState === WebSocket.OPEN) {
            globalState.websocket.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, message not sent:', message);
        }
    },

    handleMessage(message) {
        console.log('WebSocket message received:', message);
        
        switch (message.type) {
            case 'activity':
                this.handleActivityMessage(message);
                break;
            case 'memory_update':
                this.handleMemoryUpdate(message);
                break;
            case 'search_result':
                this.handleSearchResult(message);
                break;
            case 'collaboration':
                this.handleCollaboration(message);
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    },

    handleActivityMessage(message) {
        // Add to activity feed if present
        const activityFeed = document.getElementById('activityFeed');
        if (activityFeed) {
            const item = document.createElement('div');
            item.className = 'activity-item';
            item.innerHTML = `
                <i class="${message.icon || 'fas fa-info-circle'}"></i>
                <span>${message.content}</span>
                <time>${new Date().toLocaleTimeString()}</time>
            `;
            activityFeed.insertBefore(item, activityFeed.firstChild);

            // Keep only the last 10 items
            while (activityFeed.children.length > 10) {
                activityFeed.removeChild(activityFeed.lastChild);
            }
        }
    },

    handleMemoryUpdate(message) {
        // Refresh memory list if on memories page
        if (window.location.pathname.includes('/memories')) {
            // Trigger memory reload
            if (typeof loadMemories === 'function') {
                loadMemories();
            }
        }
        utils.showToast('Memory updated', 'info');
    },

    handleSearchResult(message) {
        // Handle real-time search results
        if (typeof displaySearchResults === 'function') {
            displaySearchResults(message.results);
        }
    },

    handleCollaboration(message) {
        utils.showToast(`Collaboration: ${message.content}`, 'info');
    },

    updateConnectionStatus(status) {
        const statusElement = document.getElementById('connectionStatus');
        if (statusElement) {
            statusElement.textContent = status;
            statusElement.className = `connection-status ${status.toLowerCase()}`;
        }
    },

    disconnect() {
        if (globalState.websocket) {
            globalState.websocket.close();
            globalState.websocket = null;
            globalState.isConnected = false;
        }
    }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Initialize WebSocket connection
    websocket.connect();

    // Set up global keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K for quick search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('searchInput') || document.getElementById('searchQuery');
            if (searchInput) {
                searchInput.focus();
            } else if (typeof showQuickSearch === 'function') {
                showQuickSearch();
            }
        }

        // Escape to close modals
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal, .create-memory-modal');
            modals.forEach(modal => {
                if (modal.style.display !== 'none') {
                    modal.style.display = 'none';
                }
            });
        }
    });

    // Set up periodic data refresh
    setInterval(() => {
        if (globalState.isConnected) {
            // Refresh data periodically
            if (typeof refreshData === 'function') {
                refreshData();
            }
        }
    }, CONFIG.REFRESH_INTERVAL);

    console.log('MemVid Web Platform initialized');
});

// Cleanup on page unload
window.addEventListener('beforeunload', function() {
    websocket.disconnect();
});

// Export for global access
window.MemVid = {
    utils,
    api,
    websocket,
    globalState,
    CONFIG
};
