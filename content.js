// Content Script - Injected into all pages to monitor AI agents
class AIAgentMonitor {
  constructor() {
    this.isActive = false;
    this.currentContext = null;
    this.messageHistory = [];
    this.lastUserInput = '';
    this.taskContext = {
      mainGoal: '',
      keywords: [],
      forbiddenTopics: [],
      contextAnchors: []
    };
    
    // Enhanced hybrid architecture support
    this.platform = null;
    this.sessionStartTime = Date.now();
    this.lastActivity = Date.now();
    this.authState = null;
    this.connectionStatus = {};
    this.pendingMessages = [];
    this.offlineMode = false;
    
    this.init();
  }

  init() {
    // Detect if this page contains an AI agent
    this.detectAIAgent();
    
    // Inject monitoring script into page context
    this.injectMonitoringScript();
    
    // Set up communication with background script
    this.setupCommunication();
    
    // Start monitoring if AI agent detected
    if (this.isActive) {
      this.startMonitoring();
    }
  }

  detectAIAgent() {
    // Detection patterns for common AI platforms
    const aiPlatforms = [
      { name: 'MiniMax', selector: '[data-testid="chat-input"], .chat-input, #chat-input' },
      { name: 'ChatGPT', selector: '#prompt-textarea, [data-id="root"] textarea' },
      { name: 'Claude', selector: 'textarea[placeholder*="message"], .claude-input' },
      { name: 'Gemini', selector: '.rich-textarea, [data-test-id="input-textarea"]' },
      { name: 'Generic', selector: 'textarea[placeholder*="chat"], textarea[placeholder*="message"], input[placeholder*="ask"]' }
    ];

    for (const platform of aiPlatforms) {
      const element = document.querySelector(platform.selector);
      if (element) {
        this.isActive = true;
        this.platform = platform.name;
        this.inputElement = element;
        console.log(`[AI Supervisor] Detected ${platform.name} agent`);
        break;
      }
    }

    // Continue checking for dynamically loaded content
    if (!this.isActive) {
      setTimeout(() => this.detectAIAgent(), 2000);
    }
  }

  injectMonitoringScript() {
    const script = document.createElement('script');
    script.src = chrome.runtime.getURL('injector.js');
    script.onload = function() {
      this.remove();
    };
    (document.head || document.documentElement).appendChild(script);
  }

  setupCommunication() {
    // Listen for messages from injected script
    window.addEventListener('message', (event) => {
      if (event.source !== window || !event.data.type) return;
      
      switch (event.data.type) {
        case 'AI_AGENT_MESSAGE':
          this.handleAgentMessage(event.data);
          break;
        case 'USER_INPUT':
          this.handleUserInput(event.data);
          break;
        case 'CONTEXT_UPDATE':
          this.updateContext(event.data);
          break;
        case 'PLATFORM_DETECTED':
          this.handlePlatformDetection(event.data);
          break;
      }
    });

    // Listen for messages from background script
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      switch (request.action) {
        case 'SET_TASK_CONTEXT':
          this.setTaskContext(request.data);
          sendResponse({ success: true });
          break;
        case 'INTERVENE':
          this.performIntervention(request.data);
          sendResponse({ success: true });
          break;
        case 'GET_STATUS':
          sendResponse(this.getMonitoringStatus());
          break;
        case 'UPDATE_AUTH_STATE':
          this.handleAuthStateUpdate(request.data);
          sendResponse({ success: true });
          break;
        case 'CONNECTION_STATUS_CHANGED':
          this.handleConnectionStatusChange(request.data);
          sendResponse({ success: true });
          break;
      }
      return true; // Keep message channel open
    });
  }

  startMonitoring() {
    console.log('[AI Supervisor] Starting monitoring...');
    
    // Monitor input changes
    if (this.inputElement) {
      this.inputElement.addEventListener('input', (e) => {
        this.lastUserInput = e.target.value;
        this.analyzeUserInput(e.target.value);
      });

      // Monitor form submissions
      const form = this.inputElement.closest('form') || this.inputElement.parentElement;
      if (form) {
        form.addEventListener('submit', (e) => {
          this.handleUserSubmission(this.lastUserInput);
        });
      }
    }

    // Monitor DOM changes for new messages
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'childList') {
          mutation.addedNodes.forEach((node) => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              this.checkForAgentMessage(node);
            }
          });
        }
      });
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true
    });
  }

  analyzeUserInput(input) {
    // Detect task context from user input
    const contextKeywords = this.extractContextKeywords(input);
    
    if (contextKeywords.length > 0) {
      this.updateTaskContext(contextKeywords, input);
    }

    // Send to background for analysis
    chrome.runtime.sendMessage({
      action: 'ANALYZE_USER_INPUT',
      data: {
        input,
        context: this.taskContext,
        timestamp: Date.now()
      }
    });
  }

  extractContextKeywords(input) {
    // Extract potential task-defining keywords
    const taskPatterns = [
      /(?:build|create|develop|make)\s+(?:a|an)?\s+([\w\s]+?)(?:\s+(?:app|application|system|tool|website|platform))/gi,
      /(?:working on|building|creating|developing)\s+([\w\s]+)/gi,
      /(?:project|task|goal)\s+(?:is|involves|includes)\s+([\w\s]+)/gi
    ];

    const keywords = [];
    taskPatterns.forEach(pattern => {
      const matches = input.match(pattern);
      if (matches) {
        keywords.push(...matches);
      }
    });

    return keywords;
  }

  updateTaskContext(keywords, fullInput) {
    // Update main goal if detected
    const goalMatch = fullInput.match(/(?:build|create|develop)\s+(?:a|an)?\s+([\w\s]+?)(?:\s+(?:app|application|system))/i);
    if (goalMatch && !this.taskContext.mainGoal) {
      this.taskContext.mainGoal = goalMatch[1].trim();
    }

    // Add context anchors
    this.taskContext.contextAnchors.push({
      text: fullInput,
      timestamp: Date.now(),
      keywords
    });

    // Detect contextual mentions that shouldn't change task (like "hackathon")
    const contextualMentions = fullInput.match(/\b(hackathon|competition|contest|deadline|event)\b/gi);
    if (contextualMentions) {
      this.taskContext.forbiddenTopics = [...this.taskContext.forbiddenTopics, ...contextualMentions];
    }
  }

  checkForAgentMessage(node) {
    // Check if this node contains an agent response
    if (this.isAgentMessage(node)) {
      const messageText = this.extractMessageText(node);
      this.handleAgentMessage({
        type: 'AI_AGENT_MESSAGE',
        content: messageText,
        element: node,
        timestamp: Date.now()
      });
    }
  }

  isAgentMessage(node) {
    // Platform-specific detection logic
    const agentIndicators = [
      '.message.assistant',
      '.ai-message',
      '.bot-message',
      '[data-role="assistant"]',
      '.response-message'
    ];

    return agentIndicators.some(selector => 
      node.matches && node.matches(selector) || 
      node.querySelector && node.querySelector(selector)
    );
  }

  extractMessageText(node) {
    // Extract clean text from message element
    const clone = node.cloneNode(true);
    
    // Remove code blocks, buttons, and other non-content elements
    const elementsToRemove = clone.querySelectorAll('button, .copy-button, script, style');
    elementsToRemove.forEach(el => el.remove());
    
    return clone.textContent?.trim() || '';
  }

  handleAgentMessage(data) {
    console.log('[AI Supervisor] Agent message detected:', data.content.substring(0, 100) + '...');
    
    // Enhanced metadata collection
    const messageData = {
      content: data.content,
      platform: this.platform,
      timestamp: data.timestamp,
      url: window.location.href,
      tabId: this.getTabId()
    };
    
    // Analyze for task drift
    const driftAnalysis = this.analyzeTaskDrift(data.content);
    
    // Send to background for processing with enhanced data
    chrome.runtime.sendMessage({
      action: 'AGENT_MESSAGE',
      data: {
        ...messageData,
        context: this.taskContext,
        driftAnalysis,
        sessionInfo: {
          startTime: this.sessionStartTime,
          messageCount: this.messageHistory.length
        }
      }
    });

    // Store in history with enhanced metadata
    this.messageHistory.push({
      type: 'agent',
      content: data.content,
      timestamp: data.timestamp,
      driftScore: driftAnalysis.score,
      platform: this.platform,
      url: window.location.href
    });
    
    // Update session state
    this.lastActivity = Date.now();
  }

  handleUserInput(data) {
    console.log('[AI Supervisor] User input detected:', data.content.substring(0, 50) + '...');
    
    // Enhanced input analysis with context
    this.analyzeUserInput(data.content);
    
    // Store in history
    this.messageHistory.push({
      type: 'user',
      content: data.content,
      timestamp: data.timestamp || Date.now(),
      platform: this.platform
    });
    
    this.lastActivity = Date.now();
  }

  handlePlatformDetection(data) {
    console.log('[AI Supervisor] Platform detection update:', data);
    
    // Update platform information
    if (data.platform && data.platform !== this.platform) {
      this.platform = data.platform;
      console.log(`[AI Supervisor] Platform updated to: ${this.platform}`);
    }
    
    // Update input element reference
    if (data.inputElement) {
      this.inputElement = data.inputElement;
    }
    
    // Notify background of platform change
    chrome.runtime.sendMessage({
      action: 'PLATFORM_UPDATED',
      data: {
        platform: this.platform,
        url: window.location.href,
        timestamp: Date.now()
      }
    });
  }

  handleAuthStateUpdate(data) {
    console.log('[AI Supervisor] Auth state updated:', data.isAuthenticated ? 'authenticated' : 'not authenticated');
    
    this.authState = data;
    
    // Update UI indicators if needed
    this.updateAuthIndicators();
  }

  handleConnectionStatusChange(data) {
    console.log('[AI Supervisor] Connection status changed:', data);
    
    this.connectionStatus = data;
    
    // Update UI to reflect connection status
    this.updateConnectionIndicators();
    
    // If we just connected, sync any pending data
    if (data.hasActiveConnections && this.pendingMessages.length > 0) {
      this.syncPendingMessages();
    }
  }

  updateAuthIndicators() {
    // Add visual indicators for authentication status
    if (this.authState && this.authState.isAuthenticated) {
      this.showTemporaryNotification('🔐 Authenticated with supervisor services', 'success');
    }
  }

  updateConnectionIndicators() {
    const hasConnections = this.connectionStatus && 
      Object.values(this.connectionStatus).some(conn => conn.connected);
    
    if (hasConnections) {
      this.showTemporaryNotification('🔗 Connected to supervisor services', 'success');
    } else {
      this.showTemporaryNotification('❌ Disconnected from supervisor services', 'warning');
    }
  }

  showTemporaryNotification(message, type = 'info') {
    // Create a temporary notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 10px;
      right: 10px;
      padding: 10px 15px;
      border-radius: 6px;
      font-size: 12px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      z-index: 999999;
      max-width: 300px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.15);
      animation: slideInFromRight 0.3s ease-out;
    `;
    
    // Set colors based on type
    const colors = {
      success: { bg: '#d4edda', text: '#155724', border: '#28a745' },
      warning: { bg: '#fff3cd', text: '#856404', border: '#ffc107' },
      error: { bg: '#f8d7da', text: '#721c24', border: '#dc3545' },
      info: { bg: '#e3f2fd', text: '#1976d2', border: '#2196f3' }
    };
    
    const color = colors[type] || colors.info;
    notification.style.backgroundColor = color.bg;
    notification.style.color = color.text;
    notification.style.border = `1px solid ${color.border}`;
    
    notification.textContent = message;
    
    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
      @keyframes slideInFromRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
      }
      @keyframes slideOutToRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
      }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(notification);
    
    // Auto remove after 3 seconds
    setTimeout(() => {
      notification.style.animation = 'slideOutToRight 0.3s ease-in';
      setTimeout(() => {
        if (notification.parentElement) {
          notification.remove();
        }
        if (style.parentElement) {
          style.remove();
        }
      }, 300);
    }, 3000);
  }

  getTabId() {
    // Try to get tab ID from chrome API if available in content script context
    // This is a fallback method since content scripts don't have direct access
    return window.location.href.hashCode ? window.location.href.hashCode() : Date.now();
  }

  analyzeTaskDrift(agentResponse) {
    let driftScore = 0;
    const issues = [];

    // Check if agent is switching away from main goal
    if (this.taskContext.mainGoal) {
      const mainGoalKeywords = this.taskContext.mainGoal.toLowerCase().split(' ');
      const responseWords = agentResponse.toLowerCase().split(' ');
      
      const goalMentions = mainGoalKeywords.filter(keyword => 
        responseWords.some(word => word.includes(keyword))
      ).length;
      
      if (goalMentions === 0 && agentResponse.length > 200) {
        driftScore += 0.5;
        issues.push('No mention of main goal');
      }
    }

    // Check for forbidden topic focus
    this.taskContext.forbiddenTopics.forEach(topic => {
      const topicRegex = new RegExp(`\\b${topic}\\b`, 'gi');
      const mentions = agentResponse.match(topicRegex);
      if (mentions && mentions.length > 2) {
        driftScore += 0.3;
        issues.push(`Excessive focus on contextual topic: ${topic}`);
      }
    });

    // Check for complete topic change
    const newTopicPatterns = [
      /let.?s?\s+(?:talk about|discuss|focus on|switch to)\s+([\w\s]+)/gi,
      /instead,?\s+(?:let.?s?|we could|how about)\s+([\w\s]+)/gi,
      /(?:different|another)\s+(?:topic|subject|project)/gi
    ];

    newTopicPatterns.forEach(pattern => {
      if (pattern.test(agentResponse)) {
        driftScore += 0.4;
        issues.push('Agent suggesting topic change');
      }
    });

    return {
      score: Math.min(driftScore, 1.0),
      issues,
      needsIntervention: driftScore > 0.6
    };
  }

  performIntervention(interventionData) {
    console.log('[AI Supervisor] Performing intervention:', interventionData);
    
    // Show user notification
    this.showInterventionNotification(interventionData.message);
    
    // If severe drift, prepare corrective prompt
    if (interventionData.type === 'SEVERE_DRIFT') {
      this.prepareCorrectionPrompt(interventionData);
    }
  }

  showInterventionNotification(message) {
    // Create floating notification
    const notification = document.createElement('div');
    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #ff6b35;
      color: white;
      padding: 15px 20px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3);
      z-index: 10000;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 14px;
      max-width: 300px;
      animation: slideIn 0.3s ease-out;
    `;
    
    notification.innerHTML = `
      <div style="font-weight: bold; margin-bottom: 5px;">🚨 AI Supervisor Alert</div>
      <div>${message}</div>
      <button onclick="this.parentElement.remove()" style="
        background: none; border: none; color: white; 
        position: absolute; top: 5px; right: 10px; 
        cursor: pointer; font-size: 16px;
      ">×</button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 10 seconds
    setTimeout(() => {
      if (notification.parentElement) {
        notification.remove();
      }
    }, 10000);
  }

  prepareCorrectionPrompt(interventionData) {
    const correctionPrompt = `Please refocus on the main task: ${this.taskContext.mainGoal}. ${interventionData.suggestion || 'Continue with the original project requirements.'}`;
    
    // Show correction suggestion to user
    if (this.inputElement) {
      const suggestion = document.createElement('div');
      suggestion.style.cssText = `
        position: absolute;
        background: #e3f2fd;
        border: 1px solid #2196f3;
        border-radius: 4px;
        padding: 10px;
        margin: 5px 0;
        font-size: 13px;
        color: #1976d2;
        z-index: 1000;
      `;
      
      suggestion.innerHTML = `
        <strong>💡 Suggested correction:</strong><br>
        "${correctionPrompt}"
        <button onclick="
          document.querySelector('${this.getInputSelector()}').value = '${correctionPrompt}';
          this.parentElement.remove();
        " style="
          margin-left: 10px; padding: 2px 8px; 
          background: #2196f3; color: white; 
          border: none; border-radius: 3px; cursor: pointer;
        ">Use This</button>
        <button onclick="this.parentElement.remove()" style="
          margin-left: 5px; padding: 2px 8px;
          background: none; border: 1px solid #2196f3; 
          color: #2196f3; border-radius: 3px; cursor: pointer;
        ">Dismiss</button>
      `;
      
      this.inputElement.parentElement.insertBefore(suggestion, this.inputElement.nextSibling);
    }
  }

  getInputSelector() {
    // Get CSS selector for the input element
    if (this.inputElement.id) return `#${this.inputElement.id}`;
    if (this.inputElement.className) return `.${this.inputElement.className.split(' ')[0]}`;
    return 'textarea, input[type="text"]';
  }

  getMonitoringStatus() {
    const hasActiveConnections = this.connectionStatus && 
      Object.values(this.connectionStatus).some(conn => conn.connected);
    
    return {
      isActive: this.isActive,
      platform: this.platform,
      taskContext: this.taskContext,
      messageCount: this.messageHistory.length,
      lastActivity: this.messageHistory.length > 0 ? 
        this.messageHistory[this.messageHistory.length - 1].timestamp : this.sessionStartTime,
      sessionDuration: Date.now() - this.sessionStartTime,
      connectionStatus: this.connectionStatus,
      authState: this.authState,
      offlineMode: this.offlineMode || !hasActiveConnections,
      pendingMessages: this.pendingMessages.length
    };
  }

  syncPendingMessages() {
    if (this.pendingMessages.length === 0) return;
    
    console.log(`[AI Supervisor] Syncing ${this.pendingMessages.length} pending messages`);
    
    // Send all pending messages to background
    this.pendingMessages.forEach(message => {
      chrome.runtime.sendMessage(message);
    });
    
    this.pendingMessages = [];
  }

  queueMessage(message) {
    // Add to pending messages if offline or no connections
    this.pendingMessages.push({
      ...message,
      queuedAt: Date.now()
    });
    
    // Limit queue size to prevent memory issues
    if (this.pendingMessages.length > 100) {
      this.pendingMessages = this.pendingMessages.slice(-50); // Keep only recent 50
    }
  }
}

// Initialize monitor when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => new AIAgentMonitor());
} else {
  new AIAgentMonitor();
}