// Injector Script - Runs in page context to monitor AI interactions
(function() {
  'use strict';
  
  console.log('[AI Supervisor] Injector script loaded');
  
  class PageContextMonitor {
    constructor() {
      this.originalFetch = window.fetch;
      this.originalXHROpen = XMLHttpRequest.prototype.open;
      this.originalXHRSend = XMLHttpRequest.prototype.send;
      this.originalWebSocket = window.WebSocket;
      
      this.activeRequests = new Map();
      this.messageBuffer = [];
      
      this.init();
    }
    
    init() {
      this.interceptNetworkRequests();
      this.interceptWebSocketConnections();
      this.monitorDOMChanges();
      this.interceptEventHandlers();
    }
    
    interceptNetworkRequests() {
      // Intercept fetch requests
      window.fetch = (...args) => {
        const [url, options] = args;
        
        // Check if this might be an AI API call
        if (this.isAIAPICall(url, options)) {
          console.log('[AI Supervisor] AI API call detected:', url);
          
          // Capture request data
          this.captureRequest(url, options);
          
          // Call original fetch and intercept response
          return this.originalFetch.apply(window, args)
            .then(response => {
              return this.interceptResponse(response, url, options);
            });
        }
        
        return this.originalFetch.apply(window, args);
      };
      
      // Intercept XMLHttpRequest
      const self = this;
      XMLHttpRequest.prototype.open = function(method, url, ...args) {
        this._supervisorUrl = url;
        this._supervisorMethod = method;
        return self.originalXHROpen.apply(this, [method, url, ...args]);
      };
      
      XMLHttpRequest.prototype.send = function(data) {
        if (self.isAIAPICall(this._supervisorUrl, { method: this._supervisorMethod, body: data })) {
          console.log('[AI Supervisor] AI XHR call detected:', this._supervisorUrl);
          
          this.addEventListener('load', function() {
            self.handleAIResponse(this.responseText, this._supervisorUrl);
          });
        }
        
        return self.originalXHRSend.apply(this, [data]);
      };
    }
    
    interceptWebSocketConnections() {
      const self = this;
      
      window.WebSocket = function(url, protocols) {
        const ws = new self.originalWebSocket(url, protocols);
        
        if (self.isAIWebSocket(url)) {
          console.log('[AI Supervisor] AI WebSocket detected:', url);
          
          // Intercept messages
          const originalOnMessage = ws.onmessage;
          ws.onmessage = function(event) {
            self.handleWebSocketMessage(event.data, url);
            if (originalOnMessage) originalOnMessage.call(this, event);
          };
          
          // Intercept send
          const originalSend = ws.send;
          ws.send = function(data) {
            self.handleWebSocketSend(data, url);
            return originalSend.call(this, data);
          };
        }
        
        return ws;
      };
      
      // Copy static properties
      Object.setPrototypeOf(window.WebSocket, self.originalWebSocket);
      window.WebSocket.prototype = self.originalWebSocket.prototype;
    }
    
    monitorDOMChanges() {
      const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          if (mutation.type === 'childList') {
            mutation.addedNodes.forEach((node) => {
              if (node.nodeType === Node.ELEMENT_NODE) {
                this.checkForAIMessages(node);
              }
            });
          }
        });
      });
      
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: false,
        characterData: true
      });
    }
    
    interceptEventHandlers() {
      // Monitor form submissions
      document.addEventListener('submit', (event) => {
        const form = event.target;
        if (this.isAIInputForm(form)) {
          const formData = new FormData(form);
          const textInputs = Array.from(form.elements)
            .filter(el => el.type === 'text' || el.type === 'textarea' || el.tagName.toLowerCase() === 'textarea')
            .map(el => el.value)
            .join(' ');
          
          if (textInputs.trim()) {
            this.notifyUserInput(textInputs);
          }
        }
      }, true);
      
      // Monitor enter key presses in text areas
      document.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
          const target = event.target;
          if (this.isAIInputElement(target) && target.value.trim()) {
            // Small delay to ensure any processing happens first
            setTimeout(() => {
              this.notifyUserInput(target.value);
            }, 100);
          }
        }
      }, true);
    }
    
    isAIAPICall(url, options) {
      if (!url) return false;
      
      const aiEndpoints = [
        '/chat/completions',
        '/completions',
        '/generate',
        '/conversation',
        '/api/conversation',
        '/backend-api/conversation',
        '/api/chat',
        '/chat',
        'openai.com',
        'anthropic.com',
        'api.minimax.com',
        'gemini',
        'claude'
      ];
      
      const urlStr = url.toString().toLowerCase();
      return aiEndpoints.some(endpoint => urlStr.includes(endpoint));
    }
    
    isAIWebSocket(url) {
      const aiWsPatterns = [
        'chat',
        'conversation',
        'stream',
        'websocket'
      ];
      
      const urlStr = url.toString().toLowerCase();
      return aiWsPatterns.some(pattern => urlStr.includes(pattern));
    }
    
    isAIInputForm(form) {
      // Check if form contains chat/conversation elements
      const indicators = [
        'chat-form',
        'message-form',
        'conversation-form',
        'prompt-form'
      ];
      
      const formClass = form.className.toLowerCase();
      const formId = form.id.toLowerCase();
      
      return indicators.some(indicator => 
        formClass.includes(indicator) || formId.includes(indicator)
      ) || form.querySelector('textarea[placeholder*="message"], textarea[placeholder*="chat"], input[placeholder*="ask"]');
    }
    
    isAIInputElement(element) {
      if (!element || !element.tagName) return false;
      
      const tagName = element.tagName.toLowerCase();
      if (tagName !== 'textarea' && tagName !== 'input') return false;
      
      const placeholder = (element.placeholder || '').toLowerCase();
      const className = (element.className || '').toLowerCase();
      const id = (element.id || '').toLowerCase();
      
      const aiIndicators = [
        'chat', 'message', 'prompt', 'ask', 'question', 
        'conversation', 'talk', 'input', 'query'
      ];
      
      return aiIndicators.some(indicator => 
        placeholder.includes(indicator) || 
        className.includes(indicator) || 
        id.includes(indicator)
      );
    }
    
    captureRequest(url, options) {
      try {
        const requestData = {
          url,
          method: options?.method || 'GET',
          timestamp: Date.now()
        };
        
        // Try to extract message from request body
        if (options?.body) {
          let body = options.body;
          if (typeof body === 'string') {
            try {
              const parsed = JSON.parse(body);
              if (parsed.messages || parsed.prompt || parsed.input) {
                requestData.userInput = parsed.messages?.[parsed.messages.length - 1]?.content || 
                                      parsed.prompt || parsed.input;
              }
            } catch (e) {
              // Body is not JSON, might be form data or plain text
              requestData.userInput = body;
            }
          }
        }
        
        this.notifyUserInput(requestData.userInput);
        
      } catch (error) {
        console.error('[AI Supervisor] Error capturing request:', error);
      }
    }
    
    async interceptResponse(response, url, options) {
      try {
        // Clone the response to avoid consuming the original
        const responseClone = response.clone();
        const text = await responseClone.text();
        
        this.handleAIResponse(text, url);
        
        return response;
      } catch (error) {
        console.error('[AI Supervisor] Error intercepting response:', error);
        return response;
      }
    }
    
    handleAIResponse(responseText, url) {
      try {
        // Try to parse as JSON
        let aiMessage = '';
        
        try {
          const parsed = JSON.parse(responseText);
          
          // Common AI API response structures
          if (parsed.choices && parsed.choices[0]?.message?.content) {
            aiMessage = parsed.choices[0].message.content;
          } else if (parsed.message && parsed.message.content) {
            aiMessage = parsed.message.content;
          } else if (parsed.content) {
            aiMessage = parsed.content;
          } else if (parsed.text) {
            aiMessage = parsed.text;
          } else if (parsed.response) {
            aiMessage = parsed.response;
          }
        } catch (e) {
          // Response is not JSON, might be plain text or SSE
          if (responseText.includes('data:')) {
            // Server-sent events format
            const lines = responseText.split('\n');
            for (const line of lines) {
              if (line.startsWith('data:')) {
                try {
                  const data = JSON.parse(line.slice(5));
                  if (data.choices && data.choices[0]?.delta?.content) {
                    aiMessage += data.choices[0].delta.content;
                  }
                } catch (e) {
                  // Ignore parsing errors for SSE
                }
              }
            }
          } else {
            aiMessage = responseText;
          }
        }
        
        if (aiMessage && aiMessage.trim().length > 10) {
          this.notifyAgentMessage(aiMessage.trim());
        }
        
      } catch (error) {
        console.error('[AI Supervisor] Error handling AI response:', error);
      }
    }
    
    handleWebSocketMessage(data, url) {
      try {
        let message = data;
        
        if (typeof data === 'string') {
          try {
            const parsed = JSON.parse(data);
            if (parsed.content || parsed.message || parsed.text || parsed.data) {
              message = parsed.content || parsed.message || parsed.text || parsed.data;
            }
          } catch (e) {
            // Data is not JSON, use as is
          }
        }
        
        if (typeof message === 'string' && message.trim().length > 10) {
          this.notifyAgentMessage(message.trim());
        }
        
      } catch (error) {
        console.error('[AI Supervisor] Error handling WebSocket message:', error);
      }
    }
    
    handleWebSocketSend(data, url) {
      try {
        let message = data;
        
        if (typeof data === 'string') {
          try {
            const parsed = JSON.parse(data);
            if (parsed.message || parsed.prompt || parsed.input || parsed.content) {
              message = parsed.message || parsed.prompt || parsed.input || parsed.content;
            }
          } catch (e) {
            // Data is not JSON, use as is
          }
        }
        
        if (typeof message === 'string' && message.trim().length > 5) {
          this.notifyUserInput(message.trim());
        }
        
      } catch (error) {
        console.error('[AI Supervisor] Error handling WebSocket send:', error);
      }
    }
    
    checkForAIMessages(node) {
      // Look for message-like elements in the added node
      const messageSelectors = [
        '.message',
        '.chat-message',
        '.ai-message',
        '.assistant-message',
        '.bot-message',
        '.response',
        '[role="assistant"]',
        '[data-role="assistant"]',
        '.gpt-message'
      ];
      
      for (const selector of messageSelectors) {
        let messageElement = null;
        
        if (node.matches && node.matches(selector)) {
          messageElement = node;
        } else if (node.querySelector) {
          messageElement = node.querySelector(selector);
        }
        
        if (messageElement) {
          const text = messageElement.textContent?.trim();
          if (text && text.length > 10) {
            this.notifyAgentMessage(text);
            break;
          }
        }
      }
    }
    
    notifyUserInput(input) {
      if (!input || typeof input !== 'string' || input.trim().length < 3) return;
      
      window.postMessage({
        type: 'USER_INPUT',
        content: input.trim(),
        timestamp: Date.now(),
        source: 'page-context'
      }, '*');
    }
    
    notifyAgentMessage(content) {
      if (!content || typeof content !== 'string' || content.trim().length < 10) return;
      
      // Debounce rapid messages
      const now = Date.now();
      if (this.lastMessageTime && now - this.lastMessageTime < 500) {
        return;
      }
      this.lastMessageTime = now;
      
      window.postMessage({
        type: 'AI_AGENT_MESSAGE',
        content: content.trim(),
        timestamp: now,
        source: 'page-context'
      }, '*');
    }
  }
  
  // Initialize the monitor
  const monitor = new PageContextMonitor();
  
  // Expose some functionality for debugging
  window.__aiSupervisorMonitor = monitor;
  
})();