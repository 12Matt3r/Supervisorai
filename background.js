// Enhanced Background Service Worker - Hybrid Architecture Integration
// Supports both local supervisor server and cloud-based services

class HybridSupervisorService {
  constructor() {
    // Connection endpoints
    this.endpoints = {
      local: {
        url: 'ws://localhost:8765',
        priority: 1,
        type: 'local'
      },
      cloud: {
        url: null, // Will be set from settings or auto-detected
        priority: 2,
        type: 'cloud'
      }
    };
    
    // Connection state
    this.activeConnections = new Map();
    this.connectionAttempts = new Map();
    this.maxRetries = 3;
    this.isOnline = navigator.onLine;
    
    // Data management
    this.messageQueue = [];
    this.offlineData = new Map();
    this.syncQueue = new Set();
    this.lastSyncTime = 0;
    
    // Session management
    this.activeTabSessions = new Map();
    this.taskContexts = new Map();
    this.authState = {
      isAuthenticated: false,
      userId: null,
      sessionToken: null,
      refreshToken: null
    };
    
    this.init();
  }

  async init() {
    console.log('[Hybrid Supervisor] Initializing hybrid architecture service...');
    
    // Load configuration and auth state
    await this.loadConfiguration();
    await this.loadAuthState();
    
    // Set up event listeners
    this.setupEventListeners();
    
    // Initialize connections
    await this.initializeConnections();
    
    // Set up periodic tasks
    this.setupPeriodicTasks();
    
    // Restore offline data if reconnecting
    await this.handleReconnection();
  }

  async loadConfiguration() {
    try {
      const config = await chrome.storage.sync.get([
        'supervisorServerUrl',
        'cloudEndpointUrl',
        'deploymentMode',
        'autoDetectEndpoints',
        'enableOfflineMode',
        'syncSettings'
      ]);
      
      // Update endpoints from configuration
      if (config.supervisorServerUrl) {
        this.endpoints.local.url = config.supervisorServerUrl;
      }
      
      if (config.cloudEndpointUrl) {
        this.endpoints.cloud.url = config.cloudEndpointUrl;
      } else {
        // Auto-detect cloud endpoint from current domain
        await this.autoDetectCloudEndpoint();
      }
      
      this.deploymentMode = config.deploymentMode || 'auto';
      this.autoDetectEndpoints = config.autoDetectEndpoints !== false;
      this.enableOfflineMode = config.enableOfflineMode !== false;
      
      console.log('[Hybrid Supervisor] Configuration loaded:', {
        deploymentMode: this.deploymentMode,
        localUrl: this.endpoints.local.url,
        cloudUrl: this.endpoints.cloud.url
      });
      
    } catch (error) {
      console.error('[Hybrid Supervisor] Failed to load configuration:', error);
    }
  }

  async autoDetectCloudEndpoint() {
    try {
      // Check if we're on a known platform
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tabs.length > 0) {
        const url = new URL(tabs[0].url);
        
        // Check for known cloud platforms
        if (url.hostname.includes('minimax.com') || url.hostname.includes('minimaxi.cn') || 
            url.hostname.includes('space.minimax.io')) {
          // Use our hybrid architecture gateway
          this.endpoints.cloud.url = 'ws://localhost:8888/ws';
          // For production: this.endpoints.cloud.url = 'wss://your-hybrid-gateway.space.minimax.io/ws';
        }
      }
    } catch (error) {
      console.log('[Hybrid Supervisor] Auto-detection failed, using manual configuration');
    }
  }

  async loadAuthState() {
    try {
      const authData = await chrome.storage.local.get(['authState', 'userId', 'sessionToken']);
      if (authData.authState) {
        this.authState = { ...this.authState, ...authData.authState };
      }
    } catch (error) {
      console.error('[Hybrid Supervisor] Failed to load auth state:', error);
    }
  }

  setupEventListeners() {
    // Handle messages from content scripts
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      this.handleContentScriptMessage(request, sender, sendResponse);
      return true; // Keep message channel open for async response
    });

    // Handle tab updates
    chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
      if (changeInfo.status === 'complete' && tab.url) {
        this.handleTabUpdate(tabId, tab);
      }
    });

    // Handle tab removal
    chrome.tabs.onRemoved.addListener((tabId) => {
      this.activeTabSessions.delete(tabId);
      this.taskContexts.delete(tabId);
      this.cleanupTabData(tabId);
    });

    // Handle network status changes
    window.addEventListener('online', () => {
      this.isOnline = true;
      this.handleNetworkReconnection();
    });

    window.addEventListener('offline', () => {
      this.isOnline = false;
      this.handleNetworkDisconnection();
    });

    // Handle extension lifecycle
    chrome.runtime.onStartup.addListener(() => {
      this.handleStartup();
    });

    chrome.runtime.onInstalled.addListener((details) => {
      this.handleInstallation(details);
    });

    // Handle alarm for periodic tasks
    chrome.alarms.onAlarm.addListener((alarm) => {
      this.handleAlarm(alarm);
    });
  }

  async initializeConnections() {
    console.log('[Hybrid Supervisor] Initializing connections...');
    
    const connectionPromises = [];
    
    // Determine connection strategy based on deployment mode
    switch (this.deploymentMode) {
      case 'local':
        connectionPromises.push(this.connectToEndpoint('local'));
        break;
      case 'cloud':
        connectionPromises.push(this.connectToEndpoint('cloud'));
        break;
      case 'hybrid':
        connectionPromises.push(this.connectToEndpoint('local'));
        connectionPromises.push(this.connectToEndpoint('cloud'));
        break;
      case 'auto':
      default:
        // Try local first, fallback to cloud
        try {
          await this.connectToEndpoint('local');
        } catch (error) {
          console.log('[Hybrid Supervisor] Local connection failed, trying cloud...');
          connectionPromises.push(this.connectToEndpoint('cloud'));
        }
        break;
    }
    
    // Wait for at least one connection to succeed
    if (connectionPromises.length > 0) {
      await Promise.allSettled(connectionPromises);
    }
    
    this.logConnectionStatus();
  }

  async connectToEndpoint(endpointType) {
    const endpoint = this.endpoints[endpointType];
    if (!endpoint.url) {
      throw new Error(`No URL configured for ${endpointType} endpoint`);
    }
    
    const connectionId = `${endpointType}_${Date.now()}`;
    console.log(`[Hybrid Supervisor] Connecting to ${endpointType} endpoint: ${endpoint.url}`);
    
    try {
      // Build WebSocket URL with authentication for hybrid gateway
      const wsUrl = this.buildWebSocketUrl(endpoint.url, endpointType);
      const websocket = new WebSocket(wsUrl);
      
      const connection = {
        id: connectionId,
        type: endpointType,
        websocket: websocket,
        url: endpoint.url,
        isConnected: false,
        lastActivity: Date.now(),
        reconnectAttempts: 0
      };
      
      // Set up WebSocket event handlers
      websocket.onopen = () => {
        console.log(`[Hybrid Supervisor] Connected to ${endpointType} endpoint`);
        connection.isConnected = true;
        connection.reconnectAttempts = 0;
        this.activeConnections.set(connectionId, connection);
        this.handleConnectionOpen(connectionId, endpointType);
      };
      
      websocket.onmessage = (event) => {
        connection.lastActivity = Date.now();
        this.handleMessage(connectionId, event.data, endpointType);
      };
      
      websocket.onerror = (error) => {
        console.error(`[Hybrid Supervisor] WebSocket error for ${endpointType}:`, error);
        this.handleConnectionError(connectionId, error);
      };
      
      websocket.onclose = (event) => {
        console.log(`[Hybrid Supervisor] Connection closed for ${endpointType}:`, event.reason);
        connection.isConnected = false;
        this.handleConnectionClose(connectionId, event);
      };
      
      return connection;
      
    } catch (error) {
      console.error(`[Hybrid Supervisor] Failed to connect to ${endpointType}:`, error);
      throw error;
    }
  }
  
  buildWebSocketUrl(baseUrl, endpointType) {
    const url = new URL(baseUrl);
    
    // Add authentication and deployment mode parameters for hybrid gateway
    if (endpointType === 'cloud') {
      url.searchParams.set('user_id', this.authState.userId || 'extension_user');
      url.searchParams.set('deployment_mode', 'extension');
      url.searchParams.set('auth_token', this.authState.sessionToken || 'temp_token');
    }
    
    return url.toString();
  }y: Date.now(),
        messageCount: 0,
        reconnectAttempts: 0
      };
      
      // Set up WebSocket event handlers
      await this.setupWebSocketHandlers(connection);
      
      this.activeConnections.set(connectionId, connection);
      this.connectionAttempts.set(endpointType, 0);
      
      return connection;
      
    } catch (error) {
      console.error(`[Hybrid Supervisor] Failed to connect to ${endpointType}:`, error);
      this.handleConnectionFailure(endpointType, error);
      throw error;
    }
  }

  buildWebSocketUrl(baseUrl) {
    const url = new URL(baseUrl);
    
    // Add authentication parameters if available
    if (this.authState.userId) {
      url.searchParams.set('user_id', this.authState.userId);
    }
    
    if (this.authState.sessionToken) {
      url.searchParams.set('session_token', this.authState.sessionToken);
    }
    
    // Add extension metadata
    url.searchParams.set('extension_id', chrome.runtime.id);
    url.searchParams.set('extension_version', chrome.runtime.getManifest().version);
    url.searchParams.set('timestamp', Date.now().toString());
    
    return url.toString();
  }

  async setupWebSocketHandlers(connection) {
    return new Promise((resolve, reject) => {
      const ws = connection.websocket;
      
      ws.onopen = () => {
        console.log(`[Hybrid Supervisor] Connected to ${connection.type} endpoint`);
        connection.isConnected = true;
        connection.lastActivity = Date.now();
        
        // Send registration message
        this.sendRegistrationMessage(connection);
        
        // Process queued messages for this connection type
        this.processQueuedMessages(connection.type);
        
        resolve(connection);
      };

      ws.onmessage = (event) => {
        this.handleWebSocketMessage(event, connection);
        connection.lastActivity = Date.now();
        connection.messageCount++;
      };

      ws.onclose = (event) => {
        console.log(`[Hybrid Supervisor] ${connection.type} connection closed:`, event.code, event.reason);
        connection.isConnected = false;
        this.handleConnectionClose(connection, event);
      };

      ws.onerror = (error) => {
        console.error(`[Hybrid Supervisor] ${connection.type} WebSocket error:`, error);
        connection.isConnected = false;
        reject(error);
      };
      
      // Set timeout for connection
      setTimeout(() => {
        if (!connection.isConnected) {
          reject(new Error('Connection timeout'));
        }
      }, 10000);
    });
  }

  async sendRegistrationMessage(connection) {
    const registrationData = {
      type: 'EXTENSION_REGISTER',
      id: this.generateMessageId(),
      data: {
        extensionId: chrome.runtime.id,
        version: chrome.runtime.getManifest().version,
        userId: this.authState.userId,
        capabilities: [
          'real-time-monitoring',
          'task-coherence-protection', 
          'intervention',
          'offline-sync',
          'multi-session'
        ],
        deployment: connection.type,
        timestamp: Date.now()
      }
    };
    
    await this.sendMessage(connection, registrationData);
  }

  async handleWebSocketMessage(event, connection) {
    try {
      const message = JSON.parse(event.data);
      
      console.log(`[Hybrid Supervisor] Message from ${connection.type}:`, message.type);
      
      switch (message.type) {
        case 'connection_established':
          await this.handleConnectionEstablished(message, connection);
          break;
          
        case 'intervention_required':
          await this.handleInterventionRequired(message);
          break;
          
        case 'sync_request':
          await this.handleSyncRequest(message, connection);
          break;
          
        case 'auth_token_refresh':
          await this.handleAuthTokenRefresh(message);
          break;
          
        case 'configuration_update':
          await this.handleConfigurationUpdate(message);
          break;
          
        case 'pong':
          this.handlePong(connection);
          break;
          
        default:
          console.log(`[Hybrid Supervisor] Unknown message type: ${message.type}`);
      }
      
    } catch (error) {
      console.error('[Hybrid Supervisor] Error processing WebSocket message:', error);
    }
  }

  async handleConnectionEstablished(message, connection) {
    const { session_id, server_info, capabilities } = message.data;
    
    connection.sessionId = session_id;
    connection.serverCapabilities = capabilities;
    
    console.log(`[Hybrid Supervisor] ${connection.type} connection established:`, {
      sessionId: session_id,
      capabilities: capabilities
    });
    
    // Update UI
    this.broadcastStatusUpdate();
    
    // Sync offline data if available
    if (this.offlineData.size > 0) {
      await this.syncOfflineData(connection);
    }
  }

  async handleInterventionRequired(message) {
    const { task_id, intervention_type, reason, suggested_action, severity } = message.data;
    
    console.log('[Hybrid Supervisor] Intervention required:', {
      taskId: task_id,
      type: intervention_type,
      severity: severity
    });
    
    // Find the relevant tab
    const tabId = await this.findTabByTaskId(task_id);
    if (tabId) {
      // Send intervention to content script
      await chrome.tabs.sendMessage(tabId, {
        action: 'INTERVENE',
        data: {
          type: intervention_type,
          reason: reason,
          suggestedAction: suggested_action,
          severity: severity
        }
      });
      
      // Log intervention
      await this.logIntervention({
        tabId,
        taskId: task_id,
        type: intervention_type,
        reason,
        timestamp: Date.now()
      });
    }
  }

  async handleSyncRequest(message, connection) {
    const { sync_type, last_sync_timestamp } = message.data;
    
    console.log(`[Hybrid Supervisor] Sync request from ${connection.type}:`, sync_type);
    
    switch (sync_type) {
      case 'activity_data':
        await this.syncActivityData(connection, last_sync_timestamp);
        break;
      case 'task_contexts':
        await this.syncTaskContexts(connection);
        break;
      case 'configuration':
        await this.syncConfiguration(connection);
        break;
    }
  }

  async handleContentScriptMessage(request, sender, sendResponse) {
    const tabId = sender.tab?.id;
    
    try {
      switch (request.action) {
        case 'ANALYZE_USER_INPUT':
          await this.analyzeUserInput(request.data, tabId);
          sendResponse({ success: true });
          break;
          
        case 'AGENT_MESSAGE':
          await this.processAgentMessage(request.data, tabId);
          sendResponse({ success: true });
          break;
          
        case 'GET_STATUS':
          const status = await this.getMonitoringStatus(tabId);
          sendResponse(status);
          break;
          
        case 'SET_TASK_CONTEXT':
          await this.setTaskContext(request.data, tabId);
          sendResponse({ success: true });
          break;
          
        case 'REQUEST_AUTH':
          const authResult = await this.requestAuthentication();
          sendResponse(authResult);
          break;
          
        default:
          sendResponse({ success: false, error: 'Unknown action' });
      }
    } catch (error) {
      console.error('[Hybrid Supervisor] Error handling message:', error);
      sendResponse({ success: false, error: error.message });
    }
  }

  async analyzeUserInput(data, tabId) {
    console.log('[Hybrid Supervisor] Analyzing user input for tab:', tabId);
    
    // Update task context
    await this.updateTaskContext(data, tabId);
    
    // Create analysis message
    const analysisMessage = {
      type: 'USER_INPUT_ANALYSIS',
      id: this.generateMessageId(),
      data: {
        tabId,
        input: data.input,
        context: this.taskContexts.get(tabId),
        timestamp: Date.now(),
        url: await this.getTabUrl(tabId),
        platform: data.platform
      }
    };
    
    // Send to all connected endpoints or queue if offline
    await this.broadcastOrQueue(analysisMessage);
    
    // Store in offline cache
    if (this.enableOfflineMode) {
      await this.cacheActivity('user_input', analysisMessage.data, tabId);
    }
  }

  async processAgentMessage(data, tabId) {
    console.log('[Hybrid Supervisor] Processing agent message for tab:', tabId);
    
    const context = this.taskContexts.get(tabId);
    if (!context) return;
    
    // Enhanced task drift analysis with hybrid intelligence
    const driftAnalysis = await this.performAdvancedDriftAnalysis(data.content, context, tabId);
    
    // Create message for supervisor
    const messageData = {
      type: 'AGENT_MESSAGE_ANALYSIS',
      id: this.generateMessageId(),
      data: {
        tabId,
        content: data.content,
        platform: data.platform,
        context: context,
        driftAnalysis,
        timestamp: Date.now(),
        url: await this.getTabUrl(tabId)
      }
    };
    
    // Send to connected endpoints
    await this.broadcastOrQueue(messageData);
    
    // Log interaction
    await this.logInteraction({
      tabId,
      type: 'agent_response',
      content: data.content,
      driftScore: driftAnalysis.score,
      issues: driftAnalysis.issues,
      timestamp: Date.now()
    });
    
    // Cache for offline mode
    if (this.enableOfflineMode) {
      await this.cacheActivity('agent_message', messageData.data, tabId);
    }
    
    // Trigger intervention if needed
    if (driftAnalysis.needsIntervention) {
      await this.triggerIntervention(driftAnalysis, tabId);
    }
  }

  async performAdvancedDriftAnalysis(agentResponse, context, tabId) {
    // Local analysis first
    const localAnalysis = this.performLocalDriftAnalysis(agentResponse, context);
    
    // If we have cloud connection, get enhanced analysis
    const cloudConnection = this.getConnectedEndpoint('cloud');
    if (cloudConnection && localAnalysis.score > 0.3) {
      try {
        const enhancedAnalysis = await this.requestCloudAnalysis({
          content: agentResponse,
          context: context,
          localAnalysis: localAnalysis,
          tabId: tabId
        });
        
        // Combine local and cloud analysis
        return this.combineAnalysisResults(localAnalysis, enhancedAnalysis);
      } catch (error) {
        console.log('[Hybrid Supervisor] Cloud analysis failed, using local only:', error);
      }
    }
    
    return localAnalysis;
  }

  performLocalDriftAnalysis(agentResponse, context) {
    let driftScore = 0;
    const issues = [];
    const suggestions = [];

    // Main goal abandonment check
    if (context.mainGoal) {
      const mainGoalWords = context.mainGoal.toLowerCase().split(/\s+/);
      const responseWords = agentResponse.toLowerCase().split(/\s+/);
      
      const goalMentions = mainGoalWords.filter(word => 
        responseWords.some(respWord => 
          respWord.includes(word) || word.includes(respWord)
        )
      ).length;
      
      const goalRelevance = goalMentions / mainGoalWords.length;
      
      if (goalRelevance < 0.2 && agentResponse.length > 150) {
        driftScore += 0.6;
        issues.push(`No mention of main goal: "${context.mainGoal}"`);
        suggestions.push(`Please focus on ${context.mainGoal} as discussed`);
      } else if (goalRelevance < 0.4) {
        driftScore += 0.3;
        issues.push('Weak connection to main goal');
      }
    }

    // Topic switching detection
    const topicSwitchPatterns = [
      /let.?s?\s+(?:talk about|discuss|focus on|switch to|try)\s+([^.!?]+)/gi,
      /instead,?\s+(?:let.?s?|we could|how about|why don.?t we)\s+([^.!?]+)/gi,
      /(?:different|another|alternative)\s+(?:topic|subject|project|idea|approach)/gi
    ];

    topicSwitchPatterns.forEach(pattern => {
      const matches = agentResponse.match(pattern);
      if (matches) {
        driftScore += 0.5;
        issues.push('Agent suggesting topic change: ' + matches[0]);
        suggestions.push('Please stay focused on the original task');
      }
    });

    // Contextual keyword hijacking
    const contextualKeywords = ['hackathon', 'competition', 'contest', 'deadline', 'event'];
    contextualKeywords.forEach(keyword => {
      const keywordRegex = new RegExp(`\\b${keyword}\\b`, 'gi');
      const mentions = agentResponse.match(keywordRegex);
      if (mentions && mentions.length > 2) {
        driftScore += 0.4;
        issues.push(`Excessive focus on contextual topic: ${keyword}`);
      }
    });

    return {
      score: Math.min(driftScore, 1.0),
      issues,
      suggestions,
      needsIntervention: driftScore > 0.6,
      confidence: 0.7, // Local analysis confidence
      analysisType: 'local'
    };
  }

  async requestCloudAnalysis(data) {
    const cloudConnection = this.getConnectedEndpoint('cloud');
    if (!cloudConnection) {
      throw new Error('No cloud connection available');
    }
    
    const requestMessage = {
      type: 'REQUEST_ANALYSIS',
      id: this.generateMessageId(),
      data: data
    };
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('Cloud analysis timeout'));
      }, 10000);
      
      // Store the resolver for the response
      this.pendingAnalysisRequests = this.pendingAnalysisRequests || new Map();
      this.pendingAnalysisRequests.set(requestMessage.id, { resolve, reject, timeout });
      
      this.sendMessage(cloudConnection, requestMessage);
    });
  }

  combineAnalysisResults(localAnalysis, cloudAnalysis) {
    return {
      score: Math.max(localAnalysis.score, cloudAnalysis.score),
      issues: [...new Set([...localAnalysis.issues, ...cloudAnalysis.issues])],
      suggestions: [...new Set([...localAnalysis.suggestions, ...cloudAnalysis.suggestions])],
      needsIntervention: localAnalysis.needsIntervention || cloudAnalysis.needsIntervention,
      confidence: Math.max(localAnalysis.confidence, cloudAnalysis.confidence),
      analysisType: 'hybrid',
      localScore: localAnalysis.score,
      cloudScore: cloudAnalysis.score
    };
  }

  async broadcastOrQueue(message) {
    const connectedEndpoints = Array.from(this.activeConnections.values())
      .filter(conn => conn.isConnected);
    
    if (connectedEndpoints.length > 0) {
      // Send to all connected endpoints
      const promises = connectedEndpoints.map(conn => this.sendMessage(conn, message));
      await Promise.allSettled(promises);
    } else {
      // Queue for later delivery
      this.messageQueue.push({
        message,
        timestamp: Date.now(),
        attempts: 0
      });
      console.log('[Hybrid Supervisor] Message queued for later delivery');
    }
  }

  async sendMessage(connection, message) {
    if (!connection.isConnected) {
      throw new Error(`Connection ${connection.type} is not active`);
    }
    
    try {
      const messageStr = JSON.stringify(message);
      connection.websocket.send(messageStr);
      connection.lastActivity = Date.now();
      return true;
    } catch (error) {
      console.error(`[Hybrid Supervisor] Failed to send message to ${connection.type}:`, error);
      connection.isConnected = false;
      throw error;
    }
  }

  getConnectedEndpoint(type) {
    return Array.from(this.activeConnections.values())
      .find(conn => conn.type === type && conn.isConnected);
  }

  async processQueuedMessages(connectionType) {
    const connection = this.getConnectedEndpoint(connectionType);
    if (!connection || this.messageQueue.length === 0) return;
    
    console.log(`[Hybrid Supervisor] Processing ${this.messageQueue.length} queued messages`);
    
    const messagesToSend = this.messageQueue.splice(0);
    for (const queuedMsg of messagesToSend) {
      try {
        await this.sendMessage(connection, queuedMsg.message);
        console.log('[Hybrid Supervisor] Queued message sent successfully');
      } catch (error) {
        // Re-queue failed messages with attempt limit
        queuedMsg.attempts++;
        if (queuedMsg.attempts < 3) {
          this.messageQueue.push(queuedMsg);
        }
      }
    }
  }

  // Offline Support Methods
  async cacheActivity(type, data, tabId) {
    const cacheKey = `${tabId}_${Date.now()}`;
    const cacheEntry = {
      type,
      data,
      tabId,
      timestamp: Date.now(),
      synced: false
    };
    
    this.offlineData.set(cacheKey, cacheEntry);
    
    // Persist to local storage
    const existingCache = await chrome.storage.local.get(['offlineActivityCache']) || {};
    existingCache.offlineActivityCache = existingCache.offlineActivityCache || {};
    existingCache.offlineActivityCache[cacheKey] = cacheEntry;
    
    await chrome.storage.local.set(existingCache);
  }

  async syncOfflineData(connection) {
    if (this.offlineData.size === 0) return;
    
    console.log(`[Hybrid Supervisor] Syncing ${this.offlineData.size} offline activities`);
    
    const syncMessage = {
      type: 'OFFLINE_DATA_SYNC',
      id: this.generateMessageId(),
      data: {
        activities: Array.from(this.offlineData.values()).filter(entry => !entry.synced),
        lastSyncTime: this.lastSyncTime
      }
    };
    
    try {
      await this.sendMessage(connection, syncMessage);
      
      // Mark data as synced
      this.offlineData.forEach(entry => entry.synced = true);
      this.lastSyncTime = Date.now();
      
      // Clear old synced data
      setTimeout(() => this.cleanupSyncedData(), 60000);
      
    } catch (error) {
      console.error('[Hybrid Supervisor] Failed to sync offline data:', error);
    }
  }

  async cleanupSyncedData() {
    const now = Date.now();
    const maxAge = 24 * 60 * 60 * 1000; // 24 hours
    
    this.offlineData.forEach((entry, key) => {
      if (entry.synced && (now - entry.timestamp) > maxAge) {
        this.offlineData.delete(key);
      }
    });
    
    // Update local storage
    const cacheData = {};
    this.offlineData.forEach((entry, key) => {
      cacheData[key] = entry;
    });
    
    await chrome.storage.local.set({ offlineActivityCache: cacheData });
  }

  // Authentication Integration
  async requestAuthentication() {
    try {
      // Try to get auth from web app session first
      const webAppAuth = await this.getWebAppAuthentication();
      if (webAppAuth) {
        this.authState = { ...this.authState, ...webAppAuth };
        await this.saveAuthState();
        return { success: true, method: 'webapp_session' };
      }
      
      // Fallback to extension-based auth
      const extensionAuth = await this.performExtensionAuth();
      if (extensionAuth) {
        this.authState = { ...this.authState, ...extensionAuth };
        await this.saveAuthState();
        return { success: true, method: 'extension_auth' };
      }
      
      return { success: false, error: 'Authentication failed' };
      
    } catch (error) {
      console.error('[Hybrid Supervisor] Authentication error:', error);
      return { success: false, error: error.message };
    }
  }

  async getWebAppAuthentication() {
    try {
      // Try to extract auth from cookies or local storage of current tab
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tabs.length === 0) return null;
      
      const currentUrl = new URL(tabs[0].url);
      
      // Get cookies for the current domain
      const cookies = await chrome.cookies.getAll({ domain: currentUrl.hostname });
      const authCookie = cookies.find(cookie => 
        cookie.name.includes('auth') || 
        cookie.name.includes('session') || 
        cookie.name.includes('token')
      );
      
      if (authCookie) {
        return {
          sessionToken: authCookie.value,
          userId: null, // Will be extracted from token validation
          source: 'webapp_cookie'
        };
      }
      
      return null;
    } catch (error) {
      console.log('[Hybrid Supervisor] Web app auth extraction failed:', error);
      return null;
    }
  }

  async performExtensionAuth() {
    // Generate extension-specific auth
    const extensionId = chrome.runtime.id;
    const timestamp = Date.now();
    const nonce = this.generateNonce();
    
    return {
      extensionId: extensionId,
      timestamp: timestamp,
      nonce: nonce,
      userId: `ext_${extensionId}_${timestamp}`,
      sessionToken: `ext_token_${nonce}`,
      source: 'extension_generated'
    };
  }

  async saveAuthState() {
    await chrome.storage.local.set({ 
      authState: this.authState,
      lastAuthTime: Date.now()
    });
  }

  // Utility Methods
  generateMessageId() {
    return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  generateNonce() {
    return Math.random().toString(36).substr(2, 15);
  }

  async getTabUrl(tabId) {
    try {
      const tab = await chrome.tabs.get(tabId);
      return tab.url;
    } catch (error) {
      return null;
    }
  }

  logConnectionStatus() {
    const activeCount = Array.from(this.activeConnections.values())
      .filter(conn => conn.isConnected).length;
    
    console.log(`[Hybrid Supervisor] Connection status: ${activeCount}/${this.activeConnections.size} active`);
    
    this.activeConnections.forEach((conn, id) => {
      console.log(`  ${conn.type}: ${conn.isConnected ? '✓' : '✗'} (${conn.url})`);
    });
  }

  broadcastStatusUpdate() {
    // Notify popup and content scripts of status change
    chrome.runtime.sendMessage({
      action: 'STATUS_UPDATE',
      data: {
        connectionStatus: this.getConnectionStatus(),
        authStatus: this.authState,
        queueSize: this.messageQueue.length,
        offlineDataSize: this.offlineData.size
      }
    }).catch(() => {
      // Ignore errors if popup is not open
    });
  }

  getConnectionStatus() {
    const status = {};
    this.activeConnections.forEach((conn, id) => {
      status[conn.type] = {
        connected: conn.isConnected,
        url: conn.url,
        messageCount: conn.messageCount,
        lastActivity: conn.lastActivity
      };
    });
    return status;
  }

  // Periodic Tasks
  setupPeriodicTasks() {
    // Health check every 30 seconds
    chrome.alarms.create('healthCheck', { periodInMinutes: 0.5 });
    
    // Sync offline data every 5 minutes
    chrome.alarms.create('offlineSync', { periodInMinutes: 5 });
    
    // Cleanup old data every hour
    chrome.alarms.create('dataCleanup', { periodInMinutes: 60 });
    
    // Connection retry every minute
    chrome.alarms.create('connectionRetry', { periodInMinutes: 1 });
  }

  async handleAlarm(alarm) {
    switch (alarm.name) {
      case 'healthCheck':
        await this.performHealthCheck();
        break;
      case 'offlineSync':
        await this.performOfflineSync();
        break;
      case 'dataCleanup':
        await this.cleanupOldData();
        break;
      case 'connectionRetry':
        await this.retryFailedConnections();
        break;
    }
  }

  async performHealthCheck() {
    // Send ping to all connections
    const connections = Array.from(this.activeConnections.values())
      .filter(conn => conn.isConnected);
    
    for (const connection of connections) {
      try {
        await this.sendMessage(connection, {
          type: 'PING',
          id: this.generateMessageId(),
          data: { timestamp: Date.now() }
        });
      } catch (error) {
        console.log(`[Hybrid Supervisor] Health check failed for ${connection.type}:`, error);
        connection.isConnected = false;
      }
    }
  }

  async retryFailedConnections() {
    const failedEndpoints = Object.keys(this.endpoints).filter(type => {
      const connection = this.getConnectedEndpoint(type);
      return !connection || !connection.isConnected;
    });
    
    for (const endpointType of failedEndpoints) {
      const attemptCount = this.connectionAttempts.get(endpointType) || 0;
      if (attemptCount < this.maxRetries) {
        console.log(`[Hybrid Supervisor] Retrying ${endpointType} connection (attempt ${attemptCount + 1})`);
        try {
          await this.connectToEndpoint(endpointType);
        } catch (error) {
          this.connectionAttempts.set(endpointType, attemptCount + 1);
        }
      }
    }
  }

  // Event Handlers
  async handleStartup() {
    console.log('[Hybrid Supervisor] Extension startup - reinitializing...');
    await this.init();
  }

  async handleInstallation(details) {
    if (details.reason === 'install') {
      console.log('[Hybrid Supervisor] Extension installed - setting up defaults...');
      await this.setDefaultConfiguration();
    }
  }

  async setDefaultConfiguration() {
    const defaultConfig = {
      deploymentMode: 'auto',
      autoDetectEndpoints: true,
      enableOfflineMode: true,
      interventionThreshold: 0.5,
      enableNotifications: true,
      syncSettings: {
        syncInterval: 5, // minutes
        maxOfflineData: 1000 // entries
      }
    };
    
    await chrome.storage.sync.set(defaultConfig);
  }

  handleNetworkReconnection() {
    console.log('[Hybrid Supervisor] Network reconnected - attempting to restore connections...');
    this.retryFailedConnections();
  }

  handleNetworkDisconnection() {
    console.log('[Hybrid Supervisor] Network disconnected - entering offline mode...');
    // Connections will be marked as disconnected by their error handlers
  }

  // Additional Methods (continuing in local analysis pattern...)
  async getMonitoringStatus(tabId) {
    const context = this.taskContexts.get(tabId);
    const connectionStatus = this.getConnectionStatus();
    
    return {
      isActive: !!context,
      platform: context?.platform,
      taskContext: context,
      messageCount: context?.messageCount || 0,
      connectionStatus: connectionStatus,
      authStatus: {
        isAuthenticated: this.authState.isAuthenticated,
        userId: this.authState.userId
      },
      offlineMode: !this.isOnline || Object.keys(connectionStatus).every(k => !connectionStatus[k].connected),
      queueSize: this.messageQueue.length
    };
  }
}

// Initialize the hybrid supervisor service
const supervisorService = new HybridSupervisorService();
