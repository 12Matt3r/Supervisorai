// Enhanced Popup JavaScript - Hybrid Architecture Support
// Controls, status display, and configuration management for multiple deployment modes

class HybridSupervisorPopup {
  constructor() {
    this.currentTab = null;
    this.monitoringStatus = null;
    this.connectionStatus = {};
    this.syncStatus = null;
    
    this.init();
  }

  async init() {
    // Get current active tab
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    this.currentTab = tabs[0];
    
    // Set up UI event handlers
    this.setupEventHandlers();
    
    // Load saved settings
    await this.loadSettings();
    
    // Update status
    await this.updateStatus();
    
    // Set up periodic updates
    this.startPeriodicUpdates();
    
    // Listen for status updates from background
    chrome.runtime.onMessage.addListener((message) => {
      if (message.action === 'STATUS_UPDATE') {
        this.handleStatusUpdate(message.data);
      }
    });
  }

  setupEventHandlers() {
    // Deployment mode selection
    const deploymentMode = document.getElementById('deploymentMode');
    deploymentMode.addEventListener('change', (e) => {
      this.updateDeploymentMode(e.target.value);
    });
    
    // Connection management
    document.getElementById('reconnectBtn').addEventListener('click', () => {
      this.reconnectServices();
    });
    
    document.getElementById('configureEndpointsBtn').addEventListener('click', () => {
      this.showEndpointConfiguration();
    });
    
    // Intervention threshold slider
    const thresholdSlider = document.getElementById('interventionThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    
    thresholdSlider.addEventListener('input', (e) => {
      const value = parseFloat(e.target.value);
      let label;
      if (value <= 0.3) label = 'Low';
      else if (value <= 0.6) label = 'Moderate';
      else label = 'High';
      
      thresholdValue.textContent = `${value} - ${label}`;
      this.saveSettings();
    });
    
    // Checkboxes
    document.getElementById('enableNotifications').addEventListener('change', () => {
      this.saveSettings();
    });
    
    document.getElementById('enableAutoCorrection').addEventListener('change', () => {
      this.saveSettings();
    });
    
    document.getElementById('enableOfflineMode').addEventListener('change', () => {
      this.saveSettings();
    });
    
    document.getElementById('autoDetectEndpoints').addEventListener('change', () => {
      this.saveSettings();
    });
    
    // Buttons
    document.getElementById('setTaskBtn').addEventListener('click', () => {
      this.showTaskContextDialog();
    });
    
    document.getElementById('syncDataBtn').addEventListener('click', () => {
      this.forceSyncData();
    });
    
    document.getElementById('viewLogsBtn').addEventListener('click', () => {
      this.openActivityLog();
    });
    
    document.getElementById('clearDataBtn').addEventListener('click', () => {
      this.clearAllData();
    });
    
    document.getElementById('authBtn').addEventListener('click', () => {
      this.requestAuthentication();
    });
    
    // Mode-specific toggles
    document.getElementById('toggleLocalConnection').addEventListener('click', () => {
      this.toggleConnection('local');
    });
    
    document.getElementById('toggleCloudConnection').addEventListener('click', () => {
      this.toggleConnection('cloud');
    });
  }

  async loadSettings() {
    const settings = await chrome.storage.sync.get([
      'interventionThreshold',
      'enableNotifications', 
      'enableAutoCorrection',
      'deploymentMode',
      'autoDetectEndpoints',
      'enableOfflineMode',
      'supervisorServerUrl',
      'cloudEndpointUrl'
    ]);
    
    // Apply settings to UI
    const threshold = settings.interventionThreshold || 0.5;
    document.getElementById('interventionThreshold').value = threshold;
    
    let label;
    if (threshold <= 0.3) label = 'Low';
    else if (threshold <= 0.6) label = 'Moderate';
    else label = 'High';
    document.getElementById('thresholdValue').textContent = `${threshold} - ${label}`;
    
    document.getElementById('enableNotifications').checked = settings.enableNotifications !== false;
    document.getElementById('enableAutoCorrection').checked = settings.enableAutoCorrection !== false;
    document.getElementById('enableOfflineMode').checked = settings.enableOfflineMode !== false;
    document.getElementById('autoDetectEndpoints').checked = settings.autoDetectEndpoints !== false;
    
    // Deployment mode
    const deploymentMode = settings.deploymentMode || 'auto';
    document.getElementById('deploymentMode').value = deploymentMode;
    this.updateDeploymentModeUI(deploymentMode);
    
    // Endpoint URLs
    if (settings.supervisorServerUrl) {
      document.getElementById('localEndpointUrl').textContent = settings.supervisorServerUrl;
    }
    if (settings.cloudEndpointUrl) {
      document.getElementById('cloudEndpointUrl').textContent = settings.cloudEndpointUrl;
    }
  }

  async saveSettings() {
    const settings = {
      interventionThreshold: parseFloat(document.getElementById('interventionThreshold').value),
      enableNotifications: document.getElementById('enableNotifications').checked,
      enableAutoCorrection: document.getElementById('enableAutoCorrection').checked,
      enableOfflineMode: document.getElementById('enableOfflineMode').checked,
      autoDetectEndpoints: document.getElementById('autoDetectEndpoints').checked,
      deploymentMode: document.getElementById('deploymentMode').value
    };
    
    await chrome.storage.sync.set(settings);
  }

  async updateStatus() {
    try {
      // Get status from background service
      const response = await chrome.runtime.sendMessage({
        action: 'GET_STATUS',
        tabId: this.currentTab.id
      });
      
      if (response && response.success !== false) {
        this.updateStatusDisplay(response);
      } else {
        this.updateStatusDisplay({ isActive: false });
      }
    } catch (error) {
      console.error('Failed to get status:', error);
      this.updateStatusDisplay({ isActive: false });
    }
    
    // Update activity log
    await this.updateActivityLog();
  }

  handleStatusUpdate(data) {
    this.connectionStatus = data.connectionStatus || {};
    this.syncStatus = {
      queueSize: data.queueSize || 0,
      offlineDataSize: data.offlineDataSize || 0
    };
    
    this.updateConnectionStatus();
    this.updateSyncStatus();
    this.updateAuthStatus(data.authStatus);
  }

  updateStatusDisplay(status) {
    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');
    const platformName = document.getElementById('platformName');
    const taskContext = document.getElementById('taskContext');
    const taskDescription = document.getElementById('taskDescription');
    
    if (status.isActive) {
      statusDot.className = 'status-dot active';
      statusText.textContent = 'Monitoring active';
      platformName.textContent = status.platform || 'Unknown';
      
      if (status.taskContext && status.taskContext.mainGoal) {
        taskContext.style.display = 'block';
        taskDescription.textContent = status.taskContext.mainGoal;
      } else {
        taskContext.style.display = 'block';
        taskDescription.textContent = 'Waiting for task definition...';
      }
    } else {
      statusDot.className = 'status-dot inactive';
      statusText.textContent = 'No AI agent detected';
      platformName.textContent = '-';
      taskContext.style.display = 'none';
    }
    
    // Update message count
    document.getElementById('messageCount').textContent = status.messageCount || 0;
    
    // Update connection and sync status if available
    if (status.connectionStatus) {
      this.connectionStatus = status.connectionStatus;
      this.updateConnectionStatus();
    }
    
    if (status.offlineMode) {
      this.updateOfflineModeIndicator(true);
    }
    
    if (status.authStatus) {
      this.updateAuthStatus(status.authStatus);
    }
    
    if (typeof status.queueSize !== 'undefined') {
      this.syncStatus = {
        queueSize: status.queueSize,
        offlineDataSize: status.offlineDataSize || 0
      };
      this.updateSyncStatus();
    }
  }

  updateConnectionStatus() {
    const localStatus = document.getElementById('localConnectionStatus');
    const cloudStatus = document.getElementById('cloudConnectionStatus');
    const overallStatus = document.getElementById('connectionStatus');
    
    // Update individual connection statuses
    if (this.connectionStatus.local) {
      const local = this.connectionStatus.local;
      localStatus.innerHTML = local.connected 
        ? '🟢 Local Connected' 
        : '🔴 Local Disconnected';
      localStatus.title = `URL: ${local.url}\nMessages: ${local.messageCount}`;
    } else {
      localStatus.innerHTML = '⚪ Local Not Configured';
    }
    
    if (this.connectionStatus.cloud) {
      const cloud = this.connectionStatus.cloud;
      cloudStatus.innerHTML = cloud.connected 
        ? '🟢 Cloud Connected' 
        : '🔴 Cloud Disconnected';
      cloudStatus.title = `URL: ${cloud.url}\nMessages: ${cloud.messageCount}`;
    } else {
      cloudStatus.innerHTML = '⚪ Cloud Not Configured';
    }
    
    // Update overall status
    const hasConnections = Object.values(this.connectionStatus).some(conn => conn.connected);
    if (hasConnections) {
      overallStatus.innerHTML = '🔗 Connected to supervisor services';
      overallStatus.className = 'connection-status connected';
    } else {
      overallStatus.innerHTML = '❌ No supervisor connections';
      overallStatus.className = 'connection-status disconnected';
    }
    
    // Update connection controls visibility
    this.updateConnectionControls();
  }

  updateConnectionControls() {
    const localToggle = document.getElementById('toggleLocalConnection');
    const cloudToggle = document.getElementById('toggleCloudConnection');
    
    // Update local connection toggle
    if (this.connectionStatus.local) {
      localToggle.style.display = 'inline-block';
      localToggle.textContent = this.connectionStatus.local.connected 
        ? 'Disconnect Local' 
        : 'Connect Local';
    } else {
      localToggle.style.display = 'none';
    }
    
    // Update cloud connection toggle
    if (this.connectionStatus.cloud) {
      cloudToggle.style.display = 'inline-block';
      cloudToggle.textContent = this.connectionStatus.cloud.connected 
        ? 'Disconnect Cloud' 
        : 'Connect Cloud';
    } else {
      cloudToggle.style.display = 'none';
    }
  }

  updateSyncStatus() {
    const syncIndicator = document.getElementById('syncStatus');
    const queueSize = document.getElementById('queueSize');
    const offlineData = document.getElementById('offlineDataSize');
    
    if (this.syncStatus) {
      queueSize.textContent = this.syncStatus.queueSize;
      offlineData.textContent = this.syncStatus.offlineDataSize;
      
      if (this.syncStatus.queueSize > 0 || this.syncStatus.offlineDataSize > 0) {
        syncIndicator.innerHTML = '🔄 Sync Pending';
        syncIndicator.className = 'sync-status pending';
      } else {
        syncIndicator.innerHTML = '✅ Synced';
        syncIndicator.className = 'sync-status synced';
      }
    } else {
      syncIndicator.innerHTML = '❓ Sync Status Unknown';
      syncIndicator.className = 'sync-status unknown';
    }
  }

  updateAuthStatus(authStatus) {
    const authIndicator = document.getElementById('authStatus');
    const authBtn = document.getElementById('authBtn');
    const userInfo = document.getElementById('userInfo');
    
    if (authStatus && authStatus.isAuthenticated) {
      authIndicator.innerHTML = '🔐 Authenticated';
      authIndicator.className = 'auth-status authenticated';
      authBtn.textContent = 'Re-authenticate';
      
      if (authStatus.userId) {
        const userId = authStatus.userId.length > 20 
          ? authStatus.userId.substring(0, 17) + '...' 
          : authStatus.userId;
        userInfo.textContent = `User: ${userId}`;
        userInfo.style.display = 'block';
      }
    } else {
      authIndicator.innerHTML = '🔓 Not Authenticated';
      authIndicator.className = 'auth-status unauthenticated';
      authBtn.textContent = 'Authenticate';
      userInfo.style.display = 'none';
    }
  }

  updateDeploymentModeUI(mode) {
    const modeDescription = document.getElementById('modeDescription');
    const connectionDetails = document.getElementById('connectionDetails');
    
    const descriptions = {
      'auto': 'Automatically connects to available services (local first, then cloud)',
      'local': 'Connects only to local supervisor server',
      'cloud': 'Connects only to cloud-based services',
      'hybrid': 'Maintains connections to both local and cloud services'
    };
    
    modeDescription.textContent = descriptions[mode] || 'Unknown mode';
    
    // Show/hide connection details based on mode
    if (mode === 'auto' || mode === 'hybrid') {
      connectionDetails.style.display = 'block';
    } else {
      connectionDetails.style.display = 'block'; // Always show for configuration
    }
  }

  updateOfflineModeIndicator(isOffline) {
    const offlineIndicator = document.getElementById('offlineIndicator');
    
    if (isOffline) {
      offlineIndicator.innerHTML = '📵 Offline Mode';
      offlineIndicator.style.display = 'block';
      offlineIndicator.className = 'offline-indicator active';
    } else {
      offlineIndicator.style.display = 'none';
    }
  }

  async updateActivityLog() {
    try {
      const result = await chrome.storage.local.get(['interactionLog']);
      const log = result.interactionLog || [];
      
      // Get recent activities (last 10)
      const recentActivities = log
        .filter(entry => entry.tabId === this.currentTab.id)
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, 5);
      
      const activityList = document.getElementById('activityList');
      
      if (recentActivities.length === 0) {
        activityList.innerHTML = `
          <div style="text-align: center; color: #6c757d; font-size: 12px; padding: 20px;">
            No activity yet
          </div>
        `;
        return;
      }
      
      activityList.innerHTML = recentActivities.map(activity => {
        const time = new Date(activity.timestamp).toLocaleTimeString([], {
          hour: '2-digit',
          minute: '2-digit'
        });
        
        let icon = 'analysis';
        let description = 'Message analyzed';
        
        if (activity.type === 'intervention') {
          icon = 'intervention';
          description = `Intervention: ${activity.severity} drift`;
        } else if (activity.driftScore > 0.6) {
          icon = 'warning';
          description = `High drift detected (${(activity.driftScore * 100).toFixed(0)}%)`;
        } else if (activity.driftScore > 0.3) {
          icon = 'warning';
          description = `Moderate drift detected (${(activity.driftScore * 100).toFixed(0)}%)`;
        }
        
        return `
          <div class="activity-item">
            <div class="activity-icon ${icon}"></div>
            <div style="flex: 1;">
              <div>${description}</div>
              <div style="color: #6c757d; font-size: 11px;">${time}</div>
            </div>
          </div>
        `;
      }).join('');
      
      // Update intervention count
      const interventionCount = log.filter(entry => 
        entry.tabId === this.currentTab.id && entry.type === 'intervention'
      ).length;
      
      document.getElementById('interventionCount').textContent = interventionCount;
      
    } catch (error) {
      console.error('Failed to update activity log:', error);
    }
  }

  async updateDeploymentMode(mode) {
    await chrome.storage.sync.set({ deploymentMode: mode });
    this.updateDeploymentModeUI(mode);
    
    // Notify background service of mode change
    chrome.runtime.sendMessage({
      action: 'UPDATE_DEPLOYMENT_MODE',
      mode: mode
    });
  }

  showTaskContextDialog() {
    const mainGoal = prompt(
      'Set the main task/goal for the AI agent to focus on:',
      'Build a social media app'
    );
    
    if (mainGoal && mainGoal.trim()) {
      this.setTaskContext(mainGoal.trim());
    }
  }

  async setTaskContext(mainGoal) {
    try {
      await chrome.tabs.sendMessage(this.currentTab.id, {
        action: 'SET_TASK_CONTEXT',
        data: {
          mainGoal,
          timestamp: Date.now()
        }
      });
      
      // Update display
      document.getElementById('taskDescription').textContent = mainGoal;
      document.getElementById('taskContext').style.display = 'block';
      
    } catch (error) {
      alert('Failed to set task context. Make sure you\'re on a page with an AI agent.');
    }
  }

  showEndpointConfiguration() {
    const localUrl = prompt(
      'Enter local supervisor server URL:',
      document.getElementById('localEndpointUrl').textContent || 'ws://localhost:8765'
    );
    
    if (localUrl) {
      chrome.storage.sync.set({ supervisorServerUrl: localUrl });
      document.getElementById('localEndpointUrl').textContent = localUrl;
    }
    
    const cloudUrl = prompt(
      'Enter cloud endpoint URL (optional):',
      document.getElementById('cloudEndpointUrl').textContent || ''
    );
    
    if (cloudUrl) {
      chrome.storage.sync.set({ cloudEndpointUrl: cloudUrl });
      document.getElementById('cloudEndpointUrl').textContent = cloudUrl;
    }
    
    // Trigger reconnection
    this.reconnectServices();
  }

  async reconnectServices() {
    try {
      await chrome.runtime.sendMessage({ action: 'RECONNECT_SERVICES' });
      
      // Show feedback
      const btn = document.getElementById('reconnectBtn');
      const originalText = btn.textContent;
      btn.textContent = 'Reconnecting...';
      btn.disabled = true;
      
      setTimeout(() => {
        btn.textContent = originalText;
        btn.disabled = false;
      }, 3000);
      
    } catch (error) {
      console.error('Failed to reconnect services:', error);
      alert('Failed to reconnect services. Check the console for details.');
    }
  }

  async toggleConnection(connectionType) {
    try {
      await chrome.runtime.sendMessage({ 
        action: 'TOGGLE_CONNECTION', 
        connectionType: connectionType 
      });
    } catch (error) {
      console.error(`Failed to toggle ${connectionType} connection:`, error);
    }
  }

  async forceSyncData() {
    try {
      const response = await chrome.runtime.sendMessage({ action: 'FORCE_SYNC' });
      
      if (response && response.success) {
        const btn = document.getElementById('syncDataBtn');
        const originalText = btn.textContent;
        btn.textContent = 'Syncing...';
        btn.disabled = true;
        
        setTimeout(() => {
          btn.textContent = originalText;
          btn.disabled = false;
        }, 2000);
      } else {
        alert('Sync failed or no data to sync');
      }
    } catch (error) {
      console.error('Failed to force sync:', error);
      alert('Sync failed. Check connection status.');
    }
  }

  async requestAuthentication() {
    try {
      const response = await chrome.runtime.sendMessage({ action: 'REQUEST_AUTH' });
      
      if (response && response.success) {
        alert(`Authentication successful via ${response.method}`);
        await this.updateStatus(); // Refresh status to show auth state
      } else {
        alert(`Authentication failed: ${response.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Failed to authenticate:', error);
      alert('Authentication request failed');
    }
  }

  openActivityLog() {
    // Open a new tab with detailed activity log
    chrome.tabs.create({
      url: chrome.runtime.getURL('activity-log.html')
    });
  }

  async clearAllData() {
    if (confirm('Clear all monitoring data, activity logs, and offline cache?')) {
      try {
        // Clear local storage
        await chrome.storage.local.clear();
        
        // Clear sync storage (keep essential settings)
        const essentialSettings = await chrome.storage.sync.get([
          'deploymentMode', 
          'supervisorServerUrl', 
          'cloudEndpointUrl'
        ]);
        await chrome.storage.sync.clear();
        await chrome.storage.sync.set(essentialSettings);
        
        // Notify background service
        await chrome.runtime.sendMessage({ action: 'CLEAR_ALL_DATA' });
        
        // Refresh UI
        await this.updateActivityLog();
        alert('Data cleared successfully!');
        
      } catch (error) {
        console.error('Failed to clear data:', error);
        alert('Failed to clear some data. Check console for details.');
      }
    }
  }

  startPeriodicUpdates() {
    // Update status every 5 seconds
    setInterval(() => {
      this.updateStatus();
    }, 5000);
    
    // Update connection status more frequently
    setInterval(() => {
      this.updateConnectionStatus();
    }, 2000);
  }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new HybridSupervisorPopup();
});
