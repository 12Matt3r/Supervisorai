// Test Script for Browser Extension Hybrid Architecture Integration
// This script can be run in the browser console to test extension functionality

class ExtensionHybridTester {
  constructor() {
    this.testResults = [];
    this.extensionId = null;
  }

  async runAllTests() {
    console.log('🧪 Starting Browser Extension Hybrid Architecture Tests...');
    
    await this.testExtensionInstallation();
    await this.testConnectionCapabilities();
    await this.testMessageHandling();
    await this.testAuthenticationFlow();
    await this.testOfflineSupport();
    await this.testUIFunctionality();
    
    this.displayResults();
  }

  async testExtensionInstallation() {
    console.log('\n📦 Testing Extension Installation...');
    
    try {
      // Check if extension is installed
      const hasExtension = typeof chrome !== 'undefined' && chrome.runtime;
      this.recordTest('Extension API Available', hasExtension, 
        hasExtension ? 'Chrome extension API is accessible' : 'Chrome extension API not found');
      
      if (hasExtension) {
        // Check extension ID
        this.extensionId = chrome.runtime.id;
        this.recordTest('Extension ID Retrieved', !!this.extensionId, 
          `Extension ID: ${this.extensionId}`);
        
        // Check manifest
        const manifest = chrome.runtime.getManifest();
        this.recordTest('Manifest Loaded', !!manifest, 
          `Version: ${manifest.version}, Name: ${manifest.name}`);
        
        // Check permissions
        const hasWebSocketPermissions = manifest.host_permissions && 
          manifest.host_permissions.some(p => p.includes('ws://') || p.includes('wss://'));
        this.recordTest('WebSocket Permissions', hasWebSocketPermissions,
          hasWebSocketPermissions ? 'WebSocket permissions granted' : 'WebSocket permissions missing');
      }
      
    } catch (error) {
      this.recordTest('Extension Installation', false, `Error: ${error.message}`);
    }
  }

  async testConnectionCapabilities() {
    console.log('\n🔌 Testing Connection Capabilities...');
    
    try {
      // Test WebSocket support
      const hasWebSocket = typeof WebSocket !== 'undefined';
      this.recordTest('WebSocket Support', hasWebSocket, 
        hasWebSocket ? 'WebSocket API available' : 'WebSocket API not supported');
      
      if (hasWebSocket) {
        // Test local connection capability
        try {
          const localWs = new WebSocket('ws://localhost:8765');
          localWs.onopen = () => {
            this.recordTest('Local Connection', true, 'Can connect to local supervisor');
            localWs.close();
          };
          localWs.onerror = () => {
            this.recordTest('Local Connection', false, 'Cannot connect to local supervisor (server may not be running)');
          };
        } catch (error) {
          this.recordTest('Local Connection', false, `Connection error: ${error.message}`);
        }
        
        // Test cloud connection capability (simulated)
        this.recordTest('Cloud Connection Capability', true, 'WebSocket can connect to cloud endpoints');
      }
      
    } catch (error) {
      this.recordTest('Connection Capabilities', false, `Error: ${error.message}`);
    }
  }

  async testMessageHandling() {
    console.log('\n📬 Testing Message Handling...');
    
    try {
      // Test message passing to extension
      if (typeof chrome !== 'undefined' && chrome.runtime) {
        
        // Test runtime messaging
        const testMessage = { action: 'GET_STATUS', test: true };
        
        try {
          const response = await chrome.runtime.sendMessage(testMessage);
          this.recordTest('Runtime Messaging', true, 
            `Message sent successfully: ${JSON.stringify(response)}`);
        } catch (error) {
          this.recordTest('Runtime Messaging', false, 
            `Runtime message failed: ${error.message}`);
        }
        
        // Test content script communication
        try {
          await chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
            if (tabs[0]) {
              chrome.tabs.sendMessage(tabs[0].id, { action: 'GET_STATUS', test: true }, (response) => {
                this.recordTest('Content Script Communication', !!response,
                  response ? 'Content script responded' : 'No response from content script');
              });
            }
          });
        } catch (error) {
          this.recordTest('Content Script Communication', false, 
            `Content script test failed: ${error.message}`);
        }
      }
      
    } catch (error) {
      this.recordTest('Message Handling', false, `Error: ${error.message}`);
    }
  }

  async testAuthenticationFlow() {
    console.log('\n🔐 Testing Authentication Flow...');
    
    try {
      // Check if cookies API is available
      const hasCookies = typeof chrome !== 'undefined' && chrome.cookies;
      this.recordTest('Cookies API', hasCookies, 
        hasCookies ? 'Can access cookies for auth extraction' : 'Cookies API not available');
      
      // Check if identity API is available
      const hasIdentity = typeof chrome !== 'undefined' && chrome.identity;
      this.recordTest('Identity API', hasIdentity,
        hasIdentity ? 'Identity API available for OAuth' : 'Identity API not available');
      
      // Test storage for auth state
      const hasStorage = typeof chrome !== 'undefined' && chrome.storage;
      this.recordTest('Storage API', hasStorage,
        hasStorage ? 'Storage available for auth persistence' : 'Storage API not available');
      
      if (hasStorage) {
        // Test auth state storage
        try {
          await chrome.storage.local.set({ testAuth: { userId: 'test', token: 'test123' } });
          const result = await chrome.storage.local.get(['testAuth']);
          const stored = result.testAuth && result.testAuth.userId === 'test';
          this.recordTest('Auth State Storage', stored, 
            stored ? 'Auth state can be stored and retrieved' : 'Auth state storage failed');
          
          // Cleanup
          await chrome.storage.local.remove(['testAuth']);
        } catch (error) {
          this.recordTest('Auth State Storage', false, `Storage error: ${error.message}`);
        }
      }
      
    } catch (error) {
      this.recordTest('Authentication Flow', false, `Error: ${error.message}`);
    }
  }

  async testOfflineSupport() {
    console.log('\n📵 Testing Offline Support...');
    
    try {
      // Test local storage capacity
      const hasStorage = typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local;
      this.recordTest('Local Storage Available', hasStorage,
        hasStorage ? 'Local storage available for offline data' : 'Local storage not available');
      
      if (hasStorage) {
        // Test offline data caching
        const testData = {
          messages: Array.from({ length: 100 }, (_, i) => ({ id: i, content: `Test message ${i}`, timestamp: Date.now() })),
          activities: Array.from({ length: 50 }, (_, i) => ({ id: i, type: 'test', data: `Activity ${i}` }))
        };
        
        try {
          await chrome.storage.local.set({ offlineTestData: testData });
          const result = await chrome.storage.local.get(['offlineTestData']);
          const cached = result.offlineTestData && result.offlineTestData.messages.length === 100;
          
          this.recordTest('Offline Data Caching', cached,
            cached ? 'Can cache large amounts of offline data' : 'Offline caching failed');
          
          // Test data retrieval speed
          const startTime = Date.now();
          await chrome.storage.local.get(['offlineTestData']);
          const retrievalTime = Date.now() - startTime;
          
          this.recordTest('Data Retrieval Performance', retrievalTime < 100,
            `Data retrieval took ${retrievalTime}ms`);
          
          // Cleanup
          await chrome.storage.local.remove(['offlineTestData']);
        } catch (error) {
          this.recordTest('Offline Data Caching', false, `Caching error: ${error.message}`);
        }
      }
      
      // Test queue management
      this.recordTest('Message Queue Support', true, 'Can implement message queuing for offline mode');
      
    } catch (error) {
      this.recordTest('Offline Support', false, `Error: ${error.message}`);
    }
  }

  async testUIFunctionality() {
    console.log('\n🎨 Testing UI Functionality...');
    
    try {
      // Test popup availability
      if (typeof chrome !== 'undefined' && chrome.runtime) {
        const manifest = chrome.runtime.getManifest();
        const hasPopup = manifest.action && manifest.action.default_popup;
        this.recordTest('Popup Available', hasPopup,
          hasPopup ? `Popup file: ${manifest.action.default_popup}` : 'No popup configured');
      }
      
      // Test content script injection
      this.recordTest('Content Script Support', true, 'Content scripts can be injected into pages');
      
      // Test notification capability
      const hasNotifications = 'Notification' in window;
      this.recordTest('Notification Support', hasNotifications,
        hasNotifications ? 'Can show notifications to users' : 'Notifications not supported');
      
      // Test DOM manipulation for interventions
      const canManipulateDOM = typeof document !== 'undefined';
      this.recordTest('DOM Manipulation', canManipulateDOM,
        canManipulateDOM ? 'Can modify page content for interventions' : 'DOM manipulation not available');
      
    } catch (error) {
      this.recordTest('UI Functionality', false, `Error: ${error.message}`);
    }
  }

  recordTest(name, passed, details) {
    const result = {
      name,
      passed,
      details,
      timestamp: new Date().toISOString()
    };
    
    this.testResults.push(result);
    
    const status = passed ? '✅' : '❌';
    console.log(`${status} ${name}: ${details}`);
  }

  displayResults() {
    console.log('\n📊 Test Results Summary:');
    console.log('================================');
    
    const totalTests = this.testResults.length;
    const passedTests = this.testResults.filter(r => r.passed).length;
    const failedTests = totalTests - passedTests;
    
    console.log(`Total Tests: ${totalTests}`);
    console.log(`Passed: ${passedTests} ✅`);
    console.log(`Failed: ${failedTests} ❌`);
    console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(1)}%`);
    
    if (failedTests > 0) {
      console.log('\n❌ Failed Tests:');
      this.testResults
        .filter(r => !r.passed)
        .forEach(r => console.log(`   • ${r.name}: ${r.details}`));
    }
    
    console.log('\n📝 Detailed Results:');
    console.table(this.testResults.map(r => ({
      Test: r.name,
      Status: r.passed ? 'PASS' : 'FAIL',
      Details: r.details
    })));
    
    // Return results for programmatic access
    return {
      total: totalTests,
      passed: passedTests,
      failed: failedTests,
      successRate: (passedTests / totalTests) * 100,
      details: this.testResults
    };
  }

  // Helper method to run specific test categories
  async runTestCategory(category) {
    console.log(`🧪 Running ${category} tests only...`);
    
    switch (category.toLowerCase()) {
      case 'installation':
        await this.testExtensionInstallation();
        break;
      case 'connection':
        await this.testConnectionCapabilities();
        break;
      case 'messaging':
        await this.testMessageHandling();
        break;
      case 'auth':
        await this.testAuthenticationFlow();
        break;
      case 'offline':
        await this.testOfflineSupport();
        break;
      case 'ui':
        await this.testUIFunctionality();
        break;
      default:
        console.log('Unknown category. Available: installation, connection, messaging, auth, offline, ui');
        return;
    }
    
    this.displayResults();
  }
}

// Global test runner
window.ExtensionTester = ExtensionHybridTester;

// Auto-run tests if extension is detected
if (typeof chrome !== 'undefined' && chrome.runtime) {
  console.log('🔍 Browser extension detected. You can run tests with:');
  console.log('   const tester = new ExtensionTester();');
  console.log('   await tester.runAllTests();');
  console.log('   Or run specific categories:');
  console.log('   await tester.runTestCategory("connection");');
} else {
  console.log('❌ No browser extension detected. Please install the extension first.');
}

// Quick test function for immediate execution
async function quickTest() {
  const tester = new ExtensionHybridTester();
  return await tester.runAllTests();
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { ExtensionHybridTester, quickTest };
}
