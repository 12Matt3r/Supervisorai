# Browser Extension - Hybrid Architecture Integration

This document describes the enhanced browser extension that integrates with the hybrid supervisor architecture, supporting both local and cloud-based AI monitoring services.

## Overview

The browser extension has been upgraded to work seamlessly with multiple deployment modes:
- **Local Mode**: Connects to local supervisor server (MCP)
- **Cloud Mode**: Connects to cloud-based services (Supabase)
- **Hybrid Mode**: Maintains connections to both local and cloud services
- **Auto Mode**: Automatically detects and connects to available services

## Architecture Components

### 1. Enhanced Background Service (`background.js`)

The background service worker now supports:

- **Multiple WebSocket Connections**: Simultaneous connections to local and cloud endpoints
- **Connection Fallback**: Automatic failover between endpoints
- **Authentication Integration**: Support for both web app sessions and extension-based auth
- **Offline Data Caching**: Queue messages and sync when connections are restored
- **Real-time Status Broadcasting**: Updates popup and content scripts with connection status

#### Key Features:

```javascript
class HybridSupervisorService {
  // Connection management
  endpoints: {
    local: { url: 'ws://localhost:8765', priority: 1, type: 'local' },
    cloud: { url: 'wss://project.supabase.co/functions/v1/websocket-handler', priority: 2, type: 'cloud' }
  }
  
  // Deployment modes
  deploymentMode: 'auto' | 'local' | 'cloud' | 'hybrid'
  
  // Data synchronization
  messageQueue: []
  offlineData: Map()
  syncQueue: Set()
}
```

### 2. Enhanced Popup Interface (`popup.html` + `popup.js`)

The popup now provides comprehensive hybrid architecture management:

#### New UI Sections:

- **Connection Management**: View and control connections to different endpoints
- **Deployment Mode Selection**: Switch between auto, local, cloud, and hybrid modes
- **Authentication Status**: Shows authentication state and user information
- **Data Synchronization**: Displays sync status, queue size, and offline data
- **Offline Mode Indicator**: Visual feedback when operating offline

#### Connection Status Display:

```
🟢 Local Connected    | 🔴 Local Disconnected
🟢 Cloud Connected    | 🔴 Cloud Disconnected
```

### 3. Enhanced Content Script (`content.js`)

The content script now provides better integration with the hybrid system:

- **Enhanced Message Handling**: Richer metadata collection for analysis
- **Real-time Status Updates**: Receives connection and auth status changes
- **Offline Message Queuing**: Caches messages when connections are unavailable
- **Visual Feedback**: Shows connection status and auth state to users

### 4. Updated Manifest (`manifest.json`)

Added permissions for hybrid architecture:

```json
{
  "permissions": [
    "activeTab", "storage", "webNavigation", "tabs",
    "identity", "cookies", "alarms", "background"
  ],
  "host_permissions": [
    "<all_urls>",
    "ws://localhost:*/*",
    "wss://localhost:*/*", 
    "https://*.supabase.co/*",
    "wss://*.supabase.co/*"
  ]
}
```

## Configuration

### Deployment Modes

1. **Auto Mode** (Default)
   - Tries local connection first
   - Falls back to cloud if local unavailable
   - Automatically reconnects when services come online

2. **Local Mode**
   - Connects only to local supervisor server
   - Best for development and local deployments

3. **Cloud Mode**
   - Connects only to cloud-based services
   - Best for production web applications

4. **Hybrid Mode**
   - Maintains connections to both local and cloud
   - Provides redundancy and enhanced capabilities

### Endpoint Configuration

Users can configure endpoints through the popup:
- **Local Endpoint**: `ws://localhost:8765` (default)
- **Cloud Endpoint**: Auto-detected or manually configured

### Settings Storage

Settings are stored in Chrome sync storage:
```javascript
{
  deploymentMode: 'auto',
  supervisorServerUrl: 'ws://localhost:8765',
  cloudEndpointUrl: 'wss://project.supabase.co/functions/v1/websocket-handler',
  autoDetectEndpoints: true,
  enableOfflineMode: true
}
```

## Message Protocol

### WebSocket Message Types

The extension communicates with supervisor services using these message types:

#### Extension → Supervisor
- `EXTENSION_REGISTER`: Register extension with capabilities
- `USER_INPUT_ANALYSIS`: Send user input for analysis
- `AGENT_MESSAGE_ANALYSIS`: Send agent response for drift detection
- `OFFLINE_DATA_SYNC`: Sync cached offline data

#### Supervisor → Extension
- `connection_established`: Confirm connection with server info
- `intervention_required`: Request intervention for task drift
- `sync_request`: Request data synchronization
- `auth_token_refresh`: Update authentication tokens

### Authentication Flow

1. **Web App Session**: Extract auth from current tab cookies/storage
2. **Extension Auth**: Generate extension-specific authentication
3. **Token Refresh**: Handle token expiration and renewal

## Data Synchronization

### Offline Support

The extension provides robust offline functionality:

- **Message Queuing**: Store messages when connections are unavailable
- **Data Caching**: Cache activity logs and task contexts locally
- **Automatic Sync**: Sync cached data when connections are restored
- **Conflict Resolution**: Handle sync conflicts intelligently

### Sync Process

1. **Queue Messages**: Store outbound messages when offline
2. **Cache Activity**: Store interaction logs locally
3. **Detect Reconnection**: Monitor connection status changes
4. **Sync Data**: Upload cached data to available services
5. **Clean Cache**: Remove successfully synced data

## Installation & Usage

### Development Setup

1. Load the extension in Chrome Developer Mode
2. Configure local supervisor server URL if different from default
3. Set deployment mode based on your environment
4. Test connections using the popup interface

### Production Deployment

1. Configure cloud endpoint URLs
2. Set appropriate deployment mode
3. Ensure CORS settings allow extension domain
4. Test authentication flow with web application

## Error Handling & Monitoring

### Connection Resilience

- **Automatic Retry**: Exponential backoff for failed connections
- **Health Checks**: Periodic ping messages to verify connection health
- **Graceful Degradation**: Continue operation with reduced functionality when offline

### Error Recovery

- **Connection Failures**: Automatic retry with fallback endpoints
- **Authentication Errors**: Request re-authentication from user
- **Sync Failures**: Re-queue failed messages with attempt limits

### Debug Information

The extension provides comprehensive debugging:
- Console logs with prefixed identifiers
- Connection status in popup interface
- Message queue sizes and sync statistics
- Error notifications with actionable information

## Security Considerations

### Authentication

- **Token Storage**: Secure storage in Chrome's encrypted storage APIs
- **CSRF Protection**: Include anti-CSRF tokens in requests
- **Session Validation**: Regular validation of authentication tokens

### Cross-Origin Communication

- **Manifest Permissions**: Explicit host permissions for WebSocket endpoints
- **Message Validation**: Validate all incoming messages from supervisors
- **Origin Checking**: Verify message origins for injected scripts

### Data Privacy

- **Local Storage**: Sensitive data stays in local extension storage
- **Transmission Security**: Use WSS for cloud connections
- **Data Minimization**: Only transmit necessary data for analysis

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check firewall settings for WebSocket connections
   - Verify endpoint URLs are correct
   - Ensure supervisor services are running

2. **Authentication Problems** 
   - Clear extension storage and re-authenticate
   - Check web app session cookies
   - Verify cloud service authentication

3. **Sync Issues**
   - Force sync using popup button
   - Check connection status
   - Clear offline cache if corrupted

### Debug Commands

Access debug information via popup interface:
- Connection status and message counts
- Queue sizes and sync statistics
- Error logs and recent activity

## Future Enhancements

### Planned Features

- **Multi-tab Session Management**: Coordinate between multiple AI agent tabs
- **Advanced Analytics**: Enhanced drift detection with machine learning
- **Team Collaboration**: Share task contexts and interventions
- **Integration APIs**: Webhooks and external service integrations

### Extensibility

The architecture supports easy extension with:
- Custom message handlers for new supervisor services
- Plugin system for additional analysis capabilities
- Configurable intervention strategies
- Custom UI themes and layouts

## Support

For issues and feature requests:
- Check console logs for error messages
- Use popup interface to view connection status
- Report issues with specific error messages and configurations
- Include browser version and extension settings in bug reports
