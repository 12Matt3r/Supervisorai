#!/usr/bin/env python3
"""
Real-time Alert System for Supervisor Agent
Handles critical deviation detection, multi-channel notifications, and alert prioritization.
"""

import json
import logging
import smtplib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict
import hashlib
import time

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    severity: str  # 'critical', 'warning', 'info'
    title: str
    message: str
    source: str
    timestamp: str
    metadata: Dict[str, Any]
    resolved: bool = False
    acknowledged: bool = False

class AlertDeduplicator:
    """Handles alert deduplication to prevent spam"""
    
    def __init__(self, window_minutes: int = 30):
        self.window_minutes = window_minutes
        self.alert_history = {}
    
    def should_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on deduplication rules"""
        alert_key = self._get_alert_key(alert)
        current_time = datetime.now()
        
        if alert_key in self.alert_history:
            last_sent = self.alert_history[alert_key]['last_sent']
            if current_time - last_sent < timedelta(minutes=self.window_minutes):
                self.alert_history[alert_key]['count'] += 1
                return False
        
        self.alert_history[alert_key] = {
            'last_sent': current_time,
            'count': 1
        }
        return True
    
    def _get_alert_key(self, alert: Alert) -> str:
        """Generate unique key for alert deduplication"""
        key_data = f"{alert.severity}:{alert.title}:{alert.source}"
        return hashlib.md5(key_data.encode()).hexdigest()

class NotificationChannel:
    """Base class for notification channels"""
    
    def send(self, alert: Alert) -> bool:
        """Send alert notification"""
        raise NotImplementedError

class EmailNotifier(NotificationChannel):
    """Email notification channel"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, 
                 password: str, recipients: List[str]):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
    
    def send(self, alert: Alert) -> bool:
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = ', '.join(self.recipients)
            msg['Subject'] = f"[{alert.severity.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Severity: {alert.severity}
            - Source: {alert.source}
            - Time: {alert.timestamp}
            - Message: {alert.message}
            
            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.username, self.recipients, text)
            server.quit()
            
            return True
        except Exception as e:
            logging.error(f"Email notification failed: {e}")
            return False

class SlackNotifier(NotificationChannel):
    """Slack notification channel"""
    
    def __init__(self, webhook_url: str, channel: Optional[str] = None):
        self.webhook_url = webhook_url
        self.channel = channel
    
    def send(self, alert: Alert) -> bool:
        """Send Slack alert"""
        try:
            color_map = {
                'critical': '#ff0000',
                'warning': '#ffaa00',
                'info': '#00ff00'
            }
            
            payload = {
                'text': f'Supervisor Agent Alert: {alert.title}',
                'attachments': [{
                    'color': color_map.get(alert.severity, '#cccccc'),
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity, 'short': True},
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Time', 'value': alert.timestamp, 'short': False},
                        {'title': 'Message', 'value': alert.message, 'short': False}
                    ]
                }]
            }
            
            if self.channel:
                payload['channel'] = self.channel
            
            response = requests.post(self.webhook_url, json=payload)
            return response.status_code == 200
        except Exception as e:
            logging.error(f"Slack notification failed: {e}")
            return False

class WebhookNotifier(NotificationChannel):
    """Generic webhook notification channel"""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
    
    def send(self, alert: Alert) -> bool:
        """Send webhook alert"""
        try:
            payload = {
                'alert': asdict(alert),
                'timestamp': datetime.now().isoformat()
            }
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                headers=self.headers
            )
            return response.status_code < 400
        except Exception as e:
            logging.error(f"Webhook notification failed: {e}")
            return False

class RealTimeAlertSystem:
    """Main real-time alerting system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.deduplicator = AlertDeduplicator(
            window_minutes=config.get('deduplication_window', 30)
        )
        self.channels = self._initialize_channels()
        self.alert_store = []
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_channels(self) -> List[NotificationChannel]:
        """Initialize notification channels from config"""
        channels = []
        
        # Email channel
        if 'email' in self.config:
            email_config = self.config['email']
            channels.append(EmailNotifier(
                smtp_host=email_config['smtp_host'],
                smtp_port=email_config['smtp_port'],
                username=email_config['username'],
                password=email_config['password'],
                recipients=email_config['recipients']
            ))
        
        # Slack channel
        if 'slack' in self.config:
            slack_config = self.config['slack']
            channels.append(SlackNotifier(
                webhook_url=slack_config['webhook_url'],
                channel=slack_config.get('channel')
            ))
        
        # Webhook channels
        if 'webhooks' in self.config:
            for webhook_config in self.config['webhooks']:
                channels.append(WebhookNotifier(
                    webhook_url=webhook_config['url'],
                    headers=webhook_config.get('headers', {})
                ))
        
        return channels
    
    def send_alert(self, severity: str, title: str, message: str, 
                   source: str, metadata: Optional[Dict] = None) -> str:
        """Send alert through all configured channels"""
        alert_id = hashlib.sha256(
            f"{datetime.now().isoformat()}:{title}:{source}".encode()
        ).hexdigest()[:12]
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            source=source,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        
        # Store alert
        self.alert_store.append(alert)
        
        # Check deduplication
        if not self.deduplicator.should_alert(alert):
            self.logger.info(f"Alert {alert_id} deduplicated")
            return alert_id
        
        # Send through all channels
        for channel in self.channels:
            try:
                success = channel.send(alert)
                if success:
                    self.logger.info(
                        f"Alert {alert_id} sent via {channel.__class__.__name__}"
                    )
                else:
                    self.logger.warning(
                        f"Alert {alert_id} failed via {channel.__class__.__name__}"
                    )
            except Exception as e:
                self.logger.error(
                    f"Channel {channel.__class__.__name__} error: {e}"
                )
        
        return alert_id
    
    def get_alerts(self, since: Optional[str] = None, 
                   severity: Optional[str] = None) -> List[Alert]:
        """Get alerts with optional filtering"""
        alerts = self.alert_store.copy()
        
        if since:
            since_dt = datetime.fromisoformat(since)
            alerts = [
                alert for alert in alerts 
                if datetime.fromisoformat(alert.timestamp) >= since_dt
            ]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark alert as acknowledged"""
        for alert in self.alert_store:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        for alert in self.alert_store:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics"""
        stats = {
            'total_alerts': len(self.alert_store),
            'by_severity': defaultdict(int),
            'by_source': defaultdict(int),
            'unresolved': 0,
            'unacknowledged': 0
        }
        
        for alert in self.alert_store:
            stats['by_severity'][alert.severity] += 1
            stats['by_source'][alert.source] += 1
            if not alert.resolved:
                stats['unresolved'] += 1
            if not alert.acknowledged:
                stats['unacknowledged'] += 1
        
        return dict(stats)

# Example usage and testing functions
def create_demo_config() -> Dict[str, Any]:
    """Create demo configuration"""
    return {
        'deduplication_window': 30,
        'email': {
            'smtp_host': 'smtp.gmail.com',
            'smtp_port': 587,
            'username': 'your-email@gmail.com',
            'password': 'your-app-password',
            'recipients': ['admin@example.com']
        },
        'slack': {
            'webhook_url': 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK',
            'channel': '#alerts'
        },
        'webhooks': [{
            'url': 'https://your-monitoring-system.com/webhook',
            'headers': {'Authorization': 'Bearer your-token'}
        }]
    }

if __name__ == '__main__':
    # Demo usage
    config = create_demo_config()
    alert_system = RealTimeAlertSystem(config)
    
    # Send test alerts
    alert_system.send_alert(
        severity='critical',
        title='High Error Rate Detected',
        message='Agent error rate exceeded 50% in the last 5 minutes',
        source='error_tracker',
        metadata={'error_rate': 0.65, 'period': '5min'}
    )
    
    alert_system.send_alert(
        severity='warning',
        title='Resource Usage High',
        message='Memory usage is at 85%',
        source='resource_monitor',
        metadata={'memory_usage': 0.85, 'cpu_usage': 0.70}
    )
    
    # Get statistics
    stats = alert_system.get_alert_stats()
    print(f"Alert Statistics: {json.dumps(stats, indent=2)}")
