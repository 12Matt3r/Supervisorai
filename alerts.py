"""
Real-Time Alert System for Supervisor Agent
Handles critical deviation detection and multi-channel notifications
"""

import json
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
import requests
import hashlib


class AlertSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    TASK_TIMEOUT = "task_timeout"
    LOW_CONFIDENCE = "low_confidence"
    ERROR_RATE_HIGH = "error_rate_high"
    SYSTEM_FAILURE = "system_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PATTERN_ANOMALY = "pattern_anomaly"


@dataclass
class Alert:
    id: str
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    agent_id: Optional[str]
    metadata: Dict[str, Any]
    created_at: datetime
    resolved_at: Optional[datetime] = None
    notification_sent: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'agent_id': self.agent_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'notification_sent': self.notification_sent
        }


class AlertManager:
    """Manages real-time alerts with multi-channel notifications"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        
        # Deduplication
        self.alert_hashes: Dict[str, datetime] = {}
        self.dedup_window = timedelta(minutes=5)
        
        # Thresholds
        self.task_timeout_threshold = config.get('task_timeout_threshold', 300)
        self.low_confidence_threshold = config.get('low_confidence_threshold', 0.3)
        self.error_rate_threshold = config.get('error_rate_threshold', 0.1)
        
        # Notification channels
        self.email_config = config.get('email', {})
        self.slack_config = config.get('slack', {})
        self.webhook_config = config.get('webhook', {})
        
    def create_alert(self, alert_type: AlertType, severity: AlertSeverity,
                    title: str, message: str, agent_id: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Optional[Alert]:
        """Create a new alert with deduplication"""
        
        # Generate alert hash for deduplication
        alert_hash = self._generate_alert_hash(alert_type, agent_id, title)
        
        # Check for recent duplicate
        if self._is_duplicate_alert(alert_hash):
            self.logger.debug(f"Skipping duplicate alert: {title}")
            return None
            
        # Create alert
        alert_id = f"{alert_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        alert = Alert(
            id=alert_id,
            alert_type=alert_type,
            severity=severity,
            title=title,
            message=message,
            agent_id=agent_id,
            metadata=metadata or {},
            created_at=datetime.now()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_hashes[alert_hash] = datetime.now()
        
        self.logger.info(f"Created {severity.value} alert: {title}")
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert
        
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert"""
        if alert_id not in self.active_alerts:
            return False
            
        alert = self.active_alerts.pop(alert_id)
        alert.resolved_at = datetime.now()
        self.resolved_alerts.append(alert)
        
        self.logger.info(f"Resolved alert: {alert.title}")
        return True
        
    def evaluate_conditions(self, data: Dict[str, Any], agent_id: str, context: str):
        """Evaluate alert conditions based on incoming data"""
        
        # Task timeout check
        if 'task_duration' in data and data['task_duration'] > self.task_timeout_threshold:
            self.create_alert(
                AlertType.TASK_TIMEOUT,
                AlertSeverity.HIGH,
                f"Task Timeout Detected",
                f"Task exceeded {self.task_timeout_threshold}s threshold with {data['task_duration']}s duration",
                agent_id,
                {'task_duration': data['task_duration'], 'context': context}
            )
            
        # Low confidence check
        if 'confidence' in data and data['confidence'] < self.low_confidence_threshold:
            self.create_alert(
                AlertType.LOW_CONFIDENCE,
                AlertSeverity.MEDIUM,
                f"Low Confidence Score",
                f"Task confidence ({data['confidence']:.2f}) below threshold ({self.low_confidence_threshold})",
                agent_id,
                {'confidence': data['confidence'], 'context': context}
            )
            
        # Error rate check
        if 'error_rate' in data and data['error_rate'] > self.error_rate_threshold:
            self.create_alert(
                AlertType.ERROR_RATE_HIGH,
                AlertSeverity.HIGH,
                f"High Error Rate",
                f"Error rate ({data['error_rate']:.2%}) exceeds threshold ({self.error_rate_threshold:.2%})",
                agent_id,
                {'error_rate': data['error_rate'], 'context': context}
            )
            
    def _generate_alert_hash(self, alert_type: AlertType, agent_id: Optional[str], title: str) -> str:
        """Generate hash for alert deduplication"""
        content = f"{alert_type.value}:{agent_id or 'system'}:{title}"
        return hashlib.md5(content.encode()).hexdigest()
        
    def _is_duplicate_alert(self, alert_hash: str) -> bool:
        """Check if alert is a recent duplicate"""
        if alert_hash not in self.alert_hashes:
            return False
            
        last_occurrence = self.alert_hashes[alert_hash]
        return datetime.now() - last_occurrence < self.dedup_window
        
    def _send_notifications(self, alert: Alert):
        """Send notifications via configured channels"""
        try:
            # Email notification
            if self.email_config.get('enabled', False):
                self._send_email_notification(alert)
                
            # Slack notification
            if self.slack_config.get('enabled', False):
                self._send_slack_notification(alert)
                
            # Webhook notification
            if self.webhook_config.get('enabled', False):
                self._send_webhook_notification(alert)
                
            alert.notification_sent = True
            
        except Exception as e:
            self.logger.error(f"Failed to send notifications for alert {alert.id}: {e}")
            
    def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['from_address']
            msg['To'] = ', '.join(self.email_config['to_addresses'])
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
            Alert Details:
            - Type: {alert.alert_type.value}
            - Severity: {alert.severity.value}
            - Agent: {alert.agent_id or 'System'}
            - Time: {alert.created_at}
            - Message: {alert.message}
            
            Metadata: {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_host'], self.email_config['smtp_port'])
            if self.email_config.get('use_tls', True):
                server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")
            
    def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        try:
            color_map = {
                AlertSeverity.LOW: "#36a64f",
                AlertSeverity.MEDIUM: "#ff9500", 
                AlertSeverity.HIGH: "#ff0000",
                AlertSeverity.CRITICAL: "#8B0000"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.severity, "#36a64f"),
                    "title": f"[{alert.severity.value.upper()}] {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Type", "value": alert.alert_type.value, "short": True},
                        {"title": "Agent", "value": alert.agent_id or "System", "short": True},
                        {"title": "Time", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ],
                    "ts": alert.created_at.timestamp()
                }]
            }
            
            response = requests.post(self.slack_config['webhook_url'], json=payload)
            response.raise_for_status()
            
            self.logger.info(f"Slack notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack notification: {e}")
            
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification"""
        try:
            headers = self.webhook_config.get('headers', {})
            payload = alert.to_dict()
            
            response = requests.post(
                self.webhook_config['url'],
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            
            self.logger.info(f"Webhook notification sent for alert {alert.id}")
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook notification: {e}")
            
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
        
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level"""
        return [alert for alert in self.active_alerts.values() if alert.severity == severity]
        
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        active = self.get_active_alerts()
        
        return {
            'total_active': len(active),
            'by_severity': {
                'critical': len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
                'high': len([a for a in active if a.severity == AlertSeverity.HIGH]),
                'medium': len([a for a in active if a.severity == AlertSeverity.MEDIUM]),
                'low': len([a for a in active if a.severity == AlertSeverity.LOW])
            },
            'by_type': {}
        }
        
    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Clean up old resolved alerts and hash entries"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean resolved alerts
        self.resolved_alerts = [a for a in self.resolved_alerts if a.resolved_at and a.resolved_at > cutoff_time]
        
        # Clean hash entries
        self.alert_hashes = {h: t for h, t in self.alert_hashes.items() if t > cutoff_time}
        
        self.logger.debug(f"Cleaned up alerts older than {max_age_hours} hours")
