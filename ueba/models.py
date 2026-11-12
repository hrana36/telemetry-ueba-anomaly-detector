"""
UEBA Telemetry Models and Data Contracts
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum


class EventStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class AuthEvent:
    timestamp: str  # ISO 8601 format e.g. "2026-11-12T03:45:12Z"
    user_id: str
    source_ip: str
    device_id: str
    action: str  # LOGIN, ACCESS_RESOURCE, ESCALATE_ROLE, EXPORT_DATA, MODIFY_CREDENTIALS
    resource: str
    status: EventStatus
    bytes_transferred: int = 0
    country: str = "US"
    is_compromised: Optional[bool] = None  # Ground truth label for benchmarking


@dataclass
class AnomalyScore:
    user_id: str
    timestamp: str
    composite_score: float  # Normalized [0.0, 1.0]
    isolation_score: float
    reconstruction_error: float
    is_anomaly: bool
    severity: AlertSeverity
    contributing_features: List[Dict[str, Any]] = field(default_factory=list)
    mitre_tactics: List[str] = field(default_factory=list)
    mitre_techniques: List[str] = field(default_factory=list)
    explanation: str = ""
