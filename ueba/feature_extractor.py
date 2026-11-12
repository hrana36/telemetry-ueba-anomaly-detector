"""
Behavioral Feature Extraction for Enterprise Telemetry
Computes statistical, temporal, and information-theoretic behavioral features per user.
"""
import math
from datetime import datetime
from collections import Counter
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from ueba.models import AuthEvent, EventStatus


class BehavioralFeatureExtractor:
    """
    Extracts high-dimensional statistical and behavioral representations from
    raw authentication and access telemetry streams.
    """

    FEATURE_NAMES = [
        "temporal_entropy",       # Normalized Shannon entropy of diurnal activity
        "off_hours_ratio",        # Fraction of actions outside 08:00-18:00
        "ip_diversity_ratio",     # Cardinality of unique IP addresses per event
        "auth_failure_rate",      # Proportion of failed authentication/access attempts
        "log_egress_bytes",       # Log-scaled mean volume of data transferred
        "privilege_action_rate",  # Frequency of high-risk / privilege escalation operations
        "geo_dispersion_count",   # Distinct geographic origins count
    ]

    SENSITIVE_ACTIONS = {
        "ESCALATE_ROLE",
        "EXPORT_DATA",
        "MODIFY_CREDENTIALS",
        "GRANT_PERMISSION",
        "ASSUME_ROLE",
        "ACCESS_SECRET",
    }

    def __init__(self, business_hours: Tuple[int, int] = (8, 18)):
        self.business_start, self.business_end = business_hours

    def _parse_iso(self, ts_str: str) -> datetime:
        # Standard ISO parser handling UTC Z or offset
        ts_clean = ts_str.replace("Z", "+00:00")
        return datetime.fromisoformat(ts_clean)

    def compute_temporal_entropy(self, hours: List[int]) -> float:
        """
        Computes normalized Shannon entropy of hourly distribution:
        H = - sum(p_i * log2(p_i)) / log2(24)
        """
        if not hours:
            return 0.0
        n = len(hours)
        counts = Counter(hours)
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(24)
        return float(np.clip(entropy / max_entropy, 0.0, 1.0))

    def extract_user_features(self, events: List[AuthEvent]) -> Dict[str, float]:
        """
        Aggregates a window of events for a single identity into a numeric feature vector.
        """
        if not events:
            return {name: 0.0 for name in self.FEATURE_NAMES}

        n_events = len(events)
        hours = []
        off_hours_count = 0
        ips = set()
        countries = set()
        failures = 0
        total_bytes = 0
        sensitive_count = 0

        for ev in events:
            try:
                dt = self._parse_iso(ev.timestamp)
                hour = dt.hour
                hours.append(hour)
                if hour < self.business_start or hour >= self.business_end:
                    off_hours_count += 1
            except Exception:
                pass

            ips.add(ev.source_ip)
            countries.add(ev.country)
            if ev.status == EventStatus.FAILURE or ev.status == "FAILURE":
                failures += 1
            total_bytes += max(0, ev.bytes_transferred)
            if ev.action.upper() in self.SENSITIVE_ACTIONS or "ADMIN" in ev.action.upper():
                sensitive_count += 1

        temporal_entropy = self.compute_temporal_entropy(hours)
        off_hours_ratio = off_hours_count / n_events
        ip_diversity_ratio = len(ips) / n_events
        auth_failure_rate = failures / n_events
        mean_bytes = total_bytes / n_events
        log_egress_bytes = float(np.log10(1.0 + mean_bytes))
        privilege_action_rate = sensitive_count / n_events
        geo_dispersion_count = float(len(countries))

        return {
            "temporal_entropy": round(temporal_entropy, 4),
            "off_hours_ratio": round(off_hours_ratio, 4),
            "ip_diversity_ratio": round(ip_diversity_ratio, 4),
            "auth_failure_rate": round(auth_failure_rate, 4),
            "log_egress_bytes": round(log_egress_bytes, 4),
            "privilege_action_rate": round(privilege_action_rate, 4),
            "geo_dispersion_count": round(geo_dispersion_count, 4),
        }

    def extract_from_dataframe(self, df: pd.DataFrame, window_col: str = "user_id") -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Processes a tabular event log into an aggregated feature matrix per entity.
        Returns:
            features_df: DataFrame with user_id and feature columns.
            X: 2D numpy array [n_users, n_features]
        """
        rows = []
        for user_id, group in df.groupby(window_col):
            events = []
            for _, r in group.iterrows():
                status = EventStatus.SUCCESS if str(r.get("status", "SUCCESS")).upper() == "SUCCESS" else EventStatus.FAILURE
                events.append(
                    AuthEvent(
                        timestamp=str(r.get("timestamp", "2026-01-01T12:00:00Z")),
                        user_id=str(user_id),
                        source_ip=str(r.get("source_ip", "10.0.0.1")),
                        device_id=str(r.get("device_id", "dev-01")),
                        action=str(r.get("action", "LOGIN")),
                        resource=str(r.get("resource", "portal")),
                        status=status,
                        bytes_transferred=int(r.get("bytes_transferred", 0)),
                        country=str(r.get("country", "US")),
                    )
                )
            feats = self.extract_user_features(events)
            feats["user_id"] = user_id
            if "is_compromised" in group.columns:
                feats["is_compromised"] = int(group["is_compromised"].any())
            rows.append(feats)

        features_df = pd.DataFrame(rows)
        X = features_df[self.FEATURE_NAMES].to_numpy(dtype=np.float64)
        return features_df, X
