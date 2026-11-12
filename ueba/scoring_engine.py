"""
UEBA Scoring and Alerting Engine with MITRE ATT&CK Mapping
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from ueba.models import AuthEvent, AnomalyScore, AlertSeverity
from ueba.feature_extractor import BehavioralFeatureExtractor
from ueba.anomaly_model import DualEnsembleUEBAModel


class UEBAScoringEngine:
    """
    End-to-end security analytics engine for ingesting telemetry, evaluating behavioral models,
    and generating MITRE ATT&CK contextualized incident alerts.
    """

    def __init__(
        self,
        alert_threshold: float = 0.65,
        alpha_ensemble_weight: float = 0.5,
        business_hours: Tuple[int, int] = (8, 18),
    ):
        self.alert_threshold = alert_threshold
        self.extractor = BehavioralFeatureExtractor(business_hours=business_hours)
        self.model = DualEnsembleUEBAModel(alpha_ensemble_weight=alpha_ensemble_weight)
        self.is_trained = False

    def train_on_dataframe(self, df: pd.DataFrame, window_col: str = "user_id") -> "UEBAScoringEngine":
        """
        Trains baseline distributions from a DataFrame of normal enterprise activity.
        """
        _, X = self.extractor.extract_from_dataframe(df, window_col=window_col)
        self.model.fit(X)
        self.is_trained = True
        return self

    def train_on_events(self, events: List[AuthEvent]) -> "UEBAScoringEngine":
        """
        Trains baseline distributions from a list of AuthEvents.
        """
        users_map: Dict[str, List[AuthEvent]] = {}
        for ev in events:
            users_map.setdefault(ev.user_id, []).append(ev)

        feat_vectors = []
        for user_id, u_events in users_map.items():
            feats = self.extractor.extract_user_features(u_events)
            vec = [feats[name] for name in self.extractor.FEATURE_NAMES]
            feat_vectors.append(vec)

        X = np.array(feat_vectors, dtype=np.float64)
        self.model.fit(X)
        self.is_trained = True
        return self

    def _map_mitre_attack(self, feats: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        Maps aberrant telemetry features to MITRE ATT&CK Tactics and Techniques.
        """
        tactics = []
        techniques = []

        if feats.get("auth_failure_rate", 0.0) >= 0.25:
            tactics.append("Credential Access (TA0006)")
            techniques.append("T1110 - Brute Force / Password Spraying")

        if feats.get("off_hours_ratio", 0.0) >= 0.5 or feats.get("temporal_entropy", 0.0) >= 0.8:
            tactics.append("Defense Evasion (TA0005)")
            techniques.append("T1078 - Valid Accounts (Compromised Diurnal Pattern)")

        if feats.get("ip_diversity_ratio", 0.0) >= 0.35 or feats.get("geo_dispersion_count", 0.0) >= 2.0:
            tactics.append("Initial Access (TA0001)")
            techniques.append("T1078.004 - Cloud Administration / Geo-dispersed Access")

        if feats.get("privilege_action_rate", 0.0) >= 0.15:
            tactics.append("Privilege Escalation (TA0004)")
            techniques.append("T1098 - Account Manipulation & Role Assignment")

        if feats.get("log_egress_bytes", 0.0) >= 4.5:
            tactics.append("Exfiltration (TA0010)")
            techniques.append("T1530 / T1048 - Exfiltration of Cloud Data / Archive Staging")

        return list(dict.fromkeys(tactics)), list(dict.fromkeys(techniques))

    def _calculate_severity(self, composite_score: float) -> AlertSeverity:
        if composite_score >= 0.85:
            return AlertSeverity.CRITICAL
        elif composite_score >= 0.70:
            return AlertSeverity.HIGH
        elif composite_score >= 0.50:
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

    def evaluate_user_window(self, user_id: str, events: List[AuthEvent]) -> AnomalyScore:
        """
        Extracts features for a user window and computes anomaly score and MITRE attribution.
        """
        if not self.is_trained:
            raise RuntimeError("Scoring engine must be trained on baseline telemetry before evaluation.")

        feats = self.extractor.extract_user_features(events)
        x_vec = np.array([feats[name] for name in self.extractor.FEATURE_NAMES], dtype=np.float64)

        composite, iso_score, pca_error = self.model.predict_score(x_vec)
        is_anomaly = composite >= self.alert_threshold
        severity = self._calculate_severity(composite)

        attributions = self.model.explain_vector(x_vec, self.extractor.FEATURE_NAMES)
        tactics, techniques = self._map_mitre_attack(feats)

        latest_ts = events[-1].timestamp if events else "2026-11-12T00:00:00Z"

        # Generate summary explanation
        top_reasons = [f"{a['feature']} (z={a['z_score']})" for a in attributions[:2] if a['z_score'] > 1.5]
        if is_anomaly:
            explanation = f"Anomaly detected for {user_id} (Score: {composite:.2f}). Key drivers: {', '.join(top_reasons) if top_reasons else 'multivariate deviation'}."
        else:
            explanation = f"User {user_id} activity matches baseline distribution (Score: {composite:.2f})."

        return AnomalyScore(
            user_id=user_id,
            timestamp=latest_ts,
            composite_score=composite,
            isolation_score=iso_score,
            reconstruction_error=pca_error,
            is_anomaly=is_anomaly,
            severity=severity,
            contributing_features=attributions,
            mitre_tactics=tactics,
            mitre_techniques=techniques,
            explanation=explanation,
        )
