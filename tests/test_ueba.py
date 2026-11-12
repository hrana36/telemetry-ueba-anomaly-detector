"""
Unit and Integration Test Suite for UEBA Anomaly Detector
"""
import os
import sys

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import pytest
import numpy as np
from ueba.models import AuthEvent, EventStatus, AlertSeverity
from ueba.feature_extractor import BehavioralFeatureExtractor
from ueba.anomaly_model import DualEnsembleUEBAModel
from ueba.scoring_engine import UEBAScoringEngine


@pytest.fixture
def extractor():
    return BehavioralFeatureExtractor(business_hours=(8, 18))


@pytest.fixture
def sample_benign_events():
    return [
        AuthEvent(
            timestamp=f"2026-11-12T{h:02d}:10:00Z",
            user_id="alice@corp.internal",
            source_ip="10.0.1.50",
            device_id="mac-01",
            action="ACCESS_RESOURCE",
            resource="portal/dashboard",
            status=EventStatus.SUCCESS,
            bytes_transferred=5000,
            country="US",
        )
        for h in range(9, 17)
    ]


@pytest.fixture
def sample_compromised_events():
    return [
        AuthEvent(
            timestamp=f"2026-11-12T{h:02d}:30:00Z",
            user_id="mallory@corp.internal",
            source_ip=f"185.220.101.{idx}",
            device_id="mac-unknown",
            action="EXPORT_DATA",
            resource="s3://db-dump",
            status=EventStatus.SUCCESS,
            bytes_transferred=25000000,
            country="RU",
        )
        for idx, h in enumerate([1, 2, 3, 4])
    ]


def test_temporal_entropy(extractor):
    # Single hour concentrated: entropy should be 0.0
    single_hour = [10] * 20
    ent_zero = extractor.compute_temporal_entropy(single_hour)
    assert ent_zero == pytest.approx(0.0, abs=1e-4)

    # Uniform distribution across all 24 hours: entropy should be ~1.0
    uniform_hours = list(range(24)) * 5
    ent_uniform = extractor.compute_temporal_entropy(uniform_hours)
    assert ent_uniform == pytest.approx(1.0, abs=1e-2)


def test_feature_extractor_outputs(extractor, sample_benign_events, sample_compromised_events):
    benign_feats = extractor.extract_user_features(sample_benign_events)
    for k in extractor.FEATURE_NAMES:
        assert k in benign_feats

    assert benign_feats["off_hours_ratio"] == 0.0
    assert benign_feats["auth_failure_rate"] == 0.0
    assert benign_feats["privilege_action_rate"] == 0.0

    threat_feats = extractor.extract_user_features(sample_compromised_events)
    assert threat_feats["off_hours_ratio"] == 1.0
    assert threat_feats["log_egress_bytes"] > 7.0  # 25MB is ~7.4 in log10
    assert threat_feats["privilege_action_rate"] == 1.0


def test_dual_ensemble_fit_and_anomaly_separation():
    np.random.seed(42)
    # 20 normal samples centered at 0 with small variance
    X_train = np.random.normal(loc=0.0, scale=0.2, size=(20, 7))

    model = DualEnsembleUEBAModel(n_estimators=50, random_state=42)
    model.fit(X_train)

    assert model.is_fitted

    normal_point = np.random.normal(loc=0.0, scale=0.1, size=(7,))
    anomaly_point = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    comp_norm, _, _ = model.predict_score(normal_point)
    comp_anom, _, _ = model.predict_score(anomaly_point)

    assert comp_anom > comp_norm
    assert comp_anom > 0.60


def test_unfitted_model_raises():
    model = DualEnsembleUEBAModel()
    with pytest.raises(RuntimeError):
        model.predict_score(np.zeros(7))


def test_scoring_engine_mitre_mapping(extractor, sample_compromised_events):
    engine = UEBAScoringEngine(alert_threshold=0.50)
    # Train on 10 benign users with normal activity
    train_events = []
    for u in range(10):
        user_email = f"user_{u}@corp.internal"
        for h in range(9, 17):
            train_events.append(
                AuthEvent(
                    timestamp=f"2026-11-12T{h:02d}:10:00Z",
                    user_id=user_email,
                    source_ip=f"10.0.1.{10 + u}",
                    device_id=f"mac-{u:02d}",
                    action="ACCESS_RESOURCE",
                    resource="portal/dashboard",
                    status=EventStatus.SUCCESS,
                    bytes_transferred=4000 + u * 200,
                    country="US",
                )
            )

    engine.train_on_events(train_events)

    # Test benign user
    benign_eval = [
        AuthEvent(
            timestamp=f"2026-11-12T{h:02d}:15:00Z",
            user_id="user_0@corp.internal",
            source_ip="10.0.1.10",
            device_id="mac-00",
            action="ACCESS_RESOURCE",
            resource="portal/dashboard",
            status=EventStatus.SUCCESS,
            bytes_transferred=4200,
            country="US",
        )
        for h in range(10, 16)
    ]

    benign_score = engine.evaluate_user_window("user_0@corp.internal", benign_eval)
    assert not benign_score.is_anomaly
    assert benign_score.severity in (AlertSeverity.LOW, AlertSeverity.MEDIUM)

    # Test compromised user
    threat_score = engine.evaluate_user_window("mallory@corp.internal", sample_compromised_events)
    assert threat_score.is_anomaly
    assert threat_score.severity in (AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL)
    assert any("Exfiltration" in t for t in threat_score.mitre_tactics)
    assert any("T1530" in t for t in threat_score.mitre_techniques)
