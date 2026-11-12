"""
End-to-End UEBA Detection Example: Detecting an Insider Data Exfiltration Campaign
Demonstrates feature extraction, dual-ensemble scoring, and MITRE ATT&CK mapping.
"""
import os
import sys

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

import pandas as pd
from ueba.scoring_engine import UEBAScoringEngine
from ueba.models import AuthEvent, EventStatus


def main():
    print("=" * 80)
    print("  UEBA TELEMETRY ANOMALY DETECTOR - INSIDER THREAT TRIAGE DEMO")
    print("=" * 80)

    dataset_path = os.path.join(repo_dir, "datasets", "enterprise_auth_sample.csv")
    df = pd.read_csv(dataset_path)

    engine = UEBAScoringEngine(alert_threshold=0.65)

    print(f"[*] Ingesting enterprise training baseline from {len(df)} telemetry events...")
    engine.train_on_dataframe(df[df["is_compromised"] == 0])
    print("[+] Dual Ensemble baseline established successfully.")

    # 1. Evaluate normal user
    benign_user = "emp_003@corp.internal"
    benign_events = [
        AuthEvent(
            timestamp=f"2026-11-12T{h:02d}:15:00Z",
            user_id=benign_user,
            source_ip="10.0.1.25",
            device_id="mac-0003",
            action="ACCESS_RESOURCE",
            resource="sharepoint/docs",
            status=EventStatus.SUCCESS,
            bytes_transferred=12000,
            country="US",
        )
        for h in range(9, 17)
    ]
    res_benign = engine.evaluate_user_window(benign_user, benign_events)
    print(f"\n[1] Benign User Profile: {benign_user}")
    print(f"    Composite Anomaly Score : {res_benign.composite_score:.4f} (Severity: {res_benign.severity.value})")
    print(f"    Alert Triggered         : {res_benign.is_anomaly}")
    print(f"    Triage Explanation      : {res_benign.explanation}")

    # 2. Evaluate compromised insider threat user
    threat_user = "alice_stolen_creds@corp.internal"
    threat_events = [
        AuthEvent(
            timestamp=f"2026-11-12T{h:02d}:30:00Z",
            user_id=threat_user,
            source_ip=ip,
            device_id="mac-rogue-99",
            action="EXPORT_DATA",
            resource="s3://financial-records-db",
            status=EventStatus.SUCCESS,
            bytes_transferred=45000000,
            country=country,
        )
        for h, ip, country in [
            (2, "185.220.101.5", "NL"),
            (3, "45.134.22.19", "DE"),
            (3, "91.240.118.8", "RU"),
            (4, "185.220.101.5", "NL"),
        ]
    ]

    res_threat = engine.evaluate_user_window(threat_user, threat_events)
    print(f"\n[2] High-Risk Alert: {threat_user}")
    print(f"    Composite Anomaly Score : {res_threat.composite_score:.4f} (Severity: {res_threat.severity.value})")
    print(f"    Alert Triggered         : {res_threat.is_anomaly}")
    print(f"    Isolation Forest Score  : {res_threat.isolation_score:.4f}")
    print(f"    PCA Reconstruction Error: {res_threat.reconstruction_error:.4f}")
    print(f"    MITRE ATT&CK Tactics    : {', '.join(res_threat.mitre_tactics)}")
    print(f"    MITRE ATT&CK Techniques : {', '.join(res_threat.mitre_techniques)}")
    print(f"    Top Feature Deviations  :")
    for feat in res_threat.contributing_features[:3]:
        print(f"      - {feat['feature']:<22} | Observed: {feat['observed_value']:<8} | Baseline: {feat['baseline_mean']:<8} | Z-Score: {feat['z_score']}")
    print(f"    Incident Explanation    : {res_threat.explanation}")
    print("=" * 80)


if __name__ == "__main__":
    main()
