"""
Synthetic Enterprise Telemetry Generator
Generates realistic authentication and access logs with baseline diurnal users
and injected adversary tactics (MITRE ATT&CK).
"""
import csv
import random
from datetime import datetime, timedelta

USERS_BENIGN = [f"emp_{i:03d}@corp.internal" for i in range(1, 26)]
ACTIONS_BENIGN = ["LOGIN", "ACCESS_RESOURCE", "QUERY_DATABASE", "READ_DOCUMENT", "REFRESH_TOKEN"]
RESOURCES_BENIGN = ["sharepoint/hr", "jira/sprint", "slack/general", "salesforce/leads", "gitlab/frontend"]

COMPROMISED_SCENARIOS = {
    "alice_stolen_creds@corp.internal": {
        "tactics": ["Exfiltration", "Defense Evasion"],
        "off_hours": True,
        "high_bytes": True,
        "countries": ["NL", "DE", "RU"],
        "ips": ["185.220.101.5", "45.134.22.19", "91.240.118.8"],
        "action": "EXPORT_DATA",
    },
    "bob_privilege_escalation@corp.internal": {
        "tactics": ["Privilege Escalation"],
        "off_hours": False,
        "high_bytes": False,
        "actions": ["ESCALATE_ROLE", "MODIFY_CREDENTIALS", "GRANT_PERMISSION"],
        "ips": ["10.0.1.45"],
        "countries": ["US"],
    },
    "charlie_credential_spraying@corp.internal": {
        "tactics": ["Credential Access"],
        "off_hours": True,
        "high_failures": True,
        "ips": ["198.51.100.12", "198.51.100.14", "198.51.100.16", "198.51.100.20"],
        "countries": ["US", "CA"],
    },
    "dave_impossible_travel@corp.internal": {
        "tactics": ["Initial Access"],
        "off_hours": False,
        "countries": ["US", "JP", "BR", "SG", "GB"],
        "ips": ["104.28.14.2", "118.27.3.1", "177.18.9.22", "165.225.112.5"],
    },
    "eve_insider_beaconing@corp.internal": {
        "tactics": ["Exfiltration"],
        "off_hours": True,
        "high_entropy": True,
        "action": "EXPORT_DATA",
        "countries": ["US"],
        "ips": ["10.0.4.88"],
    },
}


def generate_enterprise_logs(output_file: str, days: int = 7):
    random.seed(42)
    start_date = datetime(2026, 11, 1, 8, 0, 0)
    events = []

    # 1. Generate Benign Users Events (25 users)
    for user in USERS_BENIGN:
        dept_ip = f"10.0.{random.randint(1, 5)}.{random.randint(10, 240)}"
        home_vpn = f"192.168.10.{random.randint(2, 250)}"
        device_id = f"mac-{hash(user) % 10000:04d}"

        # 40-60 events per user over the week
        num_events = random.randint(35, 55)
        for _ in range(num_events):
            day_offset = random.randint(0, days - 1)
            # Diurnal hours: 8 to 18 (with 90% probability)
            if random.random() < 0.90:
                hour = random.randint(8, 17)
            else:
                hour = random.choice([7, 18, 19])
            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = start_date + timedelta(days=day_offset, hours=hour - 8, minutes=minute, seconds=second)

            action = random.choice(ACTIONS_BENIGN)
            resource = random.choice(RESOURCES_BENIGN)
            source_ip = dept_ip if random.random() < 0.75 else home_vpn
            status = "SUCCESS" if random.random() < 0.96 else "FAILURE"
            bytes_transferred = random.randint(150, 45000)

            events.append({
                "timestamp": ts.isoformat() + "Z",
                "user_id": user,
                "source_ip": source_ip,
                "device_id": device_id,
                "action": action,
                "resource": resource,
                "status": status,
                "bytes_transferred": bytes_transferred,
                "country": "US",
                "is_compromised": 0,
            })

    # 2. Inject Anomaly Campaigns
    for user, spec in COMPROMISED_SCENARIOS.items():
        dev_id = f"mac-comp-{hash(user) % 1000:03d}"
        num_events = 30

        for idx in range(num_events):
            day_offset = random.randint(3, days - 1)
            if spec.get("off_hours", False):
                hour = random.choice([1, 2, 3, 4, 22, 23])
            elif spec.get("high_entropy", False):
                hour = random.randint(0, 23)
            else:
                hour = random.randint(9, 16)

            minute = random.randint(0, 59)
            second = random.randint(0, 59)
            ts = start_date + timedelta(days=day_offset, hours=hour - 8, minutes=minute, seconds=second)

            source_ip = random.choice(spec.get("ips", ["198.51.100.99"]))
            country = random.choice(spec.get("countries", ["US"]))

            if spec.get("high_failures", False) and idx < 18:
                status = "FAILURE"
                action = "LOGIN"
                resource = "auth/saml/sso"
                bytes_xfer = 240
            elif "actions" in spec:
                status = "SUCCESS"
                action = random.choice(spec["actions"])
                resource = "iam/admin_roles"
                bytes_xfer = 1200
            elif spec.get("high_bytes", False):
                status = "SUCCESS"
                action = spec.get("action", "EXPORT_DATA")
                resource = "s3://corp-financial-backups"
                bytes_xfer = random.randint(15000000, 85000000)  # 15MB - 85MB
            else:
                status = "SUCCESS"
                action = spec.get("action", "ACCESS_RESOURCE")
                resource = "confidential/payroll"
                bytes_xfer = random.randint(500000, 2000000)

            events.append({
                "timestamp": ts.isoformat() + "Z",
                "user_id": user,
                "source_ip": source_ip,
                "device_id": dev_id,
                "action": action,
                "resource": resource,
                "status": status,
                "bytes_transferred": bytes_xfer,
                "country": country,
                "is_compromised": 1,
            })

    # Sort chronologically
    events.sort(key=lambda x: x["timestamp"])

    # Write CSV
    fieldnames = [
        "timestamp", "user_id", "source_ip", "device_id", "action",
        "resource", "status", "bytes_transferred", "country", "is_compromised"
    ]
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)

    print(f"Generated {len(events)} telemetry events across {len(USERS_BENIGN) + len(COMPROMISED_SCENARIOS)} identities -> {output_file}")


if __name__ == "__main__":
    import os
    target_path = os.path.join(os.path.dirname(__file__), "enterprise_auth_sample.csv")
    generate_enterprise_logs(target_path)
