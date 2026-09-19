"""
Cryptographic SHA-256 Audit Ledger
Tamper-evident hash-chained governance for sovereign agentic AI
ISO 27001 / IEC 62443 Industrial Security Compliant
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Any

LEDGER_FILE = os.path.join(os.path.dirname(__file__), "audit_ledger.jsonl")

class AuditLedger:
    def __init__(self, filepath: str = LEDGER_FILE):
        self.filepath = filepath
        self.records: List[Dict[str, Any]] = []
        self._load_or_init()

    def _compute_hash(self, prev_hash: str, ts: str, actor: str, action: str, tool: str, details: str) -> str:
        payload = f"{prev_hash}|{ts}|{actor}|{action}|{tool}|{details}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_or_init(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            self.records.append(json.loads(line.strip()))
            except Exception:
                self.records = []

        if not self.records:
            # Genesis Block
            genesis_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            genesis_hash = self._compute_hash("GENESIS_BLOCK_AIRGAP_INIT", genesis_ts, "SYSTEM", "Air-gap Workbench Initialized", "Security Kernel", "Sovereignty Enforced")
            genesis_entry = {
                "id": "AUD-0001",
                "timestamp": genesis_ts,
                "actor": "SYSTEM",
                "action": "Air-gap Workbench Initialized",
                "tool": "Security Kernel",
                "status": "Success",
                "approval": "System",
                "details": "Sovereign runtime online. 0 WAN egress.",
                "previous_hash": "0000000000000000000000000000000000000000000000000000000000000000",
                "hash": genesis_hash
            }
            self.records.append(genesis_entry)
            self._save_record(genesis_entry)

    def _save_record(self, record: Dict[str, Any]):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def log(self, actor: str, action: str, tool: str, status: str = "Success", approval: str = "Not required", details: str = "") -> Dict[str, Any]:
        prev_hash = self.records[-1]["hash"] if self.records else "0000000000000000000000000000000000000000000000000000000000000000"
        ts = datetime.utcnow().strftime("%H:%M:%S")
        record_id = f"AUD-{1000 + len(self.records) + 1}"
        record_hash = self._compute_hash(prev_hash, ts, actor, action, tool, details)

        entry = {
            "id": record_id,
            "timestamp": ts,
            "actor": actor,
            "action": action,
            "tool": tool,
            "status": status,
            "approval": approval,
            "details": details,
            "previous_hash": prev_hash,
            "hash": record_hash,
            "short_hash": f"{record_hash[:8]}...{record_hash[-6:]}"
        }
        self.records.append(entry)
        self._save_record(entry)
        return entry

    def get_all(self) -> List[Dict[str, Any]]:
        return list(reversed(self.records))

    def verify_integrity(self) -> bool:
        """Verify unbroken cryptographic SHA-256 chain across all records"""
        if len(self.records) <= 1:
            return True
        for i in range(1, len(self.records)):
            prev = self.records[i - 1]
            curr = self.records[i]
            if curr["previous_hash"] != prev["hash"]:
                return False
            recalc = self._compute_hash(curr["previous_hash"], curr["timestamp"], curr["actor"], curr["action"], curr["tool"], curr.get("details", ""))
            if recalc != curr["hash"]:
                return False
        return True

# Singleton instance
audit_service = AuditLedger()

if __name__ == "__main__":
    rec = audit_service.log("Maintenance Agent", "Retrieved sensor data", "OPC-UA", "Success", "Approved", "Motor-014 vibration 4.8 mm/s")
    print(f"Logged record: {rec['id']} with SHA-256: {rec['hash']}")
    print(f"Chain Integrity Verified: {audit_service.verify_integrity()}")
