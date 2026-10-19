"""
Base Audit Module — Foundation for all audit checks.

Defines the common Finding schema and base class that all audit modules inherit.
"""

import uuid
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.config import Severity, FindingStatus, AuditCategory


@dataclass
class AuditFinding:
    """
    A single audit finding with evidence.

    Every finding must have:
    - A unique ID
    - Category and severity
    - Status (confirmed, suspected, informational, inconclusive)
    - Evidence supporting the finding
    - A clear explanation
    - Actionable recommendation
    """
    finding_id: str = ""
    category: str = ""
    severity: str = Severity.INFO
    status: str = FindingStatus.INFORMATIONAL
    description: str = ""
    evidence: dict = field(default_factory=dict)
    affected_columns: list = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    recommendation: str = ""
    auto_correctable: bool = False
    resolution_status: str = "open"

    def to_dict(self) -> dict:
        d = asdict(self)
        # Ensure evidence is serializable
        d["evidence"] = json.loads(json.dumps(d["evidence"], default=str))
        return d

    def to_db_dict(self, audit_run_id: str) -> dict:
        """Convert to dict suitable for database insertion."""
        return {
            "audit_run_id": audit_run_id,
            "finding_id": self.finding_id,
            "category": self.category,
            "severity": self.severity,
            "status": self.status,
            "description": self.description,
            "evidence": json.dumps(self.evidence, default=str),
            "affected_columns": json.dumps(self.affected_columns),
            "explanation": self.explanation,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
            "auto_correctable": self.auto_correctable,
            "resolution_status": self.resolution_status,
        }


class BaseAudit:
    """
    Base class for all audit modules.

    Each audit module:
    - Accepts data and experiment context
    - Runs specific checks
    - Returns a list of AuditFindings
    - Never fabricates findings
    """

    # Override in subclasses
    CATEGORY = ""
    FINDING_PREFIX = ""

    def __init__(self):
        self._finding_counter = 0

    def run(self, **kwargs) -> list[AuditFinding]:
        """
        Run the audit checks.

        Override in subclasses. Must return a list of AuditFindings.
        """
        raise NotImplementedError

    def _next_finding_id(self) -> str:
        """Generate the next finding ID."""
        self._finding_counter += 1
        return f"{self.FINDING_PREFIX}-{self._finding_counter:03d}"

    def _create_finding(self, **kwargs) -> AuditFinding:
        """Create a finding with auto-generated ID and category."""
        return AuditFinding(
            finding_id=self._next_finding_id(),
            category=self.CATEGORY,
            **kwargs
        )
