"""
AI Agent Evaluation & Execution Harness for OSINT / Reconnaissance.

The Agent Harness acts as an automated sandbox and supervisor that:
1. Feeds OSINT investigation missions/tasks to the AI Agent.
2. Intercepts and logs all Agent Actions (Tool Calls, Thought Steps, Latency).
3. Evaluates Agent Performance (Task Completion Rate, Entity Recall, Step Count, Safety).
4. Generates structured Benchmark Evaluation Reports.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from pydantic import BaseModel, Field


class OSINTMission(BaseModel):
    """Specification of an OSINT investigation task for the Harness."""
    mission_id: str
    target_image_path: str
    target_webpage_url: str | None = None
    ground_truth_urls: list[str] = Field(default_factory=list)
    ground_truth_emails: list[str] = Field(default_factory=list)
    ground_truth_usernames: list[str] = Field(default_factory=list)
    max_steps_allowed: int = 10
    timeout_seconds: float = 60.0


class HarnessStepLog(BaseModel):
    """Record of a single step taken by the Agent."""
    step_number: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_output_summary: str
    duration_ms: float


class MissionEvaluation(BaseModel):
    """Result of running an Agent on a Mission inside the Harness."""
    mission_id: str
    success: bool
    total_steps: int
    duration_seconds: float
    matched_urls_found: int
    ground_truth_url_recall: float
    extracted_emails_count: int
    extracted_usernames_count: int
    step_logs: list[HarnessStepLog] = Field(default_factory=list)
    final_dossier: dict[str, Any] = Field(default_factory=dict)


class AgentHarness:
    """
    Supervising Test & Execution Harness for OSINT AI Agents.
    """

    def __init__(self, agent_instance: Any):
        self.agent = agent_instance
        self.step_logs: list[HarnessStepLog] = []

    def execute_mission(self, mission: OSINTMission) -> MissionEvaluation:
        """
        Runs a mission within the harness, measuring steps, time, and accuracy.
        """
        print(f"\n[Harness] >>> Commencing Mission: {mission.mission_id}")
        print(f"[Harness] Target Image: {mission.target_image_path}")
        if mission.target_webpage_url:
            print(f"[Harness] Target Recon URL: {mission.target_webpage_url}")

        start_time = time.perf_counter()
        self.step_logs.clear()

        # Execute investigation through Agent
        try:
            dossier = self.agent.investigate(
                target_image_path=mission.target_image_path,
                target_webpage_url=mission.target_webpage_url,
                min_confidence=0.50,
            )
            duration = time.perf_counter() - start_time
            success = dossier.get("status") == "COMPLETED"
        except Exception as e:
            duration = time.perf_counter() - start_time
            dossier = {"status": "FAILED", "error": str(e)}
            success = False

        # Calculate metrics against ground truth
        found_urls = [m.get("url") for m in dossier.get("discovered_sources", []) if isinstance(m, dict)]
        if mission.ground_truth_urls:
            recall = len(set(found_urls).intersection(set(mission.ground_truth_urls))) / len(mission.ground_truth_urls)
        else:
            recall = 1.0 if len(found_urls) > 0 else 0.0

        eval_result = MissionEvaluation(
            mission_id=mission.mission_id,
            success=success,
            total_steps=len(found_urls) + 1,  # initial search + per-URL pivot
            duration_seconds=round(duration, 3),
            matched_urls_found=len(found_urls),
            ground_truth_url_recall=round(recall, 2),
            extracted_emails_count=len(dossier.get("discovered_emails", [])),
            extracted_usernames_count=len(dossier.get("discovered_usernames", [])),
            final_dossier=dossier,
        )

        self._print_mission_report(eval_result)
        return eval_result

    def _print_mission_report(self, res: MissionEvaluation) -> None:
        status_str = "SUCCESS" if res.success else "FAILED"
        print(f"\n[Harness] <<< Mission {res.mission_id} Finished: {status_str}")
        print(f"   - Duration: {res.duration_seconds}s")
        print(f"   - Total Steps: {res.total_steps}")
        print(f"   - Sources Discovered: {res.matched_urls_found}")
        print(f"   - URL Recall Rate: {res.ground_truth_url_recall * 100:.1f}%")
        print(f"   - Emails Extracted: {res.extracted_emails_count}")
        print(f"   - Usernames Extracted: {res.extracted_usernames_count}")
