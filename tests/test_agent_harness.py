"""
Unit tests for the AI Agent Harness.
"""

from unittest.mock import MagicMock
from app.agent.harness import AgentHarness, OSINTMission


def test_agent_harness_mission_execution():
    mock_agent = MagicMock()
    mock_agent.investigate.return_value = {
        "status": "COMPLETED",
        "discovered_sources": [
            {"url": "https://target-forum.org/user/john", "score": 0.89},
            {"url": "https://social-archive.net/photos/101.jpg", "score": 0.76},
        ],
        "discovered_usernames": ["@john_doe", "@johny88"],
        "discovered_emails": ["john@example.com"],
    }

    harness = AgentHarness(agent_instance=mock_agent)

    mission = OSINTMission(
        mission_id="TEST-MISSION-01",
        target_image_path="sample.jpg",
        ground_truth_urls=["https://target-forum.org/user/john"],
    )

    evaluation = harness.execute_mission(mission)

    assert evaluation.success is True
    assert evaluation.matched_urls_found == 2
    assert evaluation.ground_truth_url_recall == 1.0  # Found the ground truth URL
    assert evaluation.extracted_emails_count == 1
    assert evaluation.extracted_usernames_count == 2
