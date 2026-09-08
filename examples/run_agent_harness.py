import os
import sys
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.harness import AgentHarness, OSINTMission
from examples.agent_osint_investigator import AutonomousOSINTInvestigator


def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Agent OSINT Harness Runner")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to suspect target face photo")
    parser.add_argument("--url", "-u", type=str, default=None, help="Target website or news URL to investigate")
    parser.add_argument("--threshold", "-t", type=float, default=0.50, help="Face match score threshold (default: 0.50)")
    args = parser.parse_args()

    api_url = os.getenv("RECON_API_URL", "http://localhost:8000")
    print(f"[*] Initializing Autonomous OSINT Investigator Agent (Target API: {api_url})...")
    investigator = AutonomousOSINTInvestigator(recon_api_url=api_url)

    # Wrap the Agent inside the Evaluation & Testing Harness
    harness = AgentHarness(agent_instance=investigator)

    target_image = args.image
    if not target_image:
        # Check for any sample or existing image in tests or create one
        fixtures_dir = Path("tests/fixtures")
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        sample_path = fixtures_dir / "sample_target.jpg"
        if not sample_path.exists():
            import numpy as np, cv2
            blank = np.full((300, 300, 3), 200, dtype=np.uint8)
            cv2.imwrite(str(sample_path), blank)
        target_image = str(sample_path)

    # Define mission for the Agent
    mission = OSINTMission(
        mission_id="MISSION-OSINT-LIVE",
        target_image_path=target_image,
        target_webpage_url=args.url,
        max_steps_allowed=15,
        timeout_seconds=120.0,
    )

    print(f"[*] Dispatching Mission to Agent Harness...")
    report = harness.execute_mission(mission)

    print("\n" + "=" * 60)
    print("📋 FINAL AGENT EVALUATION REPORT:")
    print(f"   • Status: {'✅ SUCCESS' if report.success else '❌ FAILED'}")
    print(f"   • Execution Duration: {report.duration_seconds}s")
    print(f"   • High-Confidence Sources Discovered: {report.matched_urls_found}")
    print(f"   • Total Investigation Steps: {report.total_steps}")
    if report.final_dossier.get("discovered_sources"):
        print("\n🎯 MATCHED EVIDENCE FOUND:")
        for idx, src in enumerate(report.final_dossier["discovered_sources"], 1):
            print(f"   [{idx}] Score: {src.get('score', 0):.1%} | Page: {src.get('url')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
