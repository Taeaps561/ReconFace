"""
Example: Autonomous Multi-hop OSINT Investigation Agent.

Demonstrates how an AI Agent can:
1. Reverse search a query face.
2. Scrape discovered source URLs for entity mentions, usernames, and emails.
3. Pivot into newly discovered image URLs for cross-confirmation.
4. Synthesize a comprehensive Dossier Report.
"""

import json
import re
import sys
from typing import Any
import httpx

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.agent.tools import ReconFaceTools


class AutonomousOSINTInvestigator:
    def __init__(self, recon_api_url: str = "http://localhost:8000"):
        self.tools = ReconFaceTools(api_base_url=recon_api_url)
        self.session_intel: dict[str, Any] = {
            "query_image": None,
            "matched_urls": [],
            "extracted_entities": {
                "usernames": set(),
                "emails": set(),
                "names": set(),
            },
            "pivoted_images": [],
        }

    def investigate(
        self,
        target_image_path: str,
        target_webpage_url: str | None = None,
        min_confidence: float = 0.50,
    ) -> dict[str, Any]:
        self.session_intel["query_image"] = target_image_path

        if target_webpage_url:
            print(f"[1] 🕸️ Agent: Scanning target website {target_webpage_url} for face in {target_image_path}...")
            scan_result = self.tools.scan_webpage_for_target(
                image_input=target_image_path,
                webpage_url=target_webpage_url,
                score_threshold=min_confidence,
            )
            matches = scan_result.get("matches", [])
            print(f"    -> Crawled {scan_result.get('pages_crawled_count', 0)} pages, scanned {scan_result.get('images_scanned', 0)} images.")
            print(f"    -> Found {len(matches)} high-confidence visual matches.")
            
            for m in matches:
                url = m.get("page_url") or m.get("image_url")
                score = m.get("score", 0.0)
                self.session_intel["matched_urls"].append({"url": url, "score": score, "image_url": m.get("image_url")})
                if url and url.startswith("http"):
                    page_text = self._scrape_page_content(url)
                    self._extract_entities(page_text)
        else:
            print(f"[1] 🔍 Agent: Performing Reverse Face Search for {target_image_path}...")
            search_result = self.tools.search_face(
                image_input=target_image_path,
                top_k=5,
                score_threshold=min_confidence,
            )
            matches = search_result.get("matches", [])
            print(f"    -> Found {len(matches)} potential visual matches in vector index.")

            # Pivot to each matched source URL
            for match in matches:
                url = match.get("source_url")
                score = match.get("score", 0.0)
                if not url or not url.startswith("http"):
                    continue

                print(f"[2] 🌐 Agent: Pivoting & scraping source URL: {url} (Similarity: {score:.2%})...")
                self.session_intel["matched_urls"].append({"url": url, "score": score})
                page_text = self._scrape_page_content(url)
                self._extract_entities(page_text)

        # Synthesize Dossier
        print("\n[3] 📑 Agent: Synthesizing Final Intelligence Dossier...")
        dossier = {
            "status": "COMPLETED",
            "target": target_image_path,
            "high_confidence_matches": len(self.session_intel["matched_urls"]),
            "discovered_sources": self.session_intel["matched_urls"],
            "discovered_usernames": list(self.session_intel["extracted_entities"]["usernames"]),
            "discovered_emails": list(self.session_intel["extracted_entities"]["emails"]),
        }
        return dossier

    def _scrape_page_content(self, url: str) -> str:
        """Fetches and strips HTML content."""
        try:
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                res = client.get(url)
                if res.status_code == 200:
                    return res.text
        except Exception:
            pass
        return ""

    def _extract_entities(self, text: str) -> None:
        """Simple regex extraction for emails and usernames."""
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        for email in emails:
            self.session_intel["extracted_entities"]["emails"].add(email)

        handles = re.findall(r"@([a-zA-Z0-9_]{3,25})", text)
        for h in handles:
            self.session_intel["extracted_entities"]["usernames"].add(f"@{h}")


if __name__ == "__main__":
    # Quick demonstration execution
    agent = AutonomousOSINTInvestigator()
    print("Agent OSINT Investigator Initialized.")
