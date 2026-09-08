"""
LLM Integration Service for ReconFace OSINT Agent.
Supports Google Gemini, OpenAI, and custom OpenAI-compatible endpoints via HTTPX.
"""

from __future__ import annotations

from typing import Any
import httpx

from app.core.logging import get_logger

logger = get_logger("llm_service")


class LLMService:
    """Provides LLM-driven deep OSINT analysis and dossier synthesis."""

    @classmethod
    async def synthesize_dossier(
        cls,
        provider: str,
        api_key: str,
        model_name: str,
        base_url: str | None,
        target_name_or_query: str,
        matched_pages: list[dict[str, Any]],
    ) -> str:
        """
        Synthesizes a structured intelligence dossier from scraped article texts using LLM.
        """
        if not api_key and provider.lower() != "ollama":
            return ""

        # Prepare context from matched articles
        context_snippets = []
        for idx, page in enumerate(matched_pages[:10], 1):
            title = page.get("page_title", "")
            url = page.get("page_url", "")
            text = page.get("page_text", "")[:1200]
            context_snippets.append(f"[{idx}] หัวข้อ: {title}\nURL: {url}\nเนื้อหา: {text}\n")

        articles_context = "\n---\n".join(context_snippets)

        system_prompt = (
            "คุณคือ AI OSINT Analyst ผู้เชี่ยวชาญด้านการสืบสวนและวิเคราะห์ข้อมูลข่าวสาร "
            "หน้าที่ของคุณคือวิเคราะห์เนื้อหาข่าวสารที่รวบรวมได้จากเว็บไซต์เป้าหมาย "
            "และจัดทำรายงานสรุปข้อมูลประวัติบุคคล (Intelligence Dossier) อย่างเป็นมืออาชีพ เป็นภาษาไทย\n\n"
            "โครงสร้างรายงานที่ต้องมี:\n"
            "1. 👤 **ข้อมูลบุคคลและตำแหน่งหน้าที่**: (ชื่อ, ตำแหน่งทางวิชาการ/บริหาร, หน่วยงาน/คณะที่สังกัด)\n"
            "2. 🏢 **บทบาทและกิจกรรมหลัก**: (ภารกิจ, งานวิจัย, โครงการ หรือการประชุมสำคัญที่ปรากฏในข่าว)\n"
            "3. 📅 **ไทม์ไลน์ข่าวและเหตุการณ์ที่พบ**: (สรุปเหตุการณ์สำคัญเรียงตามข่าวที่พบ)\n"
            "4. 🎯 **ข้อสรุปและเบาะแสเพิ่มเติม**: (จุดสังเกต หรือข้อมูลติดต่อที่พบ)"
        )

        user_content = (
            f"เป้าหมายการสืบสวน: {target_name_or_query}\n\n"
            f"เนื้อหาข่าวที่รวบรวมได้จากเว็บไซต์:\n{articles_context}\n\n"
            "กรุณาจัดทำรายงานสรุปประวัติบุคคลเชิงลึก (Intelligence Dossier) ตามโครงสร้างที่กำหนด:"
        )

        try:
            prov = provider.lower()
            if prov == "ollama":
                # Local Ollama without API key
                target_model = model_name or "dolphin-llama3:latest"
                target_base = base_url or "http://localhost:11434/v1"
                return await cls._call_openai("ollama", target_model, target_base, system_prompt, user_content)
            elif prov == "gemini":
                return await cls._call_gemini(api_key, model_name or "gemini-1.5-flash", system_prompt, user_content)
            elif prov == "claude":
                return await cls._call_claude(api_key, model_name or "claude-3-5-sonnet-20241022", system_prompt, user_content)
            elif prov == "hermes":
                # Hermes Agent by Nous Research (via OpenRouter or Local Ollama/vLLM)
                hermes_model = model_name or "nousresearch/hermes-3-llama-3.1-405b"
                if base_url:
                    return await cls._call_openai(api_key or "ollama", hermes_model, base_url, system_prompt, user_content)
                return await cls._call_openrouter(api_key, hermes_model, system_prompt, user_content)
            elif prov == "openrouter":
                return await cls._call_openrouter(api_key, model_name or "anthropic/claude-3.5-sonnet", system_prompt, user_content)
            elif prov in ("openai", "custom"):
                return await cls._call_openai(api_key, model_name or "gpt-4o-mini", base_url, system_prompt, user_content)
        except Exception as e:
            logger.error("LLM synthesis error (%s): %s", provider, e)
            return f"\n⚠️ *หมายเหตุ: การวิเคราะห์เชิงลึกด้วย {provider} ขัดข้อง: {str(e)}*"

        return ""

    @classmethod
    async def _call_gemini(cls, api_key: str, model: str, system_prompt: str, user_content: str) -> str:
        # Clean model name if user entered with models/
        clean_model = model.replace("models/", "")
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{clean_model}:generateContent?key={api_key}"

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_content}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 2048},
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(endpoint, json=payload)
            if resp.status_code != 200:
                error_detail = resp.json().get("error", {}).get("message", resp.text)
                raise RuntimeError(f"Gemini API Error ({resp.status_code}): {error_detail}")

            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts:
                    return parts[0].get("text", "")
            return "ไม่สามารถดึงข้อความสรุปจาก Gemini ได้"

    @classmethod
    async def _call_claude(cls, api_key: str, model: str, system_prompt: str, user_content: str) -> str:
        endpoint = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": 2048,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_content}],
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code != 200:
                error_detail = resp.json().get("error", {}).get("message", resp.text)
                raise RuntimeError(f"Claude API Error ({resp.status_code}): {error_detail}")

            data = resp.json()
            content_items = data.get("content", [])
            if content_items and "text" in content_items[0]:
                return content_items[0]["text"]
            return "ไม่สามารถดึงข้อความสรุปจาก Claude ได้"

    @classmethod
    async def _call_openrouter(cls, api_key: str, model: str, system_prompt: str, user_content: str) -> str:
        endpoint = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "ReconFace OSINT",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code != 200:
                error_detail = resp.json().get("error", {}).get("message", resp.text)
                raise RuntimeError(f"OpenRouter API Error ({resp.status_code}): {error_detail}")

            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
            return "ไม่สามารถดึงข้อความสรุปจาก OpenRouter ได้"

    @classmethod
    async def _call_openai(cls, api_key: str, model: str, base_url: str | None, system_prompt: str, user_content: str) -> str:
        endpoint = (base_url.rstrip("/") + "/chat/completions") if base_url else "https://api.openai.com/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.3,
            "max_tokens": 2048,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(endpoint, json=payload, headers=headers)
            if resp.status_code != 200:
                error_detail = resp.json().get("error", {}).get("message", resp.text)
                raise RuntimeError(f"OpenAI API Error ({resp.status_code}): {error_detail}")

            data = resp.json()
            choices = data.get("choices", [])
            if choices and "message" in choices[0]:
                return choices[0]["message"].get("content", "")
            return "ไม่สามารถดึงข้อความสรุปจาก OpenAI ได้"
