"""
DART Disclosure Analyzer
------------------------
Uses a local LLM to classify importance, tag categories, and generate summaries.
"""
from __future__ import annotations

import json
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class DartDisclosureAnalyzer:
    def __init__(self, model_dir: str = "models/qwen2.5-1.5b-instruct"):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True
        )

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "mps" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            local_files_only=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        self.model.to(device)
        self.model.eval()

    def analyze(self, raw_text: str) -> dict[str, Any]:
        trimmed = self._trim_text(raw_text, max_chars=8000)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a DART disclosure analyst. Output ONLY valid JSON.\n"
                    "Required JSON keys:\n"
                    "- importance_level: one of IMP | NORMAL | NOT_IMPORTANT\n"
                    "- tags: array of [earnings, investment, contract, risk, governance, restructuring, compliance, other]\n"
                    "- title_summary: one-line summary (Korean)\n"
                    "- paragraph_summary: 3-5 sentence summary (Korean)\n\n"
                    "Importance rules:\n"
                    "IMP: direct impact on revenue/profit, major contracts/orders, core investment, sharp performance change\n"
                    "NORMAL: meaningful but limited short-term revenue impact\n"
                    "NOT_IMPORTANT: formal/compliance/repetitive disclosures\n\n"
                    "Rules:\n"
                    "- importance_level must be exactly one label\n"
                    "- if uncertain, choose NORMAL\n"
                    "- tags cannot be empty; use other if nothing fits\n"
                    "- output JSON only, no extra text"
                ),
            },
            {
                "role": "user",
                "content": f"Disclosure text:\n{trimmed}",
            },
        ]

        chat_prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=280,
                do_sample=False,
                temperature=None,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id
            )

        generated = output[0][inputs["input_ids"].shape[-1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        parsed = self._extract_json(text)
        if parsed:
            return self._normalize_output(parsed)
        return self._fallback_analysis(trimmed)

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        snippet = text[start:end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return None

    def _normalize_output(self, data: dict[str, Any]) -> dict[str, Any]:
        importance = data.get("importance_level", "NORMAL")
        if importance not in {"IMP", "NORMAL", "NOT_IMPORTANT"}:
            importance = "NORMAL"

        tags = data.get("tags") or ["other"]
        if isinstance(tags, str):
            tags = [tags]
        tags = [t for t in tags if t in {
            "earnings", "investment", "contract", "risk",
            "governance", "restructuring", "compliance", "other"
        }]
        if not tags:
            tags = ["other"]

        return {
            "importance_level": importance,
            "tags": tags,
            "title_summary": str(data.get("title_summary", "")).strip() or "(summary unavailable)",
            "paragraph_summary": str(data.get("paragraph_summary", "")).strip() or "(summary unavailable)",
        }

    def _fallback_analysis(self, text: str) -> dict[str, Any]:
        keywords_imp = ["매출", "영업이익", "순이익", "공급계약", "수주", "대형", "투자", "신규사업"]
        importance = "IMP" if any(k in text for k in keywords_imp) else "NORMAL"
        tags = ["other"]
        if "실적" in text or "매출" in text:
            tags = ["earnings"]
        elif "계약" in text or "수주" in text:
            tags = ["contract"]
        elif "투자" in text or "CAPEX" in text:
            tags = ["investment"]
        elif "지배구조" in text or "이사회" in text:
            tags = ["governance"]
        elif "리스크" in text or "소송" in text:
            tags = ["risk"]
        elif "정정" in text or "감사" in text:
            tags = ["compliance"]

        return {
            "importance_level": importance,
            "tags": tags,
            "title_summary": text.splitlines()[0][:120] if text else "(summary unavailable)",
            "paragraph_summary": "Summary unavailable; using raw text excerpt.",
        }

    @staticmethod
    def _trim_text(text: str, max_chars: int = 8000) -> str:
        if len(text) <= max_chars:
            return text
        return text[:max_chars]
