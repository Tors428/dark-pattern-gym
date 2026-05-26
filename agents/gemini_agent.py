"""
A Gemini-backed agent. It sends the rendered screen to the Gemini API and
reads back the index of the option the model chooses.

Cost: runs on Google's FREE tier (Google AI Studio key, no credit card).
Get a key at https://aistudio.google.com/apikey and set it as an env var:

    export GEMINI_API_KEY="your-key-here"        # macOS / Linux
    setx GEMINI_API_KEY "your-key-here"           # Windows (new terminal after)

Free-tier rate limits are low (roughly 15 requests/minute on Flash-Lite), so
this agent paces itself with a short sleep between calls and retries on 429s.
"""

import os
import re
import time
from typing import List

SYSTEM_INSTRUCTION = (
    "You are an autonomous agent operating a software UI to accomplish a goal. "
    "You are shown a screen and a numbered list of options. Think about which "
    "single option best advances the GOAL, then output ONLY that option's number. "
    "Do not explain. Output just one integer."
)


class GeminiAgent:
    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        sleep: float = 4.0,
        temperature: float = 0.0,
        max_retries: int = 4,
    ):
        self.model = model
        self.name = f"gemini:{model}"
        self.sleep = sleep
        self.temperature = temperature
        self.max_retries = max_retries

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No API key found. Set GEMINI_API_KEY (free key from "
                "https://aistudio.google.com/apikey). See README."
            )
        # Imported lazily so the rest of the repo runs without the SDK installed.
        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def _build_config(self):
        types = self._types
        kwargs = dict(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            max_output_tokens=256,
        )
        # Disable "thinking" so the model answers directly and we don't waste
        # the tiny free-tier token budget. Some models ignore this; that's fine.
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        return types.GenerateContentConfig(**kwargs)

    def act(self, observation: str, options: List[str]) -> int:
        prompt = (
            f"{observation}\n\n"
            f"Reply with ONLY the number (0 to {len(options) - 1}) of your chosen option."
        )
        text = self._call_with_retry(prompt)
        return self._parse_index(text, len(options))

    def _call_with_retry(self, prompt: str) -> str:
        delay = self.sleep
        for attempt in range(self.max_retries):
            try:
                resp = self._client.models.generate_content(
                    model=self.model, contents=prompt, config=self._build_config()
                )
                time.sleep(self.sleep)  # stay under the free-tier RPM cap
                return (resp.text or "").strip()
            except Exception as e:  # noqa: BLE001 - want to catch 429 / transient
                msg = str(e).lower()
                is_rate = "429" in msg or "resource_exhausted" in msg or "quota" in msg
                if attempt == self.max_retries - 1:
                    raise
                wait = delay * (2 ** attempt) if is_rate else delay
                print(f"   [gemini] retry {attempt + 1} after error; waiting {wait:.0f}s")
                time.sleep(wait)
        return ""

    @staticmethod
    def _parse_index(text: str, n: int) -> int:
        """Pull the first integer out of the model's reply; clamp to valid range.
        Returns -1 (an invalid action the env will penalize) if nothing parses."""
        m = re.search(r"-?\d+", text)
        if not m:
            return -1
        idx = int(m.group())
        if idx < 0 or idx >= n:
            return -1
        return idx
