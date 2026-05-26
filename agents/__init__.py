from .mock_agent import KeywordAgent

__all__ = ["KeywordAgent", "make_agent"]


def make_agent(kind: str, **kwargs):
    """Factory so the SDK is only imported when a Gemini agent is requested."""
    if kind in ("keyword", "mock", "baseline"):
        return KeywordAgent()
    if kind in ("gemini", "llm"):
        from .gemini_agent import GeminiAgent
        return GeminiAgent(**kwargs)
    raise ValueError(f"Unknown agent '{kind}'. Use 'keyword' or 'gemini'.")
