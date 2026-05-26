"""Agent interface. An agent looks at the rendered screen plus the list of
option labels and returns the index of the option it chooses."""

from typing import List, Protocol


class Agent(Protocol):
    name: str

    def act(self, observation: str, options: List[str]) -> int:
        """Return the index (0-based) of the chosen option."""
        ...
