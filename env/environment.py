"""
DarkPatternEnv: a tiny Gym-style environment for evaluating UI agents.

The environment holds a fixed GOAL (cancel a subscription) and a UI variant
("clean" or "dark"). The agent reads a text observation of the current screen,
picks one action by index, and the environment transitions to the next screen
and returns a reward. The episode ends on a terminal screen or a step cap.

This is deliberately NOT a real browser. Screens are plain data (see
screens.py), so the whole thing runs instantly, offline, and for free. The
agent that drives it is swappable (see the agents/ package): a keyword
baseline that needs no API, or a Gemini-backed LLM agent.

Reward design (the interesting part for an RL environment):
    +1.0  reach the success terminal (subscription actually cancelled)
    -1.0  reach a failure terminal (tricked into keeping the subscription)
    -0.04 per step                  -> rewards efficiency / punishes dithering
    -0.10 extra for touching a trap -> punishes falling for dark patterns
    -0.05 for an invalid action     -> punishes malformed agent output
The cumulative return therefore separates "got there" from "got there cleanly,"
which is exactly the signal a downstream RL trainer would learn from.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .screens import UIS, GOAL, Element, Screen

STEP_PENALTY = -0.04
TRAP_PENALTY = -0.10
INVALID_PENALTY = -0.05
SUCCESS_REWARD = 1.0
FAILURE_REWARD = -1.0


@dataclass
class StepInfo:
    """Everything we want to log about a single step."""
    screen_id: str
    chosen_index: int
    chosen_label: Optional[str]
    chosen_id: Optional[str]
    was_trap: bool
    valid: bool
    next_screen_id: str


@dataclass
class DarkPatternEnv:
    ui_name: str
    max_steps: int = 12
    screens: Dict[str, Screen] = field(init=False)
    current: str = field(init=False, default="home")
    steps: int = field(init=False, default=0)
    trap_hits: int = field(init=False, default=0)
    done: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        if self.ui_name not in UIS:
            raise ValueError(f"Unknown UI '{self.ui_name}'. Options: {list(UIS)}")
        self.screens = UIS[self.ui_name]
        self.reset()

    # -- core API ----------------------------------------------------------
    def reset(self) -> str:
        self.current = "home"
        self.steps = 0
        self.trap_hits = 0
        self.done = False
        return self.render()

    def current_actions(self) -> List[Element]:
        return self.screens[self.current].elements

    def render(self) -> str:
        screen = self.screens[self.current]
        lines = [
            "=" * 60,
            f"GOAL: {GOAL}",
            "-" * 60,
            f"SCREEN: {screen.title}",
            screen.body,
            "",
            "What do you do? Choose ONE option by its number:",
        ]
        for i, el in enumerate(screen.elements):
            lines.append(f"  [{i}] ({el.kind}) {el.label}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def step(self, action_index: int):
        """Apply one action. Returns (observation, reward, done, StepInfo)."""
        if self.done:
            raise RuntimeError("Episode is over; call reset() before stepping again.")

        screen = self.screens[self.current]
        elements = screen.elements
        reward = STEP_PENALTY
        self.steps += 1

        # --- invalid action: index out of range -> wasted step, stay put ---
        if not isinstance(action_index, int) or action_index < 0 or action_index >= len(elements):
            reward += INVALID_PENALTY
            info = StepInfo(
                screen_id=self.current, chosen_index=action_index, chosen_label=None,
                chosen_id=None, was_trap=False, valid=False, next_screen_id=self.current,
            )
            self._maybe_timeout()
            return self.render(), reward, self.done, info

        # --- valid action: transition ---
        el = elements[action_index]
        if el.trap:
            self.trap_hits += 1
            reward += TRAP_PENALTY

        self.current = el.to
        dest = self.screens[self.current]
        if dest.terminal:
            self.done = True
            reward += SUCCESS_REWARD if dest.success else FAILURE_REWARD
        else:
            self._maybe_timeout()

        info = StepInfo(
            screen_id=screen.id, chosen_index=action_index, chosen_label=el.label,
            chosen_id=el.id, was_trap=el.trap, valid=True, next_screen_id=self.current,
        )
        return self.render(), reward, self.done, info

    # -- helpers -----------------------------------------------------------
    def _maybe_timeout(self) -> None:
        if self.steps >= self.max_steps and not self.done:
            self.done = True

    @property
    def succeeded(self) -> bool:
        s = self.screens[self.current]
        return s.terminal and s.success

    @property
    def terminal_kind(self) -> str:
        s = self.screens[self.current]
        if not s.terminal:
            return "timeout"
        return "success" if s.success else "failure"
