"""
Run the experiment: have an agent attempt the cancel-subscription task on the
clean UI and the dark UI, repeat N times each, and save the results.

Examples
--------
# Free, instant, no API key — runs the keyword baseline:
python run_experiment.py --agent keyword --episodes 20

# Free Gemini agent (needs a free GEMINI_API_KEY, see README):
python run_experiment.py --agent gemini --episodes 10

# Add the stretch-goal LLM judge on top:
python run_experiment.py --agent gemini --episodes 10 --judge
"""

import argparse
import json
import os
from dataclasses import asdict
from statistics import mean
from typing import Dict, List

from env import DarkPatternEnv, GOAL
from agents import make_agent

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_episode(env: DarkPatternEnv, agent) -> Dict:
    obs = env.reset()
    trajectory: List[Dict] = []
    total_reward = 0.0
    while not env.done:
        options = [el.label for el in env.current_actions()]
        action = agent.act(obs, options)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        trajectory.append(asdict(info))
    return {
        "success": env.succeeded,
        "terminal": env.terminal_kind,
        "steps": env.steps,
        "trap_hits": env.trap_hits,
        "total_reward": round(total_reward, 3),
        "trajectory": trajectory,
    }


def summarize(episodes: List[Dict]) -> Dict:
    n = len(episodes)
    return {
        "episodes": n,
        "success_rate": round(mean(e["success"] for e in episodes), 3),
        "avg_steps": round(mean(e["steps"] for e in episodes), 2),
        "avg_trap_hits": round(mean(e["trap_hits"] for e in episodes), 2),
        "avg_reward": round(mean(e["total_reward"] for e in episodes), 3),
        "failure_rate": round(mean(e["terminal"] == "failure" for e in episodes), 3),
        "timeout_rate": round(mean(e["terminal"] == "timeout" for e in episodes), 3),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Dark Pattern Gym experiment runner")
    p.add_argument("--agent", default="keyword", choices=["keyword", "gemini"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--model", default="gemini-2.5-flash-lite")
    p.add_argument("--sleep", type=float, default=4.0, help="seconds between API calls")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--judge", action="store_true", help="run the stretch-goal LLM judge")
    p.add_argument("--ui", default="both", choices=["both", "clean", "dark"], help="run only one UI")
    p.add_argument("--out", default=os.path.join(RESULTS_DIR, "results.json"))
    args = p.parse_args()

    agent_kwargs = {}
    if args.agent == "gemini":
        agent_kwargs = {"model": args.model, "sleep": args.sleep}
    agent = make_agent(args.agent, **agent_kwargs)

    judge = None
    if args.judge:
        from agents.judge import GeminiJudge
        judge = GeminiJudge(model=args.model, sleep=args.sleep)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {"agent": agent.name, "by_ui": {}}

    def save():
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)

    uis = ("clean", "dark") if args.ui == "both" else (args.ui,)
    for ui in uis:
        print(f"\n=== UI: {ui} | agent: {agent.name} | {args.episodes} episodes ===")
        episodes = []
        try:
            for i in range(args.episodes):
                env = DarkPatternEnv(ui_name=ui, max_steps=args.max_steps)
                ep = run_episode(env, agent)
                if judge is not None:
                    j = judge.score(GOAL, ep["trajectory"])
                    ep["judge_score"] = j.get("score")
                    ep["judge_reason"] = j.get("reason")
                episodes.append(ep)
                tag = "OK " if ep["success"] else "no "
                print(f"  ep {i + 1:>2}: {tag} steps={ep['steps']} traps={ep['trap_hits']} reward={ep['total_reward']}")
        except Exception as e:
            print(f"\n  !! stopped early ({type(e).__name__}); saving {len(episodes)} episode(s).")
            if episodes:
                results["by_ui"][ui] = {"summary": summarize(episodes), "episodes": episodes}
                save()
            return
        results["by_ui"][ui] = {"summary": summarize(episodes), "episodes": episodes}
        save()
        print(f"  --> {summarize(episodes)}")

    print(f"\nSaved results to {args.out}")
    print("Next: python analyze.py")


if __name__ == "__main__":
    main()
