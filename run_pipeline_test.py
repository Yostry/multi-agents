"""Run a simple mass-spring-damper test through the 6-Agent pipeline."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def main():
    from amesim_multi_agent.orchestrator_runner import run_pipeline
    user_request = "Build a mass-spring-damper system: mass=10kg, spring stiffness=1000N/m, damping=100Ns/m, fixed base"
    state = await run_pipeline(user_request, skip_review=True, review_mode="auto")
    print(f"\n=== PIPELINE FINAL STATE ===")
    print(f"History: {' -> '.join(state.stage_history)}")
    print(f"Total retries: {sum(state.agent_retry_counts.values())}")
    print(f"Consecutive failures: {state.consecutive_failures}")
    print(f"AME file: {state.build_result.ame_path if state.build_result else 'NOT GENERATED'}")

if __name__ == "__main__":
    asyncio.run(main())
