from dmdod.motor_env import LeftThresholdReflexPolicy, MotorEnv


def main() -> None:
    env = MotorEnv(same_hand=True, control_dt_s=0.010)
    policy = LeftThresholdReflexPolicy()
    observation = env.reset()

    duration_s = 5.0
    decisions = round(duration_s / env.control_dt_s)
    down_events = 0

    print("=== Motor Agent Environment v0.1 ===")
    print(f"physics dt: {env.sim.config.dt_s * 1000:.1f} ms")
    print(f"policy dt:  {env.control_dt_s * 1000:.1f} ms")
    print("policy observation: position / velocity / digital key state")
    print("hidden from policy: exact time / fatigue / coordination / target timestamps")
    print()

    for _ in range(decisions):
        action = policy.act(observation)
        transition = env.step(action)
        observation = transition.observation
        down_events += sum(1 for event in transition.evaluator_events if event.event.value == "down")

    diagnostics = env.diagnostics()
    print(f"left DOWN events: {down_events}")
    print(f"observed rate:    {down_events / duration_s:.3f} KPS")
    print(f"final fatigue:    {diagnostics.left_fatigue:.5f}  (privileged diagnostic)")


if __name__ == "__main__":
    main()
