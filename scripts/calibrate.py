from dmdod.benchmark import find_fastest_sustainable_rate
from dmdod.calibration import INITIAL_RATE_TARGETS


def main() -> None:
    print("=== Human Calibration v0.1 ===")
    print()

    short_single = find_fastest_sustainable_rate(mode="single", duration_s=5.0)
    sustained_single = find_fastest_sustainable_rate(mode="single", duration_s=20.0)
    alternating = find_fastest_sustainable_rate(mode="alternate", duration_s=10.0)

    human = {target.name: target.rate_hz for target in INITIAL_RATE_TARGETS}
    rows = (
        ("RI single (short)", human["RI single, short"], short_single.rate_hz),
        ("RI single (20 s)", human["RI single, sustained"], sustained_single.rate_hz),
        ("2-finger alternation", human["RI/RM same-hand alternation"], alternating.rate_hz),
    )

    print(f"{'Test':28s} {'Human':>9s} {'Sim':>9s} {'Error':>9s}")
    print("-" * 58)
    for name, human_rate, sim_rate in rows:
        error = sim_rate - human_rate
        print(f"{name:28s} {human_rate:8.2f}K {sim_rate:8.2f}K {error:+8.2f}K")

    print()
    print("Controller: periodic square-wave motor command")
    print("Pass criterion: >= 98% of requested presses become physical key DOWN events")
    print("Note: this measures the current body under a simple controller; it does not fit parameters yet.")


if __name__ == "__main__":
    main()
