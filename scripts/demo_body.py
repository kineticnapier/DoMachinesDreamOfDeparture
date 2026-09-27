from dmdod import KeyEvent, Simulation


def main() -> None:
    sim = Simulation()

    # A deliberately simple open-loop pattern: alternate motor commands every 100 ms.
    period_steps = 100
    for step in range(2000):
        phase = (step // period_steps) % 2
        left_command = 1.0 if phase == 0 else -1.0
        right_command = -left_command
        result = sim.step(left_command, right_command)

        for finger, event in result.events:
            if event is KeyEvent.DOWN:
                print(f"{result.time_s:7.3f}s  {finger:5s} DOWN")


if __name__ == "__main__":
    main()
