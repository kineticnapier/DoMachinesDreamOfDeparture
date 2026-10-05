from __future__ import annotations

from dmdod.modern_cli_dagger import DaggerLiveModernTrainerConsole


class _FakeBar:
    def __init__(self, total: int, desc: str = "") -> None:
        self.total = float(total)
        self.n = 0.0
        self.desc = desc
        self.postfix = ""
        self.closed = False

    def update(self, delta: float) -> None:
        self.n += float(delta)

    def refresh(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def set_postfix_str(self, text: str) -> None:
        self.postfix = str(text)


def _console() -> tuple[DaggerLiveModernTrainerConsole, list[str]]:
    console = DaggerLiveModernTrainerConsole(
        rounds=1,
        round_epochs=1,
        bootstrap_epochs=1,
    )
    console.enabled = True

    def factory(*, total, desc, colour, position):
        return _FakeBar(total, desc)

    console._bar = factory  # type: ignore[method-assign]
    console._live_bar = factory  # type: ignore[method-assign]
    writes: list[str] = []
    console._write = writes.append  # type: ignore[method-assign]
    return console, writes


def test_dagger_parent_and_trust_alpha_progress() -> None:
    console, writes = _console()

    assert not console._handle_n_key_dagger_line(
        "anchors=20 validation=20 dagger-epochs=8 action=continuous lr=3e-05 | FINAL untouched"
    )
    assert not console._handle_n_key_dagger_line(
        "trust-alphas=0.000488281,0.000244141,0.00012207"
    )

    assert not console._handle_n_key_dagger_line(
        "aggregate-data generation=0 expert=58956 student-state=60493 total=119449 frames"
    )
    parent = console._dagger_epoch_bar
    assert parent is not None
    assert parent.total == 8
    assert parent.n == 0

    assert console._handle_n_key_dagger_line("dagger-proposal 001/8 loss=0.608031")
    assert "epoch=1" in parent.postfix
    assert "loss=0.608031" in parent.postfix

    assert console._handle_n_key_dagger_line("=== Train trust line search epoch 1/8 ===")
    first = console._stage
    assert first is not None
    assert first.desc == "Trust a=0.000488281"
    assert first.n == 0

    assert console._handle_n_key_dagger_line(
        "epoch-001-a0.000488281 05/20 01645 - First town: H=10/100 X=50.00%"
    )
    assert first.n == 5

    assert console._handle_n_key_dagger_line(
        "trust alpha=0.000488281: SAFE+IMPROVE H=1046/5845 X=53.01% early=241 over=True keydowns=1287"
    )
    assert first.closed
    second = console._stage
    assert second is not None
    assert second.desc == "Trust a=0.000244141"
    assert writes[-1].startswith("trust alpha=0.000488281:")

    assert console._handle_n_key_dagger_line(
        "epoch-continuation: ACCEPT pair epoch=1 alpha=0.000244141 H=1132/5845 X=52.43%"
    )
    assert parent.n == 1
    assert "ACCEPT epoch=1" in parent.postfix


def test_dagger_reject_early_stop_closes_parent() -> None:
    console, writes = _console()
    console._handle_n_key_dagger_line(
        "anchors=20 validation=20 dagger-epochs=8 action=continuous lr=3e-05 | FINAL untouched"
    )
    console._handle_n_key_dagger_line(
        "aggregate-data generation=0 expert=58956 student-state=60493 total=119449 frames"
    )
    parent = console._dagger_epoch_bar
    assert parent is not None

    console._handle_n_key_dagger_line("dagger-proposal 003/8 loss=0.600000")
    console._handle_n_key_dagger_line(
        "epoch-continuation: KEEP previous accepted pair epoch=2 alpha=0.000244141 H=1200/5845 X=53.00%"
    )
    assert parent.n == 3

    assert console._handle_n_key_dagger_line(
        "epoch-loop: EARLY STOP after rejected proposal; accepted model+optimizer+DAgger data are unchanged"
    )
    assert parent.closed
    assert console._dagger_epoch_bar is None
    assert writes[-1].startswith("epoch-loop: EARLY STOP")


def test_dagger_validation_uses_live_stage() -> None:
    console, writes = _console()

    assert console._handle_n_key_dagger_line(
        "validation 03/20 03057 - First Town: H=7/132 X=51.25%"
    )
    stage = console._stage
    assert stage is not None
    assert stage.desc == "Validation"
    assert stage.n == 3

    assert console._handle_n_key_dagger_line(
        "validation aggregate: H=666/4682 X=50.94% early=208 over=True keydowns=874"
    )
    assert stage.closed
    assert console._stage is None
    assert writes[-1].startswith("validation aggregate:")
