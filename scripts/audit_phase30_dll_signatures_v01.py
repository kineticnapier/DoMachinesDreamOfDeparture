from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


AUDITED_DLL_SHA256 = "0D1524DABF0F8C67AA58BA1992D6ED858522B9094490EBBB00E5E8AEFF7F40CD"

SIGNATURES = (
    ("scrPlayer.cs", "scrPlayer.Simulated_PlayerControl_Update", "public void Simulated_PlayerControl_Update(ulong? targetTick = null)"),
    ("scrPlayer.cs", "scrPlayer.CheckPostHoldFail", "private void CheckPostHoldFail(ulong? targetTick = null)"),
    ("scrPlayer.cs", "scrPlayer.HitInputEvent", "public bool HitInputEvent(bool isAuto = false, InputEventState state = InputEventState.Down)"),
    ("scrPlayer.cs", "scrPlayer.Die", "public void Die(bool overload = false, bool multipress = false, string failMessage = \"\", bool hitbox = false)"),
    ("scrController.cs", "scrController.ProcessKeyInputs", "private void ProcessKeyInputs(ulong eventTick)"),
    ("scrController.cs", "scrController.PlayerControl_Update", "private void PlayerControl_Update()"),
    ("scrController.cs", "scrController.OnPlayerDied", "public void OnPlayerDied(scrPlayer deadPlayer, bool overload = false, bool multipress = false, string failMessage = \"\", bool hitbox = false)"),
    ("scrController.cs", "scrController.Fail2Action", "public void Fail2Action()"),
    ("scrPlanet.cs", "scrPlanet.SwitchChosen", "public scrPlanet SwitchChosen()"),
    ("scrMarginTracker.cs", "scrMarginTracker.AddHit", "public void AddHit(HitMargin hit)"),
    ("scrHitTextManager.cs", "scrHitTextManager.ShowHitText", "public void ShowHitText(HitMargin hitMargin, scrPlanet planet, float missAngle = 0f)"),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit(source_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    texts: dict[str, str] = {}
    for filename, method, declaration in SIGNATURES:
        path = source_root / filename
        text = texts.setdefault(filename, path.read_text(encoding="utf-8-sig"))
        rows.append({
            "method": method,
            "source": filename,
            "declaration": declaration,
            "verified": declaration in text,
            "source_sha256": _sha256(path),
        })

    player = texts["scrPlayer.cs"]
    controller = texts["scrController.cs"]
    update_start = player.index("public void Simulated_PlayerControl_Update")
    update_end = player.index("private void CheckPostHoldFail", update_start)
    update_body = player[update_start:update_end]
    order_tokens = (
        "CheckPostHoldFail(targetTick)",
        "OttoHoldHit(targetTick)",
        "HitAutoFloors(targetTick)",
        "UpdateHoldBehavior(targetTick)",
        "HitHoldFloorsIfStartedAtHold(targetTick)",
        "CheckPreHoldFail(targetTick)",
        "UpdateHoldKeys(targetTick)",
    )
    positions = {token: update_body.find(token) for token in order_tokens}
    order_verified = all(value >= 0 for value in positions.values()) and list(positions.values()) == sorted(positions.values())

    no_fail_tokens = (
        "ADOBase.controller.noFailInfiniteMargin = true",
        "Hit()",
        "ADOBase.controller.noFailInfiniteMargin = false",
    )
    die_start = player.index("public void Die(bool")
    die_end = player.index("public void ClearMisses", die_start)
    die_body = player[die_start:die_end]
    no_fail_positions = {token: die_body.find(token) for token in no_fail_tokens}
    no_fail_order = all(value >= 0 for value in no_fail_positions.values()) and list(no_fail_positions.values()) == sorted(no_fail_positions.values())

    input_dispatch_verified = (
        "private void ProcessKeyInputs(ulong eventTick)" in controller
        and "item.Simulated_PlayerControl_Update(num);" in controller
    )
    stop_path_verified = (
        "ADOBase.controller.OnPlayerDied(this, overload, multipress, failMessage, hitbox);" in player
        and "ChangeState(States.Fail);" in controller
        and "ChangeState(States.Fail2);" in controller
    )
    all_signatures = all(bool(row["verified"]) for row in rows)
    return {
        "format": "dmdod-phase30-dll-signature-audit-v0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audited_dll_sha256": AUDITED_DLL_SHA256,
        "installed_dll_identity": "USER_CONFIRMED_NOT_CODEX_EXECUTED",
        "signatures": rows,
        "control_flow": {
            "simulated_update_order": positions,
            "simulated_update_order_verified": order_verified,
            "no_fail_die_hit_order": no_fail_positions,
            "no_fail_die_hit_order_verified": no_fail_order,
            "async_input_dispatch_verified": input_dispatch_verified,
            "fail_stop_path_verified": stop_path_verified,
        },
        "verdict": "DLL_SIGNATURES_VERIFIED" if all_signatures and order_verified and no_fail_order and input_dispatch_verified and stop_path_verified else "DLL_SIGNATURES_PARTIAL",
        "runtime_parity": "RUNTIME_PARITY_UNVERIFIED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(r"F:\adofai-code"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = audit(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
