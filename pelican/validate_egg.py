#!/usr/bin/env python3
"""Check the Pelican egg before anyone tries to import it.

An egg that imports but is subtly wrong is worse than one that fails loudly:
the panel will happily accept a startup line referring to a variable that does
not exist and then fail at boot with nothing useful in the console. So this
validates the parts a panel will not:

* every ``{{PLACEHOLDER}}`` in the startup line is either declared as a
  variable or supplied by wings itself;
* every declared variable is actually used;
* the startup line parses as real Spotter arguments, with the defaults filled
  in, for every startup mode the egg offers;
* the done-marker is a string the application genuinely logs.

Run from the repo root: ``python pelican/validate_egg.py``
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EGG = ROOT / "pelican" / "egg-spotter.json"

#: Supplied by wings, not by the egg.
WINGS_PROVIDED = {"SERVER_PORT", "SERVER_IP", "SERVER_MEMORY", "P_SERVER_UUID",
                  "P_SERVER_ALLOCATION_LIMIT", "STARTUP", "TZ"}

#: Variables the application reads from the environment rather than the
#: command line, so they legitimately never appear in the startup string.
ENV_ONLY = {"AISSTREAM_API_KEY"}


def fail(message: str) -> None:
    print(f"  FAIL  {message}")
    fail.count += 1


fail.count = 0


def main() -> int:
    sys.path.insert(0, str(ROOT))

    egg = json.loads(EGG.read_text(encoding="utf-8"))
    print(f"egg: {egg['name']} ({egg['meta']['version']})")

    declared = {v["env_variable"] for v in egg["variables"]}
    placeholders = set(re.findall(r"\{\{(\w+)\}\}", egg["startup"]))

    for name in sorted(placeholders - declared - WINGS_PROVIDED):
        fail(f"startup uses {{{{{name}}}}} but no variable declares it")
    for name in sorted(declared - placeholders - ENV_ONLY):
        fail(f"variable {name} is declared but never used")

    for variable in egg["variables"]:
        for key in ("name", "description", "env_variable", "default_value",
                    "user_viewable", "user_editable", "rules"):
            if key not in variable:
                fail(f"variable {variable.get('env_variable')} is missing {key}")

    # The done-marker has to be something we really print, or the panel will
    # sit on "starting" forever.
    done = json.loads(egg["config"]["startup"])["done"]
    sources = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (ROOT / "spotter").rglob("*.py"))
    if done not in sources:
        fail(f"done-marker {done!r} is never logged by the application")
    else:
        print(f"  ok    done-marker {done!r} is logged")

    # Now the real test: build the command wings would run, for each mode, and
    # push it through the actual argument parser.
    from spotter.cli import build_parser, merge_global_options
    from spotter.config import parse_set_overrides

    defaults = {v["env_variable"]: v["default_value"] for v in egg["variables"]}
    defaults.update({"SERVER_PORT": "25565", "SERVER_IP": "0.0.0.0"})

    modes = re.search(r"in:([a-z,]+)", next(
        v["rules"] for v in egg["variables"]
        if v["env_variable"] == "STARTUP_MODE"))
    for mode in (modes.group(1).split(",") if modes else ["run"]):
        startup = egg["startup"]
        for key, value in {**defaults, "STARTUP_MODE": mode}.items():
            startup = startup.replace("{{%s}}" % key, value)
        if "{{" in startup:
            fail(f"unexpanded placeholder for mode {mode}: {startup}")
            continue
        argv = shlex.split(startup)[3:]      # drop "python -m spotter"
        try:
            args = merge_global_options(build_parser().parse_args(argv))
            parse_set_overrides(args.set)
        except SystemExit:
            fail(f"startup does not parse for STARTUP_MODE={mode}")
            continue
        except Exception as exc:
            fail(f"startup invalid for STARTUP_MODE={mode}: {exc}")
            continue
        print(f"  ok    STARTUP_MODE={mode} parses -> {args.func.__name__}")

    if fail.count:
        print(f"\n{fail.count} problem(s)")
        return 1
    print("\negg is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
