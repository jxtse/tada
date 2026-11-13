"""Cross-process model slots shared by designer and solver entry points."""
from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOCK_ROOT = REPO / "v5_runs" / ".machine_slots"
CAPACITY = {"opus": 1, "deepseek": 3, "astra": 2}
HELD_FAMILY = ContextVar('tada_model_slot_family', default=None)


def model_family(model: str):
    return next((family for family in CAPACITY if family in model.lower()), None)


def slot_is_held(model):
    family = model_family(model)
    return family is not None and (HELD_FAMILY.get() == family or
        os.environ.get('TADA_MODEL_SLOT_HELD') == '1' and os.environ.get('TADA_MODEL_SLOT_FAMILY') == family)


@contextmanager
def borrowed_slot(model, *, lock_root: Path | None = None):
    if slot_is_held(model):
        yield
        return
    with model_slot(model, lock_root=lock_root):
        token = HELD_FAMILY.set(model_family(model))
        try:
            yield
        finally:
            HELD_FAMILY.reset(token)


def subprocess_environment(model):
    env = dict(os.environ)
    if slot_is_held(model):
        env.update(TADA_MODEL_SLOT_HELD='1', TADA_MODEL_SLOT_FAMILY=model_family(model))
    else:
        env.pop('TADA_MODEL_SLOT_HELD', None)
        env.pop('TADA_MODEL_SLOT_FAMILY', None)
    return env


@contextmanager
def model_slot(model: str, timeout: float | None = None, lock_root: Path | None = None):
    """Advisory locks are released by the kernel on crash, unlike PID files.

    All repository entry points use this fixed machine-local directory. A caller
    must not create a private lock directory in production to bypass the cap.
    Queue waiting is unbounded by default because an occupied trial slot may
    legitimately run for eight hours. Request and solver timeouts are separate.
    """
    family = model_family(model)
    if family is None:
        yield
        return
    root = lock_root or LOCK_ROOT
    root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout if timeout is not None else None
    acquired = None
    while acquired is None:
        for index in range(CAPACITY[family]):
            stream = (root / f"{family}-{index}.lock").open("a+")
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
            else:
                acquired = stream
                stream.seek(0)
                stream.truncate()
                stream.write(str(os.getpid()))
                stream.flush()
                break
        if acquired is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"No {family} model slot available")
            time.sleep(0.2 if deadline is None else min(0.2, max(0.001, deadline - time.monotonic())))
    try:
        yield
    finally:
        fcntl.flock(acquired, fcntl.LOCK_UN)
        acquired.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("A command is required")
    # The fd remains held by this parent for the entire solver lifetime.
    with model_slot(args.model):
        env = dict(os.environ, TADA_MODEL_SLOT_HELD="1", TADA_MODEL_SLOT_FAMILY=model_family(args.model) or "")
        raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
