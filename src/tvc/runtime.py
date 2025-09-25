"""Runtime helpers for configuring JAX backends at import time.

This module keeps the GPU plugin stack for JAX in a healthy state by detecting
common version mismatches (e.g. when `jaxlib` is newer than the
`jax-cuda12-plugin`). When an incompatibility is spotted we fall back to the CPU
backend so that the rest of the package can continue to operate, while emitting
clear upgrade instructions to restore GPU acceleration.

The logic is intentionally lightweight and avoids importing JAX directly so that
it can run before any PJRT clients are initialised.
"""
from __future__ import annotations

import os
import re
import sys
from importlib import metadata
from typing import Iterable, Optional, Tuple

__all__ = ["ensure_jax_runtime"]

_WARNED = False


def _distribution_version(names: Iterable[str]) -> Optional[str]:
    """Returns the installed version for the first matching distribution name."""

    for name in names:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return None


_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)")


def _major_minor(version: str) -> Tuple[int, int]:
    """Extracts the ``(major, minor)`` tuple from a version string."""

    match = _VERSION_PATTERN.match(version)
    if not match:
        return (0, 0)
    return (int(match.group(1)), int(match.group(2)))


def ensure_jax_runtime() -> None:
    """Ensures JAX initialises against a compatible backend.

    If an outdated CUDA plugin is detected, the function falls back to the CPU
    backend (by setting ``JAX_PLATFORMS=cpu``) and prints actionable guidance on
    how to upgrade. The check runs only once per interpreter session and is a
    no-op when the user already provided an explicit ``JAX_PLATFORMS`` value.
    """

    global _WARNED
    if _WARNED:
        return
    _WARNED = True

    if os.environ.get("TVC_SKIP_JAX_RUNTIME_CHECK"):
        return

    if os.environ.get("JAX_PLATFORMS"):
        # User explicitly selected a backend; assume they know what they are doing.
        return

    try:
        jaxlib_version = metadata.version("jaxlib")
    except metadata.PackageNotFoundError:
        # JAX is not installed; nothing to do.
        return

    plugin_version = _distribution_version(("jax-cuda12-plugin", "jax_cuda12_plugin"))
    pjrt_version = _distribution_version(("jax-cuda12-pjrt", "jax_cuda12_pjrt"))

    if not plugin_version and not pjrt_version:
        # No CUDA plugin on the path. Default JAX behaviour is to use CPU, so we
        # leave the environment untouched.
        return

    mismatches = []
    for label, version in (
        ("jax-cuda12-plugin", plugin_version),
        ("jax-cuda12-pjrt", pjrt_version),
    ):
        if not version:
            continue
        if _major_minor(version) != _major_minor(jaxlib_version):
            mismatches.append(f"{label} {version}")

    if not mismatches:
        return

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    formatted_mismatches = ", ".join(mismatches)
    guidance = (
        "Detected JAX CUDA plugin incompatibility.\n"
        f"  jaxlib: {jaxlib_version}\n"
        f"  plugins: {formatted_mismatches or 'unavailable'}\n"
        "Temporarily selecting the CPU backend (JAX_PLATFORMS=cpu) so execution can proceed.\n"
        "Upgrade the CUDA plugin stack to restore GPU support:\n"
        f"  pip install --upgrade \"jax=={jaxlib_version}\" \"jaxlib=={jaxlib_version}\" "
        f"\"jax-cuda12-plugin[with-cuda]=={jaxlib_version}\" \"jax-cuda12-pjrt=={jaxlib_version}\"\n"
        "If you have already upgraded, launch your job with JAX_PLATFORMS=gpu to re-enable GPU execution.\n"
        "Set TVC_SKIP_JAX_RUNTIME_CHECK=1 to suppress this warning."
    )
    print(guidance, file=sys.stderr)
