# -*- coding: utf-8 -*-

"""
Provenance stamps written into the pickles produced by iMolCRAFT.

A checkpoint on its own does not say which code wrote it, so a result that
cannot be reproduced later gives no way to tell whether the code has moved on
since. Stamping the package version and the git commit into every pickle makes
that answerable after the fact.
"""

import subprocess
from pathlib import Path

from . import __version__

#: Cached result of :func:`get_git_hash`; ``False`` means "not looked up yet"
#: (``None`` is a legitimate answer, so it cannot double as the sentinel).
_GIT_HASH_CACHE = False


def get_git_hash():
    """
    Return the git commit hash of the checked-out iMolCRAFT source.

    The lookup runs against the directory holding this file, so it reports the
    repository the running code actually comes from. An installed copy that is
    not under version control, or a machine without git, simply has no hash.

    Returns
    -------
    str or None
        Full 40-character commit hash, or None if it cannot be determined.
    """
    global _GIT_HASH_CACHE
    if _GIT_HASH_CACHE is not False:
        return _GIT_HASH_CACHE

    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _GIT_HASH_CACHE = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        _GIT_HASH_CACHE = None
    return _GIT_HASH_CACHE


def provenance_fields():
    """
    The provenance keys to merge into a pickled dictionary.

    Returns
    -------
    dict
        ``{"imolcraft_version": str, "git_hash": str or None}``.
    """
    return {
        "imolcraft_version": __version__,
        "git_hash": get_git_hash(),
    }
