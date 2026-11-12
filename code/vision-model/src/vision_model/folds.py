"""Group k-fold over capture sessions.

Each fold validates on a group of whole sessions and trains on every other
session except test, so every labeled session is used both to train and to
validate, and no session ever appears on both sides of a fold. The final model
is trained on everything except test.

``params.yaml``::

    split:
      test_sessions: ["..."]       # never in any fold
      untimed_to: train            # images without a session: always train
      folds:
        - ["20261109_184859"]
        - ["20261110_131211"]
        - rest                     # every session not named elsewhere
"""

from __future__ import annotations

from vision_model.sessions import UNTIMED_SESSION

REST = "rest"


def resolve_folds(folds_cfg: list, sessions: set[str], test: set[str]) -> list[set[str]]:
    """Validation sessions of each fold, with ``rest`` expanded.

    Raises ValueError on unknown sessions, a session in two folds or in a fold
    and test, more than one ``rest``, or an empty fold.
    """
    if not folds_cfg:
        raise ValueError("split.folds is empty: list at least two folds of sessions")
    named: list[set[str]] = []
    for fold in folds_cfg:
        if fold == REST:
            continue
        if isinstance(fold, str):
            raise ValueError(f"fold {fold!r}: a fold is a list of sessions, or 'rest'")
        named.append({str(s) for s in fold})
    if sum(1 for f in folds_cfg if f == REST) > 1:
        raise ValueError("only one fold can be 'rest'")

    listed = set().union(*named) if named else set()
    if unknown := listed - sessions:
        raise ValueError(f"sessions in split.folds not found in the data: {sorted(unknown)}")
    if in_test := listed & test:
        raise ValueError(f"sessions in both a fold and test: {sorted(in_test)}")
    if UNTIMED_SESSION in listed:
        raise ValueError(f"'{UNTIMED_SESSION}' images cannot be validated on (no session)")
    seen: set[str] = set()
    for fold in named:
        if repeated := fold & seen:
            raise ValueError(f"sessions in more than one fold: {sorted(repeated)}")
        seen |= fold

    rest = sessions - test - listed - {UNTIMED_SESSION}
    resolved = [rest if fold == REST else {str(s) for s in fold} for fold in folds_cfg]
    for i, fold in enumerate(resolved, 1):
        if not fold:
            raise ValueError(f"fold {i} has no sessions")
    return resolved


def fold_train_sessions(sessions: set[str], test: set[str], val: set[str]) -> set[str]:
    """Training sessions of a fold: everything except its val sessions and test."""
    return sessions - test - val
