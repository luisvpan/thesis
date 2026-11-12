import pytest

from vision_model.folds import REST, fold_train_sessions, resolve_folds
from vision_model.sessions import UNTIMED_SESSION

SESSIONS = {"a", "b", "old1", "old2", "t", UNTIMED_SESSION}


def test_rest_takes_unnamed_sessions_but_not_test_or_untimed():
    folds = resolve_folds([["a"], ["b"], REST], SESSIONS, test={"t"})
    assert folds == [{"a"}, {"b"}, {"old1", "old2"}]


def test_fold_train_excludes_its_val_and_test_but_keeps_untimed():
    train = fold_train_sessions(SESSIONS, test={"t"}, val={"a"})
    assert train == {"b", "old1", "old2", UNTIMED_SESSION}


@pytest.mark.parametrize(
    "folds, error",
    [
        ([["a"], ["a", "b"]], "more than one fold"),
        ([["a"], ["t"]], "fold and test"),
        ([["zzz"]], "not found"),
        ([REST, REST], "only one fold"),
        ([[UNTIMED_SESSION]], "cannot be validated"),
        (["a"], "list of sessions"),
        ([], "empty"),
    ],
)
def test_invalid_folds(folds, error):
    with pytest.raises(ValueError, match=error):
        resolve_folds(folds, SESSIONS, test={"t"})


def test_empty_rest_is_an_error():
    with pytest.raises(ValueError, match="no sessions"):
        resolve_folds([["a"], ["b", "old1", "old2"], REST], SESSIONS, test={"t"})


def test_session_ids_are_compared_as_strings():
    assert resolve_folds([[20261109]], {"20261109", "x"}, test=set()) == [{"20261109"}]
