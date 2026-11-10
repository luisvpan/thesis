from cv_system import flags


def test_profile_every(monkeypatch):
    monkeypatch.delenv("CV_PROFILE", raising=False)
    assert flags.profile_every() == 0
    for value, expected in [("0", 0), ("off", 0), ("1", 60), ("true", 60), ("120", 120)]:
        monkeypatch.setenv("CV_PROFILE", value)
        assert flags.profile_every() == expected


def test_debug_enabled_with_old_name(monkeypatch):
    monkeypatch.delenv("CV_DEBUG", raising=False)
    monkeypatch.delenv("CV_DEBUG_WINDOWS", raising=False)
    assert not flags.debug_enabled()
    monkeypatch.setenv("CV_DEBUG_WINDOWS", "1")
    assert flags.debug_enabled()
    monkeypatch.setenv("CV_DEBUG", "0")  # the new name wins
    assert not flags.debug_enabled()
