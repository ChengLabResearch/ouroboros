"""Report a broken shared path once: at its canary, not in every test that crosses it.

Keep this module free of project and third-party imports. The focused memory CI
job collects under this directory with only a few dependencies installed.
"""
import pytest

_failed = pytest.StashKey[dict]()
_is_canary = pytest.StashKey[bool]()


def pytest_configure(config):
    config.stash[_failed] = {}
    config.addinivalue_line("markers", "canary(name, case=None): smallest test of a shared path; runs first and, "
                                       "if it fails, skips the rest of its cases and everything that needs it")
    config.addinivalue_line("markers", "needs(name): skip when the named canary has failed")


def pytest_collection_modifyitems(items):
    canaries, seen = [], set()
    for item in items:
        marker = item.get_closest_marker("canary")
        if marker and marker.args[0] not in seen and marker.kwargs.get("case", item.name) in item.name:
            seen.add(marker.args[0])
            item.stash[_is_canary] = True
            canaries.append(item)
    items[:] = canaries + [item for item in items if _is_canary not in item.stash]


def pytest_runtest_setup(item):
    failed = item.config.stash[_failed]
    names = [marker.args[0] for marker in item.iter_markers("needs")]
    if _is_canary not in item.stash:
        names += [marker.args[0] for marker in item.iter_markers("canary")]
    for name in names:
        if name in failed:
            pytest.skip(f"{name} is broken; fix {failed[name]} first")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item):
    report = yield
    if report.failed and _is_canary in item.stash:
        item.config.stash[_failed].setdefault(item.get_closest_marker("canary").args[0], item.nodeid)
    return report
