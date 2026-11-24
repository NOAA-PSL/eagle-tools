"""Auto-apply the tier marker (``unit``/``integration``) by directory, so tests can
be selected with ``-m unit`` / ``-m integration`` without decorating each one."""


def pytest_collection_modifyitems(items):
    for item in items:
        for tier in ("unit", "integration"):
            if tier in item.path.parts:
                item.add_marker(tier)
