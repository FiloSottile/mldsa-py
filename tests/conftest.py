"""Pytest configuration: run @pytest.mark.slow tests last."""


def pytest_collection_modifyitems(items):
    """Reorder so that tests marked ``slow`` run at the end."""
    slow = []
    rest = []
    for item in items:
        if item.get_closest_marker("slow"):
            slow.append(item)
        else:
            rest.append(item)
    items[:] = rest + slow
