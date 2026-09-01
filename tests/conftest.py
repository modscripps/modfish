import pathlib

import pytest


@pytest.fixture(scope="session")
def rootdir():
    return pathlib.Path(__file__).parent.resolve()
