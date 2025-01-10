import pytest
from chat import app


def pytest_configure():
    """
    Load nanodjango project.
    """
    app._prepare(is_prod=False)


@pytest.fixture(autouse=True)
def _set_settings(settings):
    settings.CHAT_MODEL = "fake"
