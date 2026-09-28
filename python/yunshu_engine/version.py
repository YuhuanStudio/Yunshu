"""The installed Yunshu version (from package metadata; one source: pyproject)."""

from importlib.metadata import PackageNotFoundError, version


def yunshu_version() -> str:
    try:
        return version("yunshu")
    except PackageNotFoundError:
        return "unknown"
