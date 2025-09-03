from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("eagle.tools")
except PackageNotFoundError:
    pass
