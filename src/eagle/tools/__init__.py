try:
    from ._version import version as __version__
except ImportError:
    # This is a fallback for when the package is not yet installed,
    # for example when you are just importing the local code.
    __version__ = "unknown"
