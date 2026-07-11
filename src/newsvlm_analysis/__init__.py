"""Analysis package for `newsvlm`.

This repo (`newspaper-analysis`) is intentionally split from the extraction/engine repo
(`newsvlm`) so that:
  - extraction code stays deployable + dependency-light
  - analysis code can freely depend on heavier scientific Python stacks
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
