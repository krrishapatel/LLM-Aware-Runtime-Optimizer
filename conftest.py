"""Present so that `import llm_optimizer` works under a bare `pytest` call.

pytest prepends the directory of the topmost conftest.py to sys.path. Without
this file only `python -m pytest` worked, because that form adds the current
directory and a bare `pytest` does not. That difference shows up first in CI.
"""
