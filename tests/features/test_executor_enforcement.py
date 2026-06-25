# Copyright 2026 Firefly Software Foundation.
"""The executor must actually enforce execution.{timeout_seconds, require_approval, sandbox}.

These were declared config fields read by zero lines (an integrity gap). Real enforcement, real data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fireflyframework_datascience.features.executor import FeatureCodeExecutor, FeatureExecutionError


def test_executor_enforces_timeout() -> None:
    executor = FeatureCodeExecutor(timeout_seconds=1)
    with pytest.raises(FeatureExecutionError, match="timed out"):
        # an infinite loop never adds the column; the timeout must interrupt it
        executor.execute("df['x'] = 0\nwhile True:\n    pass", pd.DataFrame({"a": [1, 2, 3]}))


def test_executor_rejects_unimplemented_sandbox() -> None:
    # docker/e2b are not implemented — they must ERROR, not silently run in-process pretending isolation
    executor = FeatureCodeExecutor(sandbox="docker")
    with pytest.raises(FeatureExecutionError, match="docker"):
        executor.execute("df['b'] = df['a'] + 1", pd.DataFrame({"a": [1, 2]}))


def test_executor_in_process_sandboxes_still_run() -> None:
    for sandbox in ("local", "monty"):
        executor = FeatureCodeExecutor(sandbox=sandbox)
        out = executor.execute("df['b'] = df['a'] + 1", pd.DataFrame({"a": [1, 2]}))
        assert "b" in out.columns


def test_executor_enforces_approval() -> None:
    # denying approver -> rejected
    deny = FeatureCodeExecutor(require_approval=True, approver=lambda code: False)
    with pytest.raises(FeatureExecutionError, match="approval"):
        deny.execute("df['b'] = df['a'] + 1", pd.DataFrame({"a": [1, 2]}))

    # require_approval with NO approver -> fail closed
    no_approver = FeatureCodeExecutor(require_approval=True)
    with pytest.raises(FeatureExecutionError, match="approval"):
        no_approver.execute("df['b'] = df['a'] + 1", pd.DataFrame({"a": [1, 2]}))

    # approving approver -> runs
    allow = FeatureCodeExecutor(require_approval=True, approver=lambda code: True)
    out = allow.execute("df['b'] = df['a'] + 1", pd.DataFrame({"a": [1, 2]}))
    assert "b" in out.columns


def test_executor_defaults_unchanged() -> None:
    # default executor: no timeout, no approval, in-process — preserves existing behaviour
    out = FeatureCodeExecutor().execute("df['b'] = df['a'] * 2", pd.DataFrame({"a": [1, 2]}))
    assert list(out["b"]) == [2, 4]
