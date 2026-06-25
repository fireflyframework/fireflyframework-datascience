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


def test_auto_config_wires_full_execution_config_into_executor() -> None:
    # The GenAI auto-config must not silently drop any execution control: timeout, sandbox AND
    # require_approval are threaded from config into the wired executor.
    from fireflyframework_datascience import FireflyDataScienceApplication
    from fireflyframework_datascience.core.config import FireflyDataScienceConfig
    from fireflyframework_datascience.features import FeatureEngineerPort

    config = FireflyDataScienceConfig()
    config.genai.enabled = True
    config.execution.require_approval = False  # explicit unattended opt-out
    config.execution.sandbox = "local"
    config.execution.timeout_seconds = 30
    app = FireflyDataScienceApplication.run(config=config, print_output=False)

    engineer = app.container.resolve_optional(FeatureEngineerPort)
    assert engineer is not None
    executor = engineer._executor  # type: ignore[attr-defined]
    assert executor._require_approval is False
    assert executor._sandbox == "local"
    assert executor._timeout_seconds == 30


def test_auto_config_default_is_fail_closed_hitl() -> None:
    # With the default execution config (require_approval=True) and no approver wired, the wired
    # executor fail-closes — the secure-by-default posture the docs describe.
    from fireflyframework_datascience import FireflyDataScienceApplication
    from fireflyframework_datascience.core.config import FireflyDataScienceConfig
    from fireflyframework_datascience.features import FeatureEngineerPort

    config = FireflyDataScienceConfig()
    config.genai.enabled = True
    app = FireflyDataScienceApplication.run(config=config, print_output=False)

    engineer = app.container.resolve_optional(FeatureEngineerPort)
    assert engineer is not None
    executor = engineer._executor  # type: ignore[attr-defined]
    assert executor._require_approval is True
    assert executor._approver is None  # no approver shipped → fail-closed until one is wired
    with pytest.raises(FeatureExecutionError, match="approval"):
        executor.execute("df['b'] = df['a'] + 1", pd.DataFrame({"a": [1, 2]}))
