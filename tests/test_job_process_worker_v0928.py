from core.job_process_worker import ProcessJobExecutor, ProcessTaskSpec


def test_process_job_executes_in_spawned_process():
    result = ProcessJobExecutor().execute(
        ProcessTaskSpec("tests.process_job_fixtures", "add_numbers", (2, 5))
    )
    assert result == 7


def test_process_job_propagates_child_failure():
    try:
        ProcessJobExecutor().execute(ProcessTaskSpec("tests.process_job_fixtures", "fail_job"))
    except RuntimeError as exc:
        assert "fixture failure" in str(exc)
    else:
        raise AssertionError("child failure was not propagated")
