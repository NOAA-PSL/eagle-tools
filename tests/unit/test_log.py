import logging

from eagle.tools.log import SimpleFormatter, logger, setup_simple_log


def make_record(level=logging.INFO, msg="hello", relative_ms=5432.1):
    record = logging.LogRecord("test", level, "test.py", 1, msg, (), None)
    record.relativeCreated = relative_ms
    return record


def test_formatter_reports_whole_seconds():
    formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)-7s] %(message)s")

    assert formatter.format(make_record()) == "[5 s] [INFO   ] hello"


def test_setup_simple_log_adds_a_formatted_handler(capsys):
    setup_simple_log(level=logging.DEBUG)
    handler = logger.handlers[-1]

    assert logger.level == logging.DEBUG
    assert handler.level == logging.DEBUG
    assert isinstance(handler.formatter, SimpleFormatter)

    logger.debug("from the test")
    assert "[DEBUG  ] from the test" in capsys.readouterr().out
