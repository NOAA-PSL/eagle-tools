import pytest
import logging
from eagle.tools.log import SimpleFormatter, setup_simple_log


@pytest.mark.unit
class TestSimpleFormatter:
    """Tests for the SimpleFormatter class."""

    def test_simple_formatter_converts_to_seconds(self):
        """Test that SimpleFormatter converts relativeCreated to seconds."""
        formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)-7s] %(message)s")

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.relativeCreated = 5432.1  # milliseconds

        formatted = formatter.format(record)

        # Should be floor-divided by 1000, so 5432 // 1000 = 5
        assert "[5 s]" in formatted
        assert "[INFO   ]" in formatted
        assert "Test message" in formatted

    def test_simple_formatter_format_structure(self):
        """Test the format structure of SimpleFormatter."""
        formatter = SimpleFormatter(fmt="[%(relativeCreated)d s] [%(levelname)-7s] %(message)s")

        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Warning message",
            args=(),
            exc_info=None,
        )
        record.relativeCreated = 1234

        formatted = formatter.format(record)

        assert "[1 s]" in formatted
        assert "[WARNING]" in formatted
        assert "Warning message" in formatted


@pytest.mark.unit
class TestSetupSimpleLog:
    """Tests for the setup_simple_log function."""

    def test_setup_simple_log_default_level(self):
        """Test that setup_simple_log sets up logging with default INFO level."""
        # Import logger to check its state
        from eagle.tools.log import logger

        # Clear any existing handlers
        logger.handlers.clear()

        setup_simple_log()

        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

        # Check that handler is a StreamHandler
        handler = logger.handlers[-1]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.level == logging.INFO

        # Check that formatter is SimpleFormatter
        assert isinstance(handler.formatter, SimpleFormatter)

    def test_setup_simple_log_custom_level(self):
        """Test that setup_simple_log accepts custom log level."""
        from eagle.tools.log import logger

        # Clear any existing handlers
        logger.handlers.clear()

        setup_simple_log(level=logging.DEBUG)

        assert logger.level == logging.DEBUG

        handler = logger.handlers[-1]
        assert handler.level == logging.DEBUG

    def test_setup_simple_log_formatter_format(self):
        """Test that the formatter has the expected format string."""
        from eagle.tools.log import logger

        # Clear any existing handlers
        logger.handlers.clear()

        setup_simple_log()

        handler = logger.handlers[-1]
        formatter = handler.formatter

        assert isinstance(formatter, SimpleFormatter)
        assert "%(relativeCreated)d s" in formatter._fmt
        assert "%(levelname)-7s" in formatter._fmt
        assert "%(message)s" in formatter._fmt
