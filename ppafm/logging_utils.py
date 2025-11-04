import logging
import os
import sys
from typing import Any, Optional, Union

# Setup the root logger for the ppafm package. All other loggers will be derived from this one.
_root_logger = logging.getLogger("ppafm")
_log_format = "%(message)s"
_log_handler = None
_log_path = None

# Setup another logger for performance benchmarking
_perf_logger = logging.getLogger("ppafm.perf")
_perf_logger.propagate = False
_perf_log_format = "[%(asctime)s - %(name)s] %(message)s"
_perf_log_enabled = False


class _DefaultFormatter(logging.Formatter):

    def __init__(self):
        super().__init__()
        self.info_formatter = logging.Formatter("%(message)s")
        self.other_formatter = logging.Formatter("%(levelname)s: %(message)s")

    def format(self, record):
        if record.levelno == logging.INFO:
            formatter = self.info_formatter
        else:
            formatter = self.other_formatter
        return formatter.format(record)


def configure_logging(
    level: Optional[Union[int, str]] = None,
    format: Optional[str] = None,
    log_path: Optional[str] = None,
    log_performance: Optional[bool] = None,
):
    """Configure options for the ppafm logger.

    Arguments:
        level: Logging level to use. See https://docs.python.org/3/library/logging.html#levels.
            If None, the level is determined from the PPAFM_LOG_LEVEL environment variable or
            defaults to logging.INFO.
        format: Format to use for logging messages. See https://docs.python.org/3/library/logging.html#logrecord-attributes.
            If None, the format is determined from the PPAFM_LOG_FORMAT environment variable or
            defaults to "[%(asctime)s - %(name)s - %(levelname)s] %(message)s" if logging level is DEBUG,
            or otherwise '%(message)s' for INFO and `%(levelname): %(message)s` for more severe levels.
        log_path: Path where log will be written. If None, the path is determined from the the PPAFM_LOG_PATH
            environment variable, or defaults to stdout.
        log_performance: Whether to enable performance logging. If None, is True if the PPAFM_LOG_PERFORMANCE
            environment variable is set, otherwise False.
    """
    global _root_logger
    global _log_handler
    global _log_format
    global _log_path
    global _perf_log_enabled

    if level is None:
        try:
            level = os.environ["PPAFM_LOG_LEVEL"]
        except KeyError:
            level = logging.INFO
    _root_logger.setLevel(level)

    if log_path is None:
        try:
            log_path = os.environ["PPAFM_LOG_PATH"]
        except KeyError:
            pass
    _log_path = log_path

    if _log_path is None:
        _log_handler = logging.StreamHandler(sys.stdout)
    else:
        _log_handler = logging.FileHandler(_log_path)

    for handler in _root_logger.handlers:
        _root_logger.removeHandler(handler)
    _root_logger.addHandler(_log_handler)

    if log_performance is None:
        log_performance = "PPAFM_LOG_PERFORMANCE" in os.environ
    if log_performance:
        if _log_path is None:
            perf_log_handler = logging.StreamHandler(sys.stdout)
        else:
            perf_log_handler = logging.FileHandler(_log_path)
        perf_log_handler.setFormatter(logging.Formatter(fmt=_perf_log_format))
        for handler in _perf_logger.handlers:
            _perf_logger.removeHandler(handler)
        _perf_logger.addHandler(perf_log_handler)
        _perf_logger.setLevel(logging.INFO)
        _perf_log_enabled = True
    else:
        _perf_logger.setLevel(logging.CRITICAL)
        _perf_log_enabled = False

    if format is None:
        try:
            _log_format = os.environ["PPAFM_LOG_FORMAT"]
            formatter = logging.Formatter(fmt=_log_format)
        except KeyError:
            if _root_logger.level == logging.DEBUG:
                _log_format = "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
                formatter = logging.Formatter(fmt=_log_format)
            else:
                _log_format = "%(message)s"
                formatter = _DefaultFormatter()
    else:
        _log_format = format
        formatter = logging.Formatter(fmt=_log_format)
    _log_handler.setFormatter(formatter)


# This sets the logging level and other options immediately from environment variables.
configure_logging()


def get_logger(name: str) -> logging.Logger:
    return _root_logger.getChild(name)


def get_perf_logger(name: str) -> logging.Logger:
    return _perf_logger.getChild(name)


def perf_log_enabled() -> bool:
    return _perf_log_enabled


class ProgressLogger:
    """Print gradual progress messages."""

    def __init__(self, logger_name: str = "progress", pre_message: str = ""):

        self.logger = get_logger(logger_name)
        self.logger.propagate = False  # So that the parent handlers don't also print stuff

        # Remove any existing handlers if this logger name was used before
        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        if _log_path is None:
            # We are printing to terminal. Setup a handler that prints all message to the same line.
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter("\r" + _log_format)  # Carriage return deletes the previous message
            handler.terminator = ""  # Don't print a new line
            self._percent_increment = 1
            self._print_to_terminal = True
        else:
            # We are printing to a file. Print normally to subsequent lines.
            handler = logging.FileHandler(_log_path)
            formatter = logging.Formatter(_log_format)
            self._percent_increment = 20  # Don't print as often, so we don't spam the log file too much
            self._print_to_terminal = False
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.pre_message = pre_message
        self._previous_percent = -self._percent_increment

    def print_percent(self, block_num: int, block_size: int, total_size: int):
        if total_size == -1:
            return
        current_size = block_num * block_size
        percent = int(current_size / total_size * 100)
        if percent < (self._previous_percent + self._percent_increment):
            return
        self._previous_percent = percent
        if current_size < total_size:
            self.logger.info(f"{self.pre_message}{percent:2d}%")
        else:
            msg = f"{self.pre_message}Done"
            if self._print_to_terminal:
                msg += "\n"
            self.logger.info(msg)

    def print_message(self, message: Any, is_last: bool):
        message = f"{self.pre_message}{message}"
        if is_last and self._print_to_terminal:
            message += "\n"
        self.logger.info(message)
