import logging
import os
import sys
import tempfile
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

lib = None


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
            or '%(message)s' otherwise.
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
            level = os.environ["PPAFM_LOG_LEVEL"].upper()
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
            format = os.environ["PPAFM_LOG_FORMAT"]
        except KeyError:
            if _root_logger.level == logging.DEBUG:
                format = "[%(asctime)s - %(name)s - %(levelname)s] %(message)s"
            else:
                format = "%(message)s"
    _log_format = format
    _log_handler.setFormatter(logging.Formatter(fmt=_log_format))


def _init_logging():

    global lib
    if lib is not None:
        return

    configure_logging()

    # We have to do this import here, or otherwise we get a circular import
    from .cpp_utils import get_cdll

    lib = get_cdll("logging")
    lib.set_log_fd(sys.stdout.fileno())

    with CppLogger("test_logger"):
        lib.test_log()


def _get_lib():
    if lib is None:
        raise RuntimeError("Logging not initialized")
    return lib


class CppLogger:
    """Context manager for redirecting logging events in C++ to Python logging."""

    def __init__(self, logger_name: str):
        self.logger = get_logger(logger_name)
        self.lib = _get_lib()

    def __enter__(self):
        self.log = tempfile.TemporaryFile()
        self.lib.set_log_fd(self.log.fileno())

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.log.seek(0)

        # Read file line-by-line and convert messages to logging events
        log_level = logging.INFO
        msg = ""
        for line in self.log:
            line = line.decode().strip()
            if line.startswith("LOG_ENTRY_"):
                if msg:
                    self.logger.log(log_level, msg)
                    msg = ""
                line = line.removeprefix("LOG_ENTRY_")
                level_str, msg = line.split(" ", maxsplit=1)
                log_level = getattr(logging, level_str)
            else:
                msg += f"\n{line}"
        if msg:
            self.logger.log(log_level, msg)

        self.log.close()

        self.lib.set_log_fd(sys.stdout.fileno())


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
