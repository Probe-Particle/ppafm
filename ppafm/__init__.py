#!/usr/bin/python

from .common import *
from .logging_utils import configure_logging
from .version import __version__

# This sets the logging level and other options immediately from environment variables as the ppafm package is imported.
configure_logging()
