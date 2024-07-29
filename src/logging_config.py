import logging
import os

from constants import LOG_DIR, LOG_FILENAME

# create a directory for the logs if one doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# track whether the logger is already configured so we don't do it again redundantly.
_LOGGER_CONFIGURED = False


def configure_logging() -> None:
    """
    Configure the logging module settings such as file output, output format, and log level.
    """
    global _LOGGER_CONFIGURED  # pylint: disable=global-statement

    if _LOGGER_CONFIGURED:
        return

    # default logging config for loggers
    logging.basicConfig(
        level=logging.INFO,  # default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[  # logs to the file and to the console
            logging.FileHandler(os.path.join(LOG_DIR, LOG_FILENAME)),
            logging.StreamHandler(),
        ],
    )
    _LOGGER_CONFIGURED = True
