"""
Logger module for training the private detector
"""
import datetime
import logging
import sys
from pathlib import Path


def make_logger(name: str,
                directory: str = 'logs') -> logging.Logger:
    """
    Create a logger that will also print to console

    Parameters
    ----------
    name: str
        String to tag the logs with
    directory: str
        Folder in which to save the logs

    Returns
    -------
    logger: logging.Logger
        The logger object
    """
    directory = Path(directory)
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

    today = datetime.datetime.today().strftime('%Y-%m-%d-%H%M')
    log_file_path = directory / f"{today}-{name}.log"

    logging.basicConfig(
        filename=log_file_path,
        level=logging.INFO
    )
    logger = logging.getLogger(name)

    # Also print log output to console
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    logger.info(f'Writing logs to {log_file_path}')

    return logger
