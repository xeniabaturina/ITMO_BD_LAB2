import logging
import os
import sys
from logging.handlers import RotatingFileHandler


class Logger:
    def __init__(self, show_log: bool = True) -> None:
        self.logger = None
        self.show_log = show_log
        self.logs_path = os.path.join(os.getcwd(), "logs")

        os.makedirs(self.logs_path, exist_ok=True)

    def get_file_handler(self) -> logging.FileHandler:
        """
        Create a file handler for logging to a file.

        Returns:
            logging.FileHandler: File handler for logging.
        """
        try:
            file_handler = RotatingFileHandler(
                os.path.join(self.logs_path, "app.log"),
                maxBytes=1024 * 1024,
                backupCount=10,
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            return file_handler
        except Exception as e:
            print(f"Error creating file handler: {e}")
            sys.exit(1)

    def get_stream_handler(self) -> logging.StreamHandler:
        """
        Create a stream handler for logging to console.

        Returns:
            logging.StreamHandler: Stream handler for logging.
        """
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        return stream_handler

    def get_logger(self, name: str) -> logging.Logger:
        """
        Create and configure logger.

        Args:
            name (str): Name of the logger.

        Returns:
            logging.Logger: Configured logger instance.
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.get_file_handler())
        if self.show_log:
            logger.addHandler(self.get_stream_handler())
        logger.propagate = False
        return logger
