## ----- DONE BY PRIYAM PAL -----

# DEPENDENCIES

import logging
import sys
from pathlib import Path
from datetime import datetime

class LoggerSetup:
    """
    A class to configure and manage logging setup for the project.
    
    """

    def __init__(self, logger_name: str = "default_logger", log_filename_prefix: str = "log"):
        """
        Initialize the logger setup.

        Arguments:
        ----------
        logger_name (str): Name of the logger (useful for multiple loggers).
        log_filename_prefix (str): Prefix for log file name.
        
        """
        
        self.log_dir       = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        log_file           = self.log_dir / f"{log_filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        FORMAT             = "[%(asctime)s %(filename)s->%(funcName)s():%(lineno)s]%(levelname)s: %(message)s"
        
        self.logger        = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            file_handler   = logging.FileHandler(log_file)
            stream_handler = logging.StreamHandler(sys.stdout)
            
            formatter      = logging.Formatter(FORMAT)
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)

    def get_logger(self):
        """
        Returns the configured logger instance.
        """
        
        return self.logger

    @staticmethod
    def log_info(logger, message):
        """
        Logs an info-level message.
        """
        
        logger.info(message)

    @staticmethod
    def log_warning(logger, message):
        """
        Logs a warning-level message.
        """
        
        logger.warning(message)

    @staticmethod
    def log_error(logger, message):
        """
        Logs an error-level message with exception info.
        """
        
        logger.error(message, exc_info=True)

    @staticmethod
    def log_debug(logger, message):
        """
        Logs a debug-level message.
        """
        
        logger.debug(message)
