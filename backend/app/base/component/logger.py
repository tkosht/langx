import logging.config
import re
from logging import getLogger


class Logger(object):
    def __init__(self, logger_name="app", conf_file="conf/logging.ini"):
        logging.config.fileConfig(conf_file)
        self.logger = getLogger(logger_name)

    def _format(self, *args, **kwargs):
        _msg = " ".join([str(elm) for elm in args])
        if kwargs:
            _msg += "[" + (" ".join([f"{k}={v}" for k, v in kwargs.items()])) + "]"
        _msg = re.sub(r"\r?\n", "\\n", _msg)
        return _msg

    def debug(self, msg, *args, **kwargs):
        _msg = self._format(msg, *args, **kwargs)
        self.logger.debug(_msg)

    def info(self, msg, *args, **kwargs):
        _msg = self._format(msg, *args, **kwargs)
        self.logger.info(_msg)

    # def warn(self, msg, *args: tuple=(), **kwargs: dict={}, e: Exception=None, stacktrace: str=""):
    def warning(self, msg, *args, **kwargs):
        _msg = self._format(msg, *args, **kwargs)
        self.logger.warning(_msg)

    def error(self, msg, *args, **kwargs):
        _msg = self._format(msg, *args, **kwargs)
        self.logger.error(_msg)

    def critical(self, msg, *args, **kwargs):
        _msg = self._format(msg, *args, **kwargs)
        self.logger.critical(_msg)
