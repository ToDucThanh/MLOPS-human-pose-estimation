from typing import TypeVar

import loguru

from config import cfg

log_level = "DEBUG"
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | "
    "<level>{level: <8}</level> | "
    "<yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
)

_T_logoru_logger = TypeVar("_T_logoru_logger", bound=loguru._logger.Logger)


def logger_handler(
    use_log_file: bool = True, file: str = "./logs/logging_file.log"
) -> _T_logoru_logger:
    if use_log_file:
        loguru.logger.add(
            file,
            level=log_level,
            format=log_format,
            colorize=False,
            backtrace=True,
            diagnose=True,
        )
    return loguru.logger


log = logger_handler(file=cfg.logging_file)
