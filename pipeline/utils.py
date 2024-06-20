import logging
from datetime import datetime
from pathlib import Path

from loguru import logger

tag = datetime.now().strftime("%b%d_%H%M%S")
PATH_ROOT = Path("logs") / f"{tag}"

PATH_ROOT.mkdir(parents=True, exist_ok=True)


def get_path_log():
    return (PATH_ROOT / "log").as_posix()


def get_path_conf():
    return (PATH_ROOT / "conf.yml").as_posix()


def get_path_dir_ckpt():
    return (PATH_ROOT / "ckpts").as_posix()


def get_path_tune_log():
    return (PATH_ROOT / "tune_result.json").as_posix()


def get_logger():
    fh = logging.FileHandler(get_path_log())

    logger.add(fh)
    logger.add(get_path_log())

    logging.basicConfig(handlers=[fh])

    return logger
