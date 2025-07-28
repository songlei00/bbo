# Copyright 2025 songlei
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import re

BBO_ROOT_LOGGER_NAME = 'bbo'
DEFAULT_LOG_LEVEL = logging.INFO


def get_logger(name: str, level: int = DEFAULT_LOG_LEVEL, force_name: bool = False) -> logging.Logger:
    # https://www.toptal.com/python/in-depth-python-logging
    if not force_name and not re.search(rf"^{BBO_ROOT_LOGGER_NAME}(\.|$)", name):
        name = f"{BBO_ROOT_LOGGER_NAME}.{name}"
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


def _build_formatter():
    return logging.Formatter(
        fmt="[%(levelname)s %(asctime)s %(filename)s:%(lineno)d] : %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def build_stream_handler(level: int = DEFAULT_LOG_LEVEL) -> logging.StreamHandler:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(_build_formatter())
    return stream_handler


def build_file_handler(filepath: str, level: int = DEFAULT_LOG_LEVEL, mode: str = 'w') -> logging.FileHandler:
    file_handler = logging.FileHandler(filepath, mode=mode)
    file_handler.setLevel(level)
    file_handler.setFormatter(_build_formatter())
    return file_handler


ROOT_LOGGER = get_logger(BBO_ROOT_LOGGER_NAME)
ROOT_LOGGER.propagate = False
ROOT_LOGGER.setLevel(logging.DEBUG)
ROOT_LOGGER.addHandler(build_stream_handler())