# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:21:24 2018

@author: Lionel Massoulard
"""
from .__meta__ import __version__

import logging

logger = logging.getLogger("aikit")
if len(logger.handlers) == 0:  # To ensure reload() doesn't add another one
    logger.addHandler(logging.NullHandler())
