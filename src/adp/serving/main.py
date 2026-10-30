"""ASGI entry point: uvicorn adp.serving.main:app"""

import logging

from adp.serving.app import create_app

logging.basicConfig(level=logging.INFO, format="%(message)s")
app = create_app()
