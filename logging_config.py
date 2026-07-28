# logging_config.py
import os
import logging
from logtail import LogtailHandler
from dotenv import load_dotenv
load_dotenv()
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    token = os.getenv("LOGTAIL_SOURCE_TOKEN")
    host = os.getenv("LOGTAIL_HOST")
    if token and host:
        handler = LogtailHandler(source_token=token, host=host)
        logging.getLogger().addHandler(handler)  # ROOT logger = catches all modules