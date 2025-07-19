# src/logger.py
# logger.py
import logging
import os
from datetime import datetime

# Create a log directory
LOG_FILE = f"{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"
log_path = os.path.join(os.getcwd(), "logs")
os.makedirs(log_path, exist_ok=True)

# Log file path
LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    )


if __name__ == "__main__":
    logging.info("This is a test log message")
    logging.error("This is a test error message")
    logging.warning("This is a test warning message")
    logging.debug("This is a test debug message")
    logging.critical("This is a test critical message")
