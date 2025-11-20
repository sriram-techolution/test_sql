import logging
import sys

# Configure logger
logger = logging.getLogger("app_logger")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Log format: Time | Function Name | Message
log_format = logging.Formatter("%(asctime)s | %(funcName)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
console_handler.setFormatter(log_format)

# Add handler to logger
logger.addHandler(console_handler)

logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("pymongo").setLevel(logging.WARNING)
logging.getLogger("msal").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)