""" Create logger and share it among the (small) project."""
import logging

# logging.basicConfig()
logger = logging.getLogger("rotational_motion")
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s")

logger.info(f"Logger is active.")
