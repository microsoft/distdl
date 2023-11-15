import logging
import os
import sys

LOG_FORMAT = 'DistDL-%(levelname)s: PID-%(process)d - %(pathname)s:%(lineno)d (%(funcName)s) - %(message)s'

logging.basicConfig(format=LOG_FORMAT)
logging.logThreads = False
logging.logProcesses = True
logging.logMultiprocessing = True


# Check if DISTDL_LOGLEVEL is set in the environment
if 'DISTDL_LOGLEVEL' in os.environ:
    loglevel = os.environ['DISTDL_LOGLEVEL'].upper()
else:
    loglevel = 'ERROR'

# Set log level
logger = logging.getLogger(name="DistDL-Logger")
logger.setLevel(level=loglevel)

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT))
stdout_handler.setLevel(logging.ERROR)

# TODO: set path/to/proper/logfile
file_handler = logging.FileHandler('distdl.log')
file_handler.setFormatter(logging.Formatter(fmt=LOG_FORMAT))
file_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger.info("Logger initialized.")
