import sys
import logging
import argparse

LOG_FORMAT = 'DistDL-%(levelname)s: PID-%(process)d - %(pathname)s:%(lineno)d (%(funcName)s) - %(message)s'

logging.basicConfig(format=LOG_FORMAT)
logging.logThreads = False
logging.logProcesses = True
logging.logMultiprocessing = True

parser = argparse.ArgumentParser()
parser.add_argument('-log', '--loglevel', default='ERROR', 
                    choices=logging._nameToLevel.keys(), 
                    help="Provide logging level. Example --loglevel DEBUG, default=ERROR")
#try:
    #args = parser.parse_args()
#except:
# TODO fix logger setup. current solution leads to errors with pytest
args = parser.parse_args([])

logger = logging.getLogger(name="DistDL-Logger")
logger.setLevel(level=args.loglevel.upper())

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
