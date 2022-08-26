import sys
import logging
import argparse


LOG_FORMAT = 'DistDL-%(levelname)s: PID-%(process)d -- %(pathname)s::%(funcName)s -- %(message)s'

logging.basicConfig(format=LOG_FORMAT)
logging.logThreads = False
logging.logProcesses = True
logging.logMultiprocessing = True

parser = argparse.ArgumentParser()
parser.add_argument('-log', '--loglevel', default='error', 
                    choices=logging._nameToLevel.keys(), 
                    help="Provide logging level. Example --logleve debug, default=error")

args = parser.parse_args()

logger = logging.getLogger(name="DistDL-Logger")
# logger.setLevel(level=logging.INFO)
logger.setLevel(level=args.loglevel.upper())

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.ERROR)

# TODO: set path/to/logfile correctly
file_handler = logging.FileHandler('distdl.log')
file_handler.setLevel(logging.DEBUG)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

logger.info("Logger initialized.")
