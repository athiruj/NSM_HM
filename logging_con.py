# Setting Logger #

import logging as log

  # basic Config
log.basicConfig(level=log.DEBUG, filename="base.log")

  # Create Logger Handler and Formatter
logger = log.getLogger(' ')
handler = log.StreamHandler() 
formatter = log.Formatter('%(asctime)s - %(levelname)s - %(message)s')

  # Set Formatter for Handler
handler.setFormatter(formatter)

  # Add Handler
logger.addHandler(handler)