import logging
import os
from datetime import datetime

def setup_logger(folder_path, name="safe_logger", log_level=logging.INFO):
    """
    Create and configure a logger that saves logs to the specified folder.
    
    Args:
        folder_path (str): Directory path where log files will be saved
        name (str): Name for the logger, used in the log file name
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        logger: Configured logging object
    """
    # Create the directory if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(folder_path, f"{name}_{timestamp}.log")
    
    # Configure the logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Logs will be saved to {log_file}")
    
    return logger

# Example usage:
# folder_path = os.path.join(args.extract_dir, name, "safe")
# logger = setup_logger(folder_path)
# logger.info("Starting process...")
# logger.error("An error occurred")
