import os
from dotenv import load_dotenv

load_dotenv()

# --- Global Configuration ---
class AppConfig:
    APP_NAME = "Image Processor Pro"
    VERSION = "1.0.0"
    MAX_HISTORY = int(os.getenv("MAX_HISTORY", 20))  # Default to 20 if not set in .env
