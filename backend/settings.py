import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Configure logging
DEBUG_LOGGING = os.getenv("DEBUG_LOGGING", "false").lower() in ("true", "1", "yes")
LOG_LEVEL = logging.DEBUG if DEBUG_LOGGING else logging.INFO

# Configure the logger
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("podcast-rag")

# API Keys and Service Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
INDEX_NAME = os.getenv("INDEX_NAME", "podcast-clips-index")

# Development mode
DEV = os.getenv("DEV", "true").lower() in ("true", "1", "yes")

# API Base URL based on development mode
if DEV:
    API_BASE = f"http://127.0.0.1:8000"
else:
    API_BASE = "https://morphus-rag-chat.vercel.app"

# Webshare Proxy Configuration
WEBSHARE_PROXY_USERNAME = os.getenv("WEBSHARE_PROXY_USERNAME", "")
WEBSHARE_PROXY_PASSWORD = os.getenv("WEBSHARE_PROXY_PASSWORD", "")
USE_PROXIES = os.getenv("USE_PROXIES", "false").lower() in ("true", "1", "yes")
