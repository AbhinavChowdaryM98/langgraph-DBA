import re
from azure.identity import ClientSecretCredential
import os
import time


FORBIDDEN_PATTERNS = [
    r"pip\s+install",
    r"os\s*\.\s*system",
    r"subprocess",
    r"importlib",
    r"open\s*\(",
    r"eval\s*\(",
    r"exec\s*\(",
    r"compile\s*\(",
    r"__import__",
    r"setattr",
    r"delattr",
    r"input\s*\("
]

def is_safe_code_pattern(code: str) -> bool:
    lowered = code.lower()
    return not any(re.search(pattern, lowered) for pattern in FORBIDDEN_PATTERNS)

FORBIDDEN_KEYWORDS = [
    "pip install", "os.system", "subprocess", "shutil", "open(", "importlib",
    "exec", "compile", "__import__", "input", "setattr", "delattr"
]

def is_safe_code(code: str) -> bool:
    return not any(word in code for word in FORBIDDEN_KEYWORDS)

# Token cache with TTL (55 minutes)
_token_cache = {}
_CACHE_TTL = 55 * 60  # 55 minutes in seconds


def generate_token(process_type="Fabric"):
    cache_key = f"{process_type}"
    current_time = time.time()
    
    # Check if cached token exists and is still valid
    if cache_key in _token_cache:
        cached_token, cached_time = _token_cache[cache_key]
        if current_time - cached_time < _CACHE_TTL:
            return cached_token
    
    # Generate new token
    cred = ClientSecretCredential(
        tenant_id=os.getenv("TENANT_ID"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET")
    )
    if process_type == "Fabric":
        token = cred.get_token("https://api.fabric.microsoft.com/.default") # For Fabric Rest APIs authentication
    elif process_type == "SQL":
        token = cred.get_token("https://database.windows.net/.default")
    elif process_type == "Livy Session":
        token = cred.get_token("https://analysis.windows.net/powerbi/api/.default")
    else:
        token = cred.get_token("https://storage.azure.com/.default") # For ODBC usable authentication
    
    # Cache the token with current timestamp
    _token_cache[cache_key] = (token.token, current_time)
    
    return token.token