# CONSTANTS
from enum import Enum

class ResponseType(Enum):
    MSG_QUERY_NO_DATA = "No data found"
    MSG_QUERY_SUCCESS = "Query executed successfully"
    MSG_CONNECTION_ERROR = "Connection error"
    MSG_QUERY_EXECUTION_ERROR = "Query execution error"
    MSG_QUERY_TOO_MANY_ROWS = "The result set is larger than we can safely display at once, so we’ve limited the output. Please refine your request to narrow down the data."
    ROW_THRESHOLD = 100000
    TRUNCATION_ROW_LIMIT = 100000
    TRUNCATION_CELL_LIMIT = 1000000


# IMPORTS
import re
import time
from azure.identity import AzureCliCredential

# FUNCTIONS
_token_cache = {}

def generate_user_token(process_type="Fabric"):
    current_time = time.time()
    # Check cache first
    if process_type in _token_cache:
        cached_token, timestamp = _token_cache[process_type]
        if current_time - timestamp < 3540:
            return cached_token

    cred = AzureCliCredential()
    if process_type == "Fabric":
        token = cred.get_token("https://api.fabric.microsoft.com/.default") # For Fabric Rest APIs authentication
    elif process_type == "SQL":
        token = cred.get_token("https://database.windows.net/.default")
    elif process_type == "Livy Session":
        token = cred.get_token("https://analysis.windows.net/powerbi/api/.default")
    else:
        token = cred.get_token("https://storage.azure.com/.default") # For ODBC usable authentication

    _token_cache[process_type] = (token.token, current_time)
    return token.token

def strip_sql_comments(query: str) -> str:
    """Remove SQL comments before security validation"""
    query = re.sub(r'--.*$', '', query, flags=re.MULTILINE)
    query = re.sub(r'/\*[\s\S]*?\*/', '', query)
    return query

def remove_order_by(query: str) -> str:
    """
    Removes ORDER BY clause safely from end of query
    """
    # This removes ORDER BY only at the outermost level (simple version)
    pattern = r'ORDER\s+BY[\s\S]*$'
    return re.sub(pattern, '', query, flags=re.IGNORECASE).strip()