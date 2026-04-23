from langchain.tools import tool
from db_connector import get_db_connector, DBType
import json
from app_utils import is_safe_code
import logging, matplotlib, plotly, base64, json, math


# Initialize database connector
db_connector = get_db_connector(DBType.POSTGRES)

@tool
def query_db(query: str) -> str:
    """Query the database.

    Args:
        query: SQL / DAX query to execute
    """
    columns, rows, msg, status = db_connector.execute_query(query)
    if not status:
        return json.dumps({"error": msg})
    return json.dumps({"columns": columns, "data": rows[:50]}, default=str)


@tool
def get_schemas_with_tables() -> str:
    """Get all schemas with their tables.

    Returns:
        JSON string with schemas and tables
    """
    schemas_with_tables = db_connector.get_schemas_with_tables()
    if not schemas_with_tables:
        return json.dumps({"error": "Failed to get schemas with tables"})
    return json.dumps(schemas_with_tables, default=str)


@tool
def get_create_table_statements(table_name: str) -> str:
    """Get the CREATE TABLE statement for a given table.

    Args:
        table_name: Name of the table
    """
    create_table_statement = db_connector.get_create_table_statements(table_name)
    if not create_table_statement:
        return json.dumps({"error": "Failed to get create table statement"})
    return json.dumps({"create_table_statement": create_table_statement}, default=str)


@tool
def python_code_execution(code: str) -> str:
    """Execute Python code for generating visualizations and data analysis.
    
    IMPORTANT: Only use this tool ONCE per request. If the code fails, do NOT retry with different approaches.
    
    Args:
        code: Python code to execute. The result must be stored in a variable called 'result' at the end.
              For matplotlib: Use plt.savefig() with BytesIO and base64 encode, return as HTML img tag: '<img src="data:image/png;base64,<base64_string>" />'
              For plotly: Use fig.to_html(include_plotlyjs='cdn') for interactive graphs, return the full HTML string
              For text: Return plain text or JSON string
    
    Returns:
        JSON string with 'result' field containing the output, or 'error' field if execution fails
    """
    if not is_safe_code(code):
        return json.dumps({"error": "Code contains forbidden patterns"})
    
    # Create isolated execution environment
    exec_globals = {
        '__builtins__': __builtins__,
        "matplotlib": matplotlib,
        "plotly": plotly,
        "base64": base64,
        "json": json,
        "math": math,
    }
    exec_locals = {}
    
    try:
        exec(code, globals(), exec_locals)
        result = exec_locals.get("result", "No result variable found")
        logging.info(f"result: {result}")
        return json.dumps({"result": result}, default=str)
    except ImportError as e:
        logging.error(f"Import error: {str(e)}. Available libraries: matplotlib, plotly, base64, json, math")
        return json.dumps({"error": f"Import error: {str(e)}. Available libraries: matplotlib, plotly, base64, json, math"})
    except Exception as e:
        logging.error(f"Execution error: {str(e)}")
        return json.dumps({"error": f"Execution error: {str(e)}"})