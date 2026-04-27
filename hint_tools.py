import json
import logging
from typing import Dict, Any, List
from langchain.tools import tool
from hint_manager import hint_manager
from db_connector.factory import DBType
from common_config import db_connector

def _add_hint_internal(content: str, sql_query: str, schema_context: str = "", connection_id: str = None) -> Dict[str, Any]:
    """
    Internal function to add a hint (non-decorated, can be called directly).

    Args:
        content: The user's question or query description
        sql_query: The successful SQL query that was executed
        schema_context: Optional context about the schema/tables involved
        connection_id: Optional connection ID. If not provided, uses the current database connection ID.

    Returns:
        Dictionary with result status and message
    """
    try:
        # Use provided connection_id or get current one
        if connection_id is None:
            connection_id = db_connector.get_connection_id()

        # Add hint to vector store
        success = hint_manager.add_hint(
            connection_id=connection_id,
            content=content,
            sql_query=sql_query,
            schema_context=schema_context
        )

        if success:
            return {
                "status": "success",
                "message": f"Hint added successfully for connection {connection_id}",
                "connection_id": connection_id
            }
        else:
            return {
                "status": "error",
                "message": "Failed to add hint - check HintManager initialization"
            }
    except Exception as e:
        logging.error(f"Error in _add_hint_internal: {e}")
        return {
            "status": "error",
            "message": f"Error adding hint: {str(e)}"
        }

@tool
def add_hint(content: str, sql_query: str, schema_context: str = "", connection_id: str = None) -> str:
    """
    Add a successful SQL pattern as a hint for future reference.

    Use this tool when:
    - A SQL query executes successfully and provides useful results
    - The query represents a common pattern that might be useful for similar questions
    - You want to help the system learn from successful interactions

    Args:
        content: The user's question or query description
        sql_query: The successful SQL query that was executed
        schema_context: Optional context about the schema/tables involved
        connection_id: Optional connection ID. If not provided, uses the current database connection ID.

    Returns:
        JSON string with success status and message
    """
    result = _add_hint_internal(content, sql_query, schema_context, connection_id)
    return json.dumps(result, indent=2)

@tool
def get_hints(query: str, limit: int = 5, connection_id: str = None) -> str:
    """
    Retrieve relevant SQL hints based on the current query and database connection.

    Use this tool when:
    - You want to understand common SQL patterns for this database
    - You need context about how similar questions have been answered before
    - You want to learn from previous successful queries for this connection

    Args:
        query: The user's question or query description
        limit: Maximum number of hints to retrieve (default: 5)
        connection_id: Optional connection ID. If not provided, uses the current database connection ID.

    Returns:
        JSON string with relevant hints and their metadata
    """
    try:
        # Use provided connection_id or get current one
        if connection_id is None:
            connection_id = db_connector.get_connection_id()

        # Retrieve relevant hints
        hints = hint_manager.get_hints(
            connection_id=connection_id,
            query=query,
            limit=limit
        )

        if hints:
            result = {
                "status": "success",
                "connection_id": connection_id,
                "hints_found": len(hints),
                "hints": hints
            }
        else:
            result = {
                "status": "success",
                "connection_id": connection_id,
                "hints_found": 0,
                "message": "No relevant hints found for this query and connection",
                "hints": []
            }

        return json.dumps(result, indent=2)

    except Exception as e:
        logging.error(f"Error in get_hints tool: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error retrieving hints: {str(e)}"
        }, indent=2)

@tool
def get_connection_hint_stats() -> str:
    """
    Get statistics about stored hints for the current database connection.
    
    Use this tool when:
    - You want to know how many hints are available for this connection
    - You need to understand the learning progress for this database
    - You're troubleshooting hint-related issues
    
    Returns:
        JSON string with connection statistics
    """
    try:
        # Get current connection ID
        connection_id = db_connector.get_connection_id()
        
        # Get connection statistics
        stats = hint_manager.get_connection_stats(connection_id)
        
        if stats:
            result = {
                "status": "success",
                "connection_id": connection_id,
                "stats": stats
            }
        else:
            result = {
                "status": "success",
                "connection_id": connection_id,
                "stats": {
                    "total_hints": 0,
                    "message": "No hints found for this connection"
                }
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logging.error(f"Error in get_connection_hint_stats tool: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error getting connection stats: {str(e)}"
        }, indent=2)

@tool
def clear_connection_hints() -> str:
    """
    Clear all hints for the current database connection.
    
    Use this tool when:
    - You want to reset learning for this connection
    - The database schema has changed significantly
    - You're troubleshooting hint-related issues
    
    Returns:
        JSON string with operation status
    """
    try:
        # Get current connection ID
        connection_id = db_connector.get_connection_id()
        
        # Delete all hints for this connection
        success = hint_manager.delete_hints_by_connection(connection_id)
        
        if success:
            result = {
                "status": "success",
                "message": f"All hints cleared for connection {connection_id}",
                "connection_id": connection_id
            }
        else:
            result = {
                "status": "error",
                "message": "Failed to clear hints - check HintManager initialization"
            }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        logging.error(f"Error in clear_connection_hints tool: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error clearing hints: {str(e)}"
        }, indent=2)

@tool
def add_hint_simple(query: str, sql_query: str, execution_success: bool = True,
                   schema_context: str = "", connection_id: str = None) -> str:
    """
    Simply add a hint without complex logic.

    Use this tool when you've successfully solved a query and want to store the pattern.
    The orchestrator should decide when to call this based on:
    - Whether existing hints were helpful
    - How much extra effort was required
    - Complexity of the solution

    Args:
        query: The user's question or query description
        sql_query: The SQL query that was executed
        execution_success: Whether query executed successfully
        schema_context: Optional context about schema/tables involved
        connection_id: Optional connection ID. If not provided, uses the current database connection ID.

    Returns:
        JSON string with addition result
    """
    try:
        # Only add if query was successful
        if not execution_success:
            return json.dumps({
                "status": "skipped",
                "message": "Query execution failed - no hint added"
            }, indent=2)

        # Add hint using internal function
        add_result = _add_hint_internal(query, sql_query, schema_context, connection_id)

        result = {
            "status": "success",
            "action": "added",
            "message": "New hint added successfully",
            "add_result": add_result
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logging.error(f"Error in add_hint_simple tool: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Error in simple hint addition: {str(e)}"
        }, indent=2)

# Export tools for easy import
hint_tools = [add_hint, get_hints, get_connection_hint_stats, clear_connection_hints, add_hint_simple]
