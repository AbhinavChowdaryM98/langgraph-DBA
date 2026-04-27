"""
Common configuration file to avoid circular imports.
Contains database connector initialization and shared configurations.
"""

from db_connector import get_db_connector, DBType

# Initialize database connector - single instance for the application
db_connector = get_db_connector(DBType.POSTGRES)
