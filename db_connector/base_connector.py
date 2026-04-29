from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import hashlib

class BaseDBConnector(ABC):
    @abstractmethod
    def get_schemas_with_tables(self, **kwargs) -> Dict[str, List[str]]:
        """
        Get all schemas and the tables inside them.
        Returns a dictionary where keys are schema names and values are lists of string table names.
        """
        pass

    @abstractmethod
    def get_create_table_statements(self, table_name: str = None, **kwargs) -> list[str]:
        """
        Get table definition with table name.
        Returns a list of string of table definitions for the specified table names.
        """
        pass

    @abstractmethod
    def execute_query(self, query: str, **kwargs) -> tuple[list[str], list[Any], str, bool]:
        """
        Execute the query given from input as argument.
        Returns:
            columns: list of column names
            result: list of result rows
            response: string response message
            status: boolean indicating success or failure
        """
        pass

    @abstractmethod
    def get_query_generation_instructions(self) -> str:
        """
        Get query generation instructions for this connector (e.g. SQL, T-SQL, DAX).
        """
        pass

    @abstractmethod
    def get_connection_id(self) -> str:
        """
        Generate a unique connection identifier for hint storage/retrieval.
        Should be based on database type, connection parameters, and schema structure.
        Excludes sensitive information like passwords.
        """
        pass

    @abstractmethod
    def get_sample_values(self, table_name: str, column_name: str, limit: int = 10) -> list:
        """
        Get sample values from a specific column to understand data patterns.
        Useful for understanding exact filter values, enum values, and data formats.

        Args:
            table_name: Name of the table
            column_name: Name of the column to sample
            limit: Maximum number of distinct values to return (default: 10)

        Returns:
            List of sample values as strings
        """
        pass
