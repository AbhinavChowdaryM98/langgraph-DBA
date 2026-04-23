from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

class BaseDBConnector(ABC):
    @abstractmethod
    def get_schemas_with_tables(self, **kwargs) -> Dict[str, List[str]]:
        """
        Get all schemas and the tables inside them.
        Returns a dictionary where keys are schema names and values are lists of string table names.
        """
        pass

    @abstractmethod
    def get_create_table_statements(self, table_name: list[str] = None, **kwargs) -> list[str]:
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
