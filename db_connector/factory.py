from enum import Enum
from .pg_connector import PostgreConnector
from .semantic_model_connector import SemanticModelConnector
from .sqlite_connector import SQLiteConnector

class DBType(Enum):
    POSTGRES = "Postgres"
    SEMANTIC_MODEL = "Semantic Model"
    SQLITE = "SQLite"


def get_db_connector(connector_type: str | DBType):
    """
    Factory function to return the appropriate DB connector instance based on the requested type.
    """
    if isinstance(connector_type, str):
        try:
            connector_type = DBType(connector_type)
        except ValueError:
            raise ValueError(f"Unsupported DB connector type: '{connector_type}'. "
                             f"Supported types are: {[t.value for t in DBType]}")

    if connector_type == DBType.POSTGRES:
        return PostgreConnector()
    elif connector_type == DBType.SEMANTIC_MODEL:
        return SemanticModelConnector()
    elif connector_type == DBType.SQLITE:
        return SQLiteConnector()
    else:
        raise ValueError(f"Unknown DB connector type: {connector_type}")

class DBConnectorFactory:
    """
    Class-based factory (optional use if an object-oriented approach is preferred).
    """
    @staticmethod
    def create(connector_type: str | DBType):
        return get_db_connector(connector_type)
