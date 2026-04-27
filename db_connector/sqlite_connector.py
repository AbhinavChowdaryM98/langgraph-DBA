import logging, sqlite3, os
from db_connector import config
from sqlalchemy import create_engine, MetaData
from sqlalchemy.schema import CreateTable
from sqlalchemy.exc import SAWarning
import warnings

warnings.filterwarnings(
    "ignore",
    message="Did not recognize type",
    category=SAWarning
)

from .base_connector import BaseDBConnector

class SQLiteConnector(BaseDBConnector):
    def __init__(self) -> None:
        # Normalize path to handle both single and double backslashes
        self.DB_PATH = os.getenv("SQLITE_DB_PATH", "database.db")
        self.schemas = None
        self.schema_tables = None
        
        # Validate database file exists and is accessible
        if not os.path.exists(self.DB_PATH):
            raise FileNotFoundError(f"SQLite database file not found: {self.DB_PATH}")
        if not os.access(self.DB_PATH, os.R_OK):
            raise PermissionError(f"No read access to SQLite database file: {self.DB_PATH}")

    def get_all_schemas(self):
        """Fetch all schemas from the database (SQLite uses main schema)."""
        if self.schemas is not None:
            return self.schemas
        # SQLite has a main schema and temp schema
        schemas = ["main", "temp"]
        self.schemas = schemas
        return self.schemas

    def get_schemas_with_tables(self, tmp_schemas=None):
        """Get schemas with their tables."""
        if self.schema_tables is not None:
            return self.schema_tables
        
        schemas = self.get_all_schemas()
        flag = False
        if tmp_schemas is not None:
            schemas = list(tmp_schemas.keys())
            flag = True
        
        # Create engine for SQLite
        engine = create_engine(f'sqlite:///{self.DB_PATH}')
        connection = engine.connect()
        metadata = MetaData()
        result = {i: [] for i in schemas}
        
        for schema in schemas:
            try:
                # Reflect the tables
                if schema == "main":
                    metadata.reflect(bind=engine)
                else:
                    # For temp schema, we need to query sqlite_temp_master
                    temp_tables_query = "SELECT name FROM sqlite_temp_master WHERE type='table'"
                    temp_result = self.execute_query(temp_tables_query)
                    if temp_result[3]:  # status is True
                        for row in temp_result[1]:
                            result[schema].append(row[0])
                    continue

                for table in metadata.tables.values():
                    if flag and len(tmp_schemas[schema]) != 0 and table.name in tmp_schemas[schema]:
                        result[schema].append(table.name)
                    else:
                        result[schema].append(table.name)
                metadata.clear()
            except Exception as e:
                logging.warning(f"Error reflecting schema {schema}: {e}")
                result[schema] = []

        connection.close()
        self.schema_tables = result
        return result

    def get_create_table_statements(self, table_name: str = None):
        """Get CREATE TABLE statements for specified tables."""
        # Remove schema prefix from table names if present
        if isinstance(table_name, list):
            for i in range(len(table_name)):
                if "." in table_name[i]:
                    table_name[i] = table_name[i].split(".")[1]
        elif isinstance(table_name, str):
            if "." in table_name:
                table_name = table_name.split(".")[1]
            table_name = [table_name]
        
        # Connect to the database
        engine = create_engine(f'sqlite:///{self.DB_PATH}')
        connection = engine.connect()
        metadata = MetaData()

        # Generate CREATE TABLE statements
        create_table_statements = []

        # Get all schemas
        schemas = self.get_all_schemas()
        
        for schema in schemas:
            try:
                if schema == "main":
                    # Reflect the tables
                    metadata.reflect(bind=engine, schema=schema)
                    
                    for table in metadata.tables.values():
                        if table.schema == schema:
                            if table_name is not None and table.name in table_name:
                                create_table_statements.append(str(CreateTable(table).compile(engine)))
                            elif table_name is None:
                                create_table_statements.append(str(CreateTable(table).compile(engine)))
                    metadata.clear()
                else:
                    # For temp schema, query sqlite_temp_master directly
                    temp_query = "SELECT sql FROM sqlite_temp_master WHERE type='table' AND sql IS NOT NULL"
                    if table_name is not None:
                        # Format table names as strings for the IN clause
                        formatted_names = ",".join([f"'{name}'" for name in table_name])
                        temp_query += f" AND name IN ({formatted_names})"
                        temp_result = self.execute_query(temp_query)
                    else:
                        temp_result = self.execute_query(temp_query)
                    
                    if temp_result[3]:  # status is True
                        for row in temp_result[1]:
                            create_table_statements.append(row[0])
            except Exception as e:
                logging.warning(f"Error getting CREATE statements for schema {schema}: {e}")
        
        connection.close()
        return create_table_statements

    def get_row_count(self, sql_query: str):
        """Get row count for a query."""
        sql_query = sql_query.replace(";", "")
        sql_query = config.remove_order_by(sql_query)
        try:
            count_query = f"SELECT COUNT(*) FROM ({sql_query}) AS subquery"
            conn = sqlite3.connect(self.DB_PATH)
            cursor = conn.cursor()

            cursor.execute(count_query)
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return result[0][0], True
        except Exception as e:
            logging.exception("Issue while getting row count: " + str(e))
            return 0, False

    def execute_query(self, sql_query):
        """Execute a SQL query and return results."""
        THRESHOLD = config.ResponseType.ROW_THRESHOLD.value
        row_count, _ = self.get_row_count(config.strip_sql_comments(sql_query))
        logging.info(f"Row count of the SQL query: {row_count}")
        
        if row_count > THRESHOLD:
            return [], [], config.ResponseType.MSG_QUERY_TOO_MANY_ROWS.value, False

        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        
        try:
            # If it's a SELECT query, fetch the results
            if 'SELECT' in sql_query.strip().upper():
                cursor.execute(sql_query)
                result = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                response = "Your Data is ready! Check the 'Source Data' tab on the right to explore it. Click 'Start Visualization' to analyze and create visualizations, or 'Export Data' to download. Need a different dataset? Feel free to try again!"
                status = True
                if len(column_names) == 0 or len(result) == 0:
                    response = "The query executed successfully, but it appears to be there is no Data matching your criteria. Feel free to try a different query."
                    status = False
            else:
                # For other types of queries, return the number of affected rows
                logging.info("SQL query: ", sql_query)
                cursor.execute(sql_query)
                conn.commit()
                column_names = ["affected_rows"]
                result = [[cursor.rowcount]]
                response = f"Query executed successfully. {cursor.rowcount} rows affected."
                status = True
        except Exception as e:
            conn.rollback()
            error_msg = "Error executing query: {}".format(e)
            logging.exception(error_msg)
            return [], [], "Unable to execute the SQL query. {}".format(error_msg), False
        finally:
            cursor.close()
            conn.close()
        
        return column_names, result, response, status

    def get_query_generation_instructions(self) -> str:
        """Get query generation instructions for SQLite."""
        return "Generate standard SQLite SQL queries. Note: SQLite uses different syntax for some functions like date operations compared to PostgreSQL."

    def get_connection_id(self) -> str:
        """Generate unique connection ID for SQLite database."""
        # Use database path and schema structure for unique identification
        connection_string = f"sqlite:{self.DB_PATH}"
        
        # Add schema structure hash for uniqueness
        try:
            schemas_tables = self.get_schemas_with_tables()
            schema_str = str(sorted(schemas_tables.items()))
            schema_hash = hashlib.md5(schema_str.encode()).hexdigest()[:8]
        except:
            schema_hash = "unknown"
        
        # Create final connection ID
        connection_id = f"{hashlib.md5(connection_string.encode()).hexdigest()[:12]}_{schema_hash}"
        return connection_id
