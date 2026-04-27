import logging, psycopg2, os, hashlib
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

class PostgreConnector(BaseDBConnector):
    def __init__(self) -> None:
        self.DB_NAME = os.getenv("POSTGRES_DB_NAME")
        self.DB_USER = os.getenv("POSTGRES_DB_USER")
        self.DB_PASSWORD = os.getenv("POSTGRES_DB_PASSWORD")
        self.DB_HOST = os.getenv("POSTGRES_DB_HOST")
        self.DB_PORT = os.getenv("POSTGRES_DB_PORT")
        self.schemas = None
        self.schema_tables = None

    def get_all_schemas(self):
        """Fetch all non-system schemas from the database."""
        if self.schemas is not None:
            return self.schemas
        query = "SELECT nspname FROM pg_namespace;"
        result = self.execute_query(query)[1]
        schemas = [row[0] for row in result]
        self.schemas = schemas
        return self.schemas

    def get_schemas_with_tables(self, tmp_schemas=None):
        if self.schema_tables is not None:
            return self.schema_tables
        schemas = self.get_all_schemas()
        flag = False
        if tmp_schemas is not None:
            schemas = list(tmp_schemas.keys())
            flag = True
        engine = create_engine(
            f'postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}')
        connection = engine.connect()
        metadata = MetaData()
        result = {i: [] for i in schemas}
        for schema in schemas:
            # Reflect the tables
            metadata.reflect(bind=engine, schema=schema)

            for table in metadata.tables.values():
                if table.schema == schema:
                    if flag and len(tmp_schemas[schema]) != 0 and table.name in tmp_schemas[schema]:
                        result[schema].append(table.name)
                    else:
                        result[schema].append(table.name)
            metadata.clear()

        connection.close()
        self.schema_tables = result
        return result


    def get_create_table_statements(self, table_name: str = None):
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
        engine = create_engine(f'postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}')
        connection = engine.connect()
        metadata = MetaData()

        # Generate CREATE TABLE statements
        create_table_statements = []

        # Get all non-system schemas
        schemas = self.get_all_schemas()
        # print(tmp_schemas)
        for schema in schemas:
            # Reflect the tables
            metadata.reflect(bind=engine, schema=schema)
            # print(schema)
            for table in metadata.tables.values():
                if table.schema == schema:
                    if table_name is not None and table.name in table_name:
                        create_table_statements.append(str(CreateTable(table).compile(engine)))
                    elif table_name is None:
                        create_table_statements.append(str(CreateTable(table).compile(engine)))
            metadata.clear()
        
        connection.close()
        return create_table_statements

    def get_row_count(self, sql_query: str):
        sql_query = sql_query.replace(";", "")
        sql_query = config.remove_order_by(sql_query)
        try:
            count_query = f"SELECT COUNT(*) FROM ({sql_query}) AS subquery"
            conn = psycopg2.connect(
                dbname=self.DB_NAME,
                user=self.DB_USER,
                password=self.DB_PASSWORD,
                host=self.DB_HOST,
                port=self.DB_PORT
            )
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
        # Establish a connection to the PostgreSQL database
        THRESHOLD = config.ResponseType.ROW_THRESHOLD.value
        row_count, _ = self.get_row_count(config.strip_sql_comments(sql_query))
        logging.info(f"Row count of the SQL query: {row_count}")
        if row_count > THRESHOLD:
            return [], [], config.ResponseType.MSG_QUERY_TOO_MANY_ROWS.value, False

        conn = psycopg2.connect(
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT
        )
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
                # column_names = ["No. of Rows affected"]
                # result = cursor.rowcount
                logging.info("SQL query: ", sql_query)
                column_names = []
                result = []
                response = "Cannot run the SQL query generated, Please try a different query."
                status = False
            conn.commit()
        except Exception as e:
            conn.rollback()
            error_msg = "Error executing query: {}".format(e)
            # print("Error executing query: {}".format(e))
            logging.exception(error_msg)
            # raise e
            return [], [], "Unable to execute the SQL query. {}".format(error_msg), False
        finally:
            cursor.close()
            conn.close()
        
        return column_names, result, response, status

    def get_query_generation_instructions(self) -> str:
        return "Generate standard PostgreSQL SQL queries."

    def get_connection_id(self) -> str:
        """Generate unique connection ID for PostgreSQL database."""
        # Use host, port, database name (exclude user/password)
        connection_string = f"postgresql:{self.DB_HOST}:{self.DB_PORT}:{self.DB_NAME}"
        
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