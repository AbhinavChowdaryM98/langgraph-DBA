import base64
import datetime
import decimal
import hashlib

import pyodbc


from db_connector.config import remove_order_by, ResponseType


def sanitize_value(val):
    if isinstance(val, bytes):
        return base64.b64encode(val).decode("ascii")
    elif isinstance(val, (datetime.datetime, datetime.date)):
        return val.isoformat()
    elif isinstance(val, decimal.Decimal):
        return float(val)
    return val


class SqlServerConnector:
    def __init__(self, config=None) -> None:
        if config is None:
            config = {}
        self.DATABASE = config.get("MS_SQL_SERVER_DB_NAME")
        self.SERVER = config.get("MS_SQL_SERVER_HOST")
        self.USERNAME = config.get("MS_SQL_SERVER_USER")
        self.PASSWORD = config.get("MS_SQL_SERVER_PASSWORD")

        self.DRIVER = config.get("MS_SQL_SERVER_DRIVER")
        if self.DRIVER is None or self.DRIVER == "":
            self.DRIVER = "ODBC Driver 18 for SQL Server"

        if self.DATABASE is not None and self.DATABASE != "":
            self.connection_string = (
                f"DRIVER={{{self.DRIVER}}};"
                f"SERVER={self.SERVER};"
                f"DATABASE={self.DATABASE};"
                f"UID={self.USERNAME};"
                f"PWD={self.PASSWORD};"
                f"TrustServerCertificate=yes;"
            )
        else:
            self.connection_string = (
                f"DRIVER={{{self.DRIVER}}};"
                f"SERVER={self.SERVER};"
                f"UID={self.USERNAME};"
                f"PWD={self.PASSWORD};"
                f"TrustServerCertificate=yes;"
            )
        self.schema_cache = None
        self.schema_tables_cache = None
        

    def get_all_schemas(self):
        """Fetch all non-system schemas from the Microsoft SQL Server Database."""
        if self.schema_cache is not None:
            return self.schema_cache
        query = """
        SELECT name 
        FROM sys.schemas 
        WHERE name NOT IN ('sys', 'INFORMATION_SCHEMA', 'db_owner', 'db_accessadmin', 
                           'db_securityadmin', 'db_ddladmin', 'db_backupoperator', 
                           'db_datareader', 'db_datawriter', 'db_denydatareader', 
                           'db_denydatawriter')
        """
        result = self.execute_query(query, None, False)[1]
        schemas = [row[0] for row in result]
        self.schema_cache = schemas
        return schemas

    def get_schemas_with_tables(self):
        sql_query = """
                SELECT s.name AS schema_name, t.name AS table_name
                FROM sys.schemas s
                JOIN sys.tables t ON t.schema_id = s.schema_id
        """
        if self.schema_tables_cache is not None:
            return self.schema_tables_cache
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                result = cursor.fetchall()
                
            hash_schemas = {}
            for row in result:
                try:
                    hash_schemas[row[0]].append(row[1])
                except:
                    hash_schemas[row[0]] = [row[1]]
            self.schema_tables_cache = hash_schemas
            return hash_schemas
        except Exception as e:
            logger.exception("Failed to get hash schemas: " + str(e))
            return {}

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
        
        # Build SQL query
        sql_query = """
        SELECT s.name AS schema_name, t.name AS table_name, c.table_definition
        FROM sys.schemas s
        JOIN sys.tables t ON t.schema_id = s.schema_id
        CROSS APPLY (
            SELECT
                CAST(
                    'CREATE TABLE [' + s.name + '].[' + t.name + '] (' + CHAR(13) + CHAR(10) +
                    STRING_AGG(
                        CAST(
                            '    [' + c.name + '] ' +
                            CASE WHEN c.system_type_id IN (167, 175, 231, 239) THEN
                                CASE WHEN c.max_length = -1 THEN
                                    'VARCHAR(MAX)'
                                ELSE
                                    'VARCHAR(' + CAST(c.max_length AS NVARCHAR(5)) + ')'
                                END
                            ELSE
                                tp.name
                            END +
                            CASE WHEN c.is_nullable = 0 THEN ' NOT NULL' ELSE ' NULL' END
                        AS NVARCHAR(MAX)),
                        ',' + CHAR(13) + CHAR(10)
                    ) WITHIN GROUP (ORDER BY c.column_id) +
                    CHAR(13) + CHAR(10) + ');'
                AS NVARCHAR(MAX)
                ) AS table_definition
            FROM sys.columns c
            JOIN sys.types tp ON c.user_type_id = tp.user_type_id
            WHERE c.object_id = t.object_id
        ) c
        ORDER BY s.name, t.name;
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                result = cursor.fetchall()

            create_statements = []
            if not tmp_schemas or len(tmp_schemas.keys()) == 0:
                create_statements = [row[2] for row in result]
            else:
                for row in result:
                    if row[0] in tmp_schemas.keys():
                        if len(tmp_schemas[row[0]]) == 0:
                            create_statements.append(row[2])
                        elif row[1] in tmp_schemas[row[0]]:
                            create_statements.append(row[2])

            return create_statements
        except Exception as e:
            logger.exception("Failed to get create table statements: " + str(e))
            return []

    def get_row_count(self, sql_query: str, params: dict = None):
        sql_query = sql_query.replace(";", "")
        sql_query = remove_order_by(sql_query)
        count_query = f"SELECT COUNT(*) FROM ({sql_query}) AS subquery"
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                if params is not None:
                    cursor.execute(count_query, params)
                else:
                    cursor.execute(count_query)
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.exception("Failed to get row count: " + str(e))
            return 0

    def execute_query(self, sql_query: str):
        threshold = ResponseType.ROW_THRESHOLD.value
        row_count = self.get_row_count(sql_query)
        logger.info(f"Row count of the sql query: {row_count}")
        
        stream_to_file = row_count > threshold
        if row_count > threshold:
            logger.info(f"Row count exceeds threshold of {threshold}, streaming to file")
            return [], [], ResponseType.MSG_QUERY_TOO_MANY_ROWS.value, False, row_count

        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                try:
                    if "SELECT" in sql_query.strip().upper():
                        cursor.execute(sql_query)
                        result = cursor.fetchall()
                        column_names = [desc[0] for desc in cursor.description]
                        response = ResponseType.MSG_QUERY_SUCCESS.value
                        status = True
                        if len(column_names) == 0 or len(result) == 0:
                            response = ResponseType.MSG_QUERY_NO_DATA.value
                            status = False
                    else:
                        logger.info(f"SQL query: {sql_query}")
                        column_names = []
                        result = []
                        response = ResponseType.MSG_QUERY_EXECUTION_ERROR.value.format(str(e))
                        status = False
                    conn.commit()
                except Exception as e:
                    conn.rollback()
                    raise e
                finally:
                    cursor.close()

            cleaned_result = []
            for row in result:
                cleaned_row = [sanitize_value(col) for col in row]
                cleaned_result.append(cleaned_row)
            return column_names, cleaned_result, response, status, row_count

        except Exception as e:
            logger.error(f"Error executing SQL query on SqlServer connection {self.connection_string}: {e}")
            return [], [], ResponseType.MSG_QUERY_EXECUTION_ERROR.value.format(str(e)), False, 0

    def get_connection_id(self) -> str:
        """Generate unique connection ID for MS SQL Server database."""
        # Use server and database name (exclude username/password)
        connection_string = f"mssql:{self.SERVER}:{self.DATABASE}"
        
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