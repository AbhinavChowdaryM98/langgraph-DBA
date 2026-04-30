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
    
    def _get_connection(self):
        """Get a connection from pool if available, else open direct."""
        if hasattr(self, 'pool') and self.pool is not None:
            return self.pool.getconn()
        return psycopg2.connect(
            dbname=self.DB_NAME,
            user=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT
        )

    def _release_connection(self, conn):
        """Return connection to pool or close if direct."""
        if hasattr(self, 'pool') and self.pool is not None:
            self.pool.putconn(conn)
        else:
            conn.close()

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

    def get_row_count(self, sql_query: str) -> tuple[int, bool]:
        """Estimate row count for a query without fetching data.
        Strips semicolons and ORDER BY before wrapping in COUNT(*) subquery.
        Args:
            sql_query: SQL query string. Must be a SELECT query.
        Returns:
            Tuple of (row_count, success).
            On any failure: (0, False).
        """
        cleaned = config.remove_order_by(sql_query.replace(";", "").strip())
        count_query = f"SELECT COUNT(*) FROM ({cleaned}) AS _row_count_subquery"
        conn = None
        try:
            conn = self._get_connection()

            with conn.cursor() as cursor:
                cursor.execute(count_query)
                result = cursor.fetchone()

            conn.rollback()  # clean transaction state before pool return
            if result is None:
                logging.error("get_row_count: COUNT(*) returned no rows")
                return 0, False

            count = int(result[0])
            logging.debug("get_row_count: count = %d", count)
            return count, True

        except psycopg2.ProgrammingError as e:
            logging.error("get_row_count: programming error — bad SQL? %s", e)
            if conn:
                conn.rollback()
            return 0, False

        except psycopg2.OperationalError as e:
            logging.error("get_row_count: operational error: %s", e)
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return 0, False

        except Exception:
            logging.exception("get_row_count: unexpected error")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return 0, False

        finally:
            if conn is not None:
                self._release_connection(conn)

    def execute_query(self, sql_query: str) -> tuple[list, list, str, bool]:
        """Execute a SELECT query and return results.

        Args:
            sql_query: SQL query string to execute. Only SELECT queries return data;
                    non-SELECT queries are rejected.

        Returns:
            Tuple of (column_names, rows, message, status).
            On any failure: ([], [], error_message, False).
        """
        THRESHOLD = config.ResponseType.ROW_THRESHOLD.value
        stripped = config.strip_sql_comments(sql_query).strip()

        # Reject non-SELECT queries upfront — no connection needed
        if not stripped.upper().startswith("SELECT"):
            logging.warning("execute_query: rejected non-SELECT query: %.120s", stripped)
            return [], [], "Only SELECT queries are permitted.", False

        # Row count pre-check — uses pool internally via get_row_count
        try:
            row_count, count_ok = self.get_row_count(stripped)
        except Exception:
            logging.exception("execute_query: row count check failed")
            return [], [], "Unable to estimate row count before execution.", False

        if not count_ok:
            return [], [], "Unable to estimate row count before execution.", False

        logging.info("execute_query: estimated row count = %d", row_count)

        if row_count > THRESHOLD:
            return [], [], config.ResponseType.MSG_QUERY_TOO_MANY_ROWS.value, False

        if row_count == 0:
            return [], [], (
                "The query executed successfully, but no data matched your criteria. "
                "Feel free to try a different query."
            ), False

        conn = None
        try:
            conn = self._get_connection()

            with conn.cursor() as cursor:
                cursor.execute(sql_query)
                rows = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]

            # SELECT queries are read-only — no commit needed.
            # Explicit rollback cleans up any implicit transaction on the connection
            # before it goes back to the pool.
            conn.rollback()

            if not column_names or not rows:
                return [], [], (
                    "The query executed successfully, but no data matched your criteria. "
                    "Feel free to try a different query."
                ), False

            return column_names, rows, (
                "Your data is ready! Check the 'Source Data' tab to explore it."
            ), True

        except psycopg2.errors.QueryCanceled as e:
            logging.error("execute_query: query cancelled (timeout?): %s", e)
            if conn:
                conn.rollback()
            return [], [], f"Query was cancelled, possibly due to timeout: {e}", False

        except psycopg2.ProgrammingError as e:
            logging.error("execute_query: programming error: %s", e)
            if conn:
                conn.rollback()
            return [], [], f"SQL error — check query syntax or column names: {e}", False

        except psycopg2.OperationalError as e:
            logging.error("execute_query: operational error: %s", e)
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return [], [], f"Database connection error: {e}", False

        except Exception:
            logging.exception("execute_query: unexpected error")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            return [], [], "Unexpected error during query execution.", False

        finally:
            if conn is not None:
                self._release_connection(conn)

    def get_query_generation_instructions(self) -> str:
        return """
Generate standard SQLite SQL queries. Follow these rules strictly:

SYNTAX RULES:
- Column names with spaces: wrap in double quotes e.g. "Free Meal Count (K-12)"
- NO backticks, NO square brackets
- NO QUALIFY clause (not supported)
- NO CONCAT() — use || operator e.g. first_name || ' ' || last_name
- NO ISNULL() — use IS NULL / IS NOT NULL
- Date functions: use strftime('%Y', date_col) not YEAR()
- String functions: use INSTR() not CHARINDEX(), use SUBSTR() not SUBSTRING()
- Boolean: SQLite has no bool type, use 1/0 or 'Y'/'N' depending on column

ACCURACY RULES:
- Column names are case-sensitive — use exact casing from CREATE TABLE
- String filter values are case-sensitive — verify with get_sample_values before filtering
- When dividing integers, cast to REAL: CAST(numerator AS REAL) / denominator
- Always handle NULL in aggregations: use WHERE col IS NOT NULL when computing rates/ratios
- For LIMIT queries, always include ORDER BY to get deterministic results
- Never assume enum values — always check with get_sample_values first
"""

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

    def get_sample_values(self, table_name: str, column_name: str, limit: int = 10) -> list:
        """Get distinct non-null sample values from a column.
        Args:
            table_name: Table to sample from. Schema-qualified names (schema.table) are supported.
            column_name: Column to sample. Quotes are stripped and re-applied safely.
            limit: Max distinct values to return (default 10, capped at 100).

        Returns:
            List of string-coerced values, empty list on any failure.

        Raises:
            Nothing — all exceptions are caught and logged.
        """
        # Parse optional schema qualification
        if "." in table_name:
            schema_name, clean_table = table_name.split(".", 1)
            schema_name = schema_name.strip('"')
        else:
            schema_name = "public"
            clean_table = table_name

        clean_table = clean_table.strip('"')
        clean_column = column_name.strip('"')
        limit = min(max(1, limit), 100)  # clamp: 1–100
        conn = None
        try:
            conn = self._get_connection()  # use pool if available, else direct connect
            with conn.cursor() as cursor:
                # Validate table exists in information_schema
                cursor.execute(
                    """
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = %s AND table_name = %s
                    LIMIT 1
                    """,
                    (schema_name, clean_table)
                )
                if cursor.fetchone() is None:
                    logging.error(
                        "get_sample_values: table '%s.%s' not found",
                        schema_name, clean_table
                    )
                    return []

                # Validate column exists
                cursor.execute(
                    """
                    SELECT data_type FROM information_schema.columns
                    WHERE table_schema = %s AND table_name = %s AND column_name = %s
                    LIMIT 1
                    """,
                    (schema_name, clean_table, clean_column)
                )
                col_meta = cursor.fetchone()
                if col_meta is None:
                    # Fetch available columns for a useful error message
                    cursor.execute(
                        """
                        SELECT column_name FROM information_schema.columns
                        WHERE table_schema = %s AND table_name = %s
                        ORDER BY ordinal_position
                        """,
                        (schema_name, clean_table)
                    )
                    available = [r[0] for r in cursor.fetchall()]
                    logging.error(
                        "get_sample_values: column '%s' not found in '%s.%s'. Available: %s",
                        clean_column, schema_name, clean_table, available
                    )
                    return []

                # Identifiers cannot be parameterised in psycopg2 —
                # both names are validated against information_schema above.
                query = (
                    f'SELECT DISTINCT "{clean_column}" '
                    f'FROM "{schema_name}"."{clean_table}" '
                    f'WHERE "{clean_column}" IS NOT NULL '
                    f'LIMIT %s'
                )
                cursor.execute(query, (limit,))
                rows = cursor.fetchall()

            values = [str(row[0]) for row in rows]
            logging.debug(
                "get_sample_values: %d values from %s.%s.%s",
                len(values), schema_name, clean_table, clean_column
            )
            return values

        except psycopg2.OperationalError as e:
            logging.error(
                "get_sample_values: connection/operational error on %s.%s — %s",
                clean_table, clean_column, e
            )
            return []
        except psycopg2.ProgrammingError as e:
            logging.error(
                "get_sample_values: programming error on %s.%s — %s",
                clean_table, clean_column, e
            )
            return []
        except Exception:
            logging.exception(
                "get_sample_values: unexpected error on %s.%s",
                clean_table, clean_column
            )
            return []
        finally:
            if conn is not None:
                self._release_connection(conn)  # return to pool, don't close