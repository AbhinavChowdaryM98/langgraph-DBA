import asyncio, logging, json, base64, requests, re, os, time, hashlib
from db_connector.config import generate_user_token, ResponseType

def check_truncation(row_count, col_count):
    # Detect truncation
    cell_count = row_count * col_count

    return row_count >= ResponseType.TRUNCATION_ROW_LIMIT.value or cell_count >= ResponseType.TRUNCATION_CELL_LIMIT.value

def _get_definition_fabric_native(probe_response, headers):
    """
    Handle Fabric-native semantic model definition retrieval.
    probe_response is the initial POST response (200 or 202).
    """
    # ── Case 1: Immediate 200 response ────────────────────────────────────
    if probe_response.status_code == 200:
        result = probe_response.json()

    # ── Case 2: 202 - Long running operation, poll until done ─────────────
    elif probe_response.status_code == 202:
        tracking_url = probe_response.headers.get("Location")
        operation_id = probe_response.headers.get("x-ms-operation-id")
        retry_after = int(probe_response.headers.get("Retry-After", 5))
        if not tracking_url and operation_id:
            tracking_url = f"https://api.fabric.microsoft.com/v1/operations/{operation_id}"
        if not tracking_url:
            logging.error(
                f"202 received but no Location or x-ms-operation-id found. "
                f"Headers: {dict(probe_response.headers)}"
            )
            return [], []
        logging.info(f"Long running operation started. Polling: {tracking_url}")
        # ── Poll until Succeeded ───────────────────────────────────────────
        max_attempts = 20
        for attempt in range(1, max_attempts + 1):
            time.sleep(retry_after)
            status_check = requests.get(tracking_url, headers=headers)
            if status_check.status_code == 202:
                retry_after = int(status_check.headers.get("Retry-After", retry_after))
                logging.info(f"Still processing (attempt {attempt}), retrying in {retry_after}s...")
                continue

            if status_check.status_code == 200:
                status_body = status_check.json()
                status = status_body.get("status")
                if status == "Succeeded":
                    logging.info("Operation succeeded, fetching result...")
                    break
                elif status == "Failed":
                    logging.error(f"Operation failed: {status_body.get('error')}")
                    return []
                else:
                    logging.info(f"Operation status: {status} (attempt {attempt})")
                    continue
            else:
                logging.error(f"Unexpected poll status {status_check.status_code}: {status_check.text}")
                return []
        else:
            logging.error(f"Operation did not complete after {max_attempts} attempts")
            return []
        # ── Fetch actual result from /result endpoint ──────────────────────
        result_response = requests.get(tracking_url + "/result", headers=headers)
        if result_response.status_code != 200:
            logging.error(f"Failed to fetch result: {result_response.status_code} - {result_response.text}")
            return []
        result = result_response.json()

    else:
        logging.error(f"Unexpected initial status {probe_response.status_code}: {probe_response.text}")
        return []

    # ── Extract model.bim payload from definition parts ───────────────────
    try:
        parts = result.get("definition", {}).get("parts", [])
        bim_payload = next(
            (p["payload"] for p in parts if p.get("path") == "model.bim"),
            None
        )
        if not bim_payload:
            logging.error(
                f"model.bim not found in definition parts. Available paths: {[p.get('path') for p in parts]}")
            return []
        # ── Decode Base64 → JSON ───────────────────────────────────────────
        model_content = json.loads(base64.b64decode(bim_payload).decode("utf-8"))

    except Exception as e:
        logging.error(f"Failed to decode model.bim payload: {e}", exc_info=True)
        return []
    return model_content

def extract_semantic_model_structure(model_json: dict, table_name: list[str] = None) -> list:
    model = model_json.get("model", {})
    tables = model.get("tables", [])
    relationships = model.get("relationships", [])
    # 🔹 Build relationship map (from_table → list of relations)
    rel_map = {}
    for rel in relationships:
        from_table = rel.get("fromTable")
        rel_str = f"{rel.get('fromColumn')} -> {rel.get('toTable')}.{rel.get('toColumn')}"
        rel_map.setdefault(from_table, []).append(rel_str)

    result = []
    for table in tables:
        t_name = table.get("name")
        if table_name is not None and t_name not in table_name:
            continue
        # 🔹 Columns
        columns = []
        for col in table.get("columns", []):
            col_name = col.get("name")
            dtype = col.get("dataType")
            columns.append(f"{col_name} ({dtype})")

        # 🔹 Relationships (only outgoing for now)
        table_relationships = rel_map.get(t_name, [])
        # 🔹 Optional: Measures
        measures = []
        for measure in table.get("measures", []):
            measures.append(f"{measure.get('name')} = {measure.get('expression')}")

        result.append(json.dumps({
            "table": t_name,
            "columns": columns,
            "relationships": table_relationships,
            "measures": measures
        }))

    return result

from .base_connector import BaseDBConnector

class SemanticModelConnector(BaseDBConnector):
    def __init__(self):
        self.db_name = os.getenv("SEMANTIC_DB_NAME")
        self.db_id = os.getenv("SEMANTIC_DB_ID")
        self.workspace_id = os.getenv("SEMANTIC_WORKSPACE_ID")
        self.model_content = None
        self.db_host = f"https://api.powerbi.com/v1.0/myorg/groups/{self.workspace_id}/datasets/{self.db_id}/executeQueries"
        self.schemas = None
        self.schema_tables = None

    def get_all_schemas(self, schemas: dict = {}, token: str = None):
        if self.schemas is not None:
            return self.schemas
        if len(schemas.keys()):
            return list(schemas.keys())
        return ["dbo"]

    def get_schemas_with_tables(self, token: str = generate_user_token("Fabric")):

        if self.schema_tables is not None:
            return self.schema_tables
        
        if self.model_content is None:
            if not token.startswith("Bearer"):
                token = "Bearer " + token
            headers = {"Authorization": token}
            probe_url = (
                f"https://api.fabric.microsoft.com/v1/workspaces/{self.workspace_id}"
                f"/semanticModels/{self.db_id}/getDefinition?format=TMSL"
            )
            probe = requests.post(probe_url, headers=headers)
            self.model_content = _get_definition_fabric_native(probe, headers)

        tables = self.model_content.get("model", {}).get("tables", [])
        if len(tables) > 0:
            self.schema_tables = {"dbo": [i["name"] for i in tables]}
            return self.schema_tables

        self.schema_tables = {"dbo": []}
        return self.schema_tables

    def get_create_table_statements(self, table_name: str = None, token: str = generate_user_token("Fabric")):
        """Route to correct API based on whether model is Fabric-native or legacy."""
        if not token.startswith("Bearer"):
            token = "Bearer " + token
        headers = {"Authorization": token}

        # Remove schema prefix from table names if present
        if isinstance(table_name, list):
            for i in range(len(table_name)):
                if "." in table_name[i]:
                    table_name[i] = table_name[i].split(".")[1]
        elif isinstance(table_name, str):
            if "." in table_name:
                table_name = table_name.split(".")[1]
            table_name = [table_name]

        # ── Check if it's a Fabric-native model by probing getDefinition ──────
        try:
            if self.model_content is None:
                probe_url = (
                    f"https://api.fabric.microsoft.com/v1/workspaces/{self.workspace_id}"
                    f"/semanticModels/{self.db_id}/getDefinition?format=TMSL"
                )
                probe = requests.post(probe_url, headers=headers)
                # Fabric-native model — use getDefinition flow
                logging.info(f"Fabric-native model detected, using getDefinition. Status: {probe.status_code}")
                self.model_content = _get_definition_fabric_native(probe, headers)

            return extract_semantic_model_structure(self.model_content, table_name)
        except Exception as e:
            logging.info(f"getDefinition failed, falling back to Power BI REST API")
            return []

    def get_row_count(self, query: str, token: str = generate_user_token("Livy Session")) -> int:
        # Remove ORDER BY (important)
        query = re.sub(r"ORDER\s+BY[\s\S]*$", "", query, flags=re.IGNORECASE)
        # Ensure EVALUATE
        if not query.strip().upper().startswith("EVALUATE"):
            query = f"EVALUATE {query}"

        count_query = f"""
        EVALUATE
        ROW(
            "row_count",
            COUNTROWS(
                (
                    {query.replace("EVALUATE", "", 1)}
                )
            )
        )
        """
        if not token.startswith("Bearer"):
            token = "Bearer " + token
        headers = {
            "Content-Type": "application/json",
            "Authorization": token
        }
        body = {
            "queries": [{"query": count_query}],
            "serializerSettings": {"includeNulls": True}
        }
        response = requests.post(self.db_host, headers=headers, json=body)
        if response.status_code != 200:
            raise Exception(f"Error {response.status_code}: {response.text}")

        data = response.json()
        return data["results"][0]["tables"][0]["rows"][0]['[row_count]']

    def execute_query(self, query, token: str = generate_user_token("Livy Session")):
        # Note, check if in the above URL myorg should be left like that or should i get the user org?
        if not token.startswith("Bearer"):
            token = "Bearer " + token
        headers = {
            "Content-Type": "application/json",
            "Authorization": token
        }
        # The API supports one query per request in the 'queries' list
        body = {
            "queries": [{"query": query}],
            "serializerSettings": {"includeNulls": True}
        }
        response = requests.post(self.db_host, headers=headers, json=body)
        if response.status_code == 200:
            result = response.json()
            result = result['results'][0]['tables'][0]
            raw_result = result.get('rows', [])
            if len(raw_result) == 0:
                return [], [], "The query executed successfully, but no data matched your criteria.", False

            columns = raw_result[0].keys()
            try:
                rows = [[i[j] for j in columns] for i in raw_result]
            except Exception as e:
                error_msg = f"Error executing query: {e}"
                logging.exception(error_msg)
                return [], [], f"Couldn't connect to the provided Semantic model to execute the DAX query. Error: {error_msg}", False
            columns = [i[1:-1] if i.startswith('[') and i.endswith(']') else i for i in columns]
            if check_truncation(len(rows), len(columns)):
                return columns, rows, "The query executed successfully, Query result may be truncated due to Fabric API limits. Consider applying filters or aggregations.", True
            return columns, rows, "Data extraction is complete! Start building your visualizations.", True
        else:
            error_msg = f"Error {response.status_code}: {response.text}"
            logging.exception(error_msg)
            return [], [], f"Couldn't connect to the provided Semantic model to execute the DAX query. Error: {error_msg}", False

    def get_query_generation_instructions(self) -> str:
        return "Generate Microsoft DAX (Data Analysis Expressions) queries for this semantic model connection. Note that the semantic model is connected to a Fabric workspace, so you may need to use Fabric-specific DAX functions. **Note that this connection can't execute any type of SQL queries, Only DAX Queries.**"

    def get_connection_id(self) -> str:
        """Generate unique connection ID for Semantic Model database."""
        # Use database name and ID (exclude workspace ID for security)
        connection_string = f"semantic:{self.db_name}:{self.db_id}"
        
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