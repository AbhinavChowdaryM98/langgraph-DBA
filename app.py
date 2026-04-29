from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import json
from dotenv import load_dotenv
import os
import logging
import warnings

load_dotenv()

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*__path__.*")

from tool_definitions import query_db, get_schemas_with_tables, get_create_table_statements, python_code_execution
from common_config import db_connector
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from agent_common import (
    MessagesState, model_init, llm_call, tool_node, 
    should_continue, create_agent, llm_call_sql_only, create_sql_only_agent
)

app = FastAPI(title="Database Query Assistant API")

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    provider: str = "XAI"
    conversation_history: Optional[List[Dict[str, str]]] = []
    use_qdrant_hints: bool = True  # Enable/disable hint system

class ToolExecution(BaseModel):
    tool_name: str
    tool_response: str
    tool_execution_time: float

class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    provider: str
    model_name: str

class QueryResponse(BaseModel):
    response: str
    tools: List[ToolExecution]
    response_time: float
    token_usage: TokenUsage

class SQLQueryRequest(BaseModel):
    query: str
    provider: str = "XAI"
    conversation_history: Optional[List[Dict[str, str]]] = []
    db_id: Optional[str] = None  # Optional custom SQLite database ID
    connection_id: Optional[str] = None  # Optional connection ID for hint system (defaults to db_id if not provided)
    use_qdrant_hints: bool = True  # Enable/disable hint system
    db_path: Optional[str] = None  # Optional custom SQLite database path
    db_type: Optional[str] = "sqlite"  # Database type (sqlite, postgres, etc.)

class SQLQueryResponse(BaseModel):
    sql_query: str
    tools: List[ToolExecution]
    response_time: float
    token_usage: TokenUsage

# Agent setup (same as streamlit-app)
class MessagesState:
    messages: List[Any]
    llm_calls: int
    hit_limit: bool

def llm_call(state: dict, model_with_tools):
    """LLM decides whether to call a tool or not"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=f"""You are a helpful database assistant. 
                        
                        For simple greetings and conversation, respond naturally without using any tools.
                        
                        For database questions (anything involving data, queries, tables, analysis):
                        - FIRST get hints using get_hints to learn from previous successful patterns for this connection
                        - If hints are helpful, use them to generate SQL quickly
                        - If hints are not relevant, get schema context using get_schemas_with_tables to understand available tables and structure
                        - Then use query_db to answer their specific question with appropriate SQL
                        - You're connected to a datasource with query generation instructions: {db_connector.get_query_generation_instructions()}
                        - If you're struggling to generate a query, use the query execution tool to get 5 sample rows of data to understand the schema and data better
                        
                        HINT SYSTEM USAGE:
                        - Use get_hints(query) at the start to see relevant patterns
                        - After successfully solving a complex query, consider using add_hint_simple to store the pattern
                        - Only add hints when the solution required significant effort or the hints were not helpful
                        - Let your experience with the current query guide whether to add new hints
                        
                        IMPORTANT for python_code_execution tool:
                        - Use this tool ONLY ONCE per visualization request
                        - If the code fails, DO NOT retry with different approaches - instead explain the error to the user
                        - Available libraries: matplotlib, plotly, base64, json, math
                        - For matplotlib: Use plt.savefig() with BytesIO and base64 encode, return as HTML img tag: '<img src="data:image/png;base64,<base64_string>" />'
                        - For plotly: Use fig.to_html(include_plotlyjs='cdn') for interactive graphs, return the full HTML string
                        - Store the final result in a variable called 'result'
                        - The orchestrator will automatically extract and display charts/images from tool results to the user
                        - In your final response, DO NOT include the base64 image data or HTML - just describe the chart and provide insights"""
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1,
        "hit_limit": state.get('hit_limit', False)
    }

def tool_node(state: dict, tools_by_name):
    """Performs the tool call"""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> str:
    """Decide if we should continue the loop or stop"""
    messages = state["messages"]
    last_message = messages[-1]
    llm_calls = state.get('llm_calls', 0)

    # Stop if we've made too many calls (prevent infinite loops)
    if llm_calls >= 10:
        state["hit_limit"] = True
        return "END"

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return "END"

def create_agent(provider: str = "XAI"):
    """Create and compile the agent"""
    model = model_init(provider)
    from tool_definitions import all_tools
    tools = all_tools
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    # Build workflow
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", lambda state: llm_call(state, model_with_tools))
    agent_builder.add_node("tool_node", lambda state: tool_node(state, tools_by_name))

    # Add edges to connect nodes
    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        ["tool_node", END]
    )
    agent_builder.add_edge("tool_node", "llm_call")

    # Compile the agent
    return agent_builder.compile()

def llm_call_sql_only(state: dict, model_with_tools):
    """LLM generates SQL queries without execution"""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=f"""You are a SQL query generator. Your ONLY job is to generate SQL queries based on user questions.
                        
                        For database questions:
                        - FIRST get schema context using get_schemas_with_tables to understand available tables and structure
                        - Then generate ONLY the SQL query - no explanations, no formatting, no markdown
                        - Return the raw SQL query as your final response
                        - Do NOT include ```sql or any other formatting
                        - Do NOT execute queries or provide results
                        - You're connected to a datasource with query generation instructions: {db_connector.get_query_generation_instructions()}
                        
                        Available tools: get_schemas_with_tables, get_create_table_statements
                        Do NOT use query_db or python_code_execution tools."""
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1,
        "hit_limit": state.get('hit_limit', False)
    }

@app.post("/query", response_model=QueryResponse)
async def query_database(request: QueryRequest):
    """Process a database query and return response with tool execution details"""
    
    try:
        # Create agent
        agent = create_agent(request.provider, request.use_qdrant_hints)
        
        # Convert conversation history to LangChain messages
        langchain_messages = []
        for msg in request.conversation_history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        langchain_messages.append(HumanMessage(content=request.query))
        
        # Initialize tracking variables
        response_start_time = time.time()
        tool_executions = []
        final_message = ""
        hit_limit = False
        input_tokens = 0
        output_tokens = 0
        
        # Process the query
        for chunk in agent.stream({"messages": langchain_messages}, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                # Check if we hit the limit
                if "hit_limit" in node_output and node_output["hit_limit"]:
                    hit_limit = True
                
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        # Track tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_start_time = time.time()
                                
                        # Track tool results
                        elif hasattr(msg, 'content') and node_name == "tool_node":
                            tool_end_time = time.time()
                            # Extract tool name from previous tool call
                            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                                # Find the tool name from the tool call
                                tool_name = "unknown"
                                for prev_msg in langchain_messages[-5:]:  # Look at recent messages
                                    if hasattr(prev_msg, 'tool_calls'):
                                        for tc in prev_msg.tool_calls:
                                            if tc.get('id') == msg.tool_call_id:
                                                tool_name = tc.get('name', 'unknown')
                                                break
                                
                                tool_executions.append(ToolExecution(
                                    tool_name=tool_name,
                                    tool_response=msg.content,
                                    tool_execution_time=round(tool_end_time - tool_start_time, 2)
                                ))
                        
                        # Capture final response
                        elif hasattr(msg, 'content') and node_name == "llm_call" and not msg.tool_calls:
                            final_message = msg.content
        
        response_end_time = time.time()
        total_response_time = round(response_end_time - response_start_time, 2)
        
        # Handle hit limit case
        if hit_limit:
            final_message = """I've reached the maximum number of tool calls (10) to prevent infinite loops. 

Here's what I was able to gather so far:
"""
            if tool_executions:
                for i, tool_exec in enumerate(tool_executions[-3:], 1):
                    final_message += f"\n- Tool result {i}: {tool_exec.tool_response[:100]}..."
            final_message += "\n\nPlease ask a more specific question or provide additional guidance to continue the analysis."
        
        # Get model name from environment variables
        model_name = ""
        if request.provider == "Azure":
            model_name = os.getenv("AZURE_DEPLOYMENT_NAME", "")
        elif request.provider == "Claude":
            model_name = os.getenv("CLAUDE_MODEL_NAME", "")
        elif request.provider == "XAI":
            model_name = os.getenv("XAI_MODEL_NAME", "")
        elif request.provider == "Ollama":
            model_name = os.getenv("OLLAMA_MODEL_NAME", "")
        
        # For now, we'll use placeholder token counts since LangChain doesn't easily expose them
        # In a real implementation, you'd track these from the model responses
        token_usage = TokenUsage(
            input_tokens=len(request.query.split()) * 2,  # Rough estimate
            output_tokens=len(final_message.split()) * 2,  # Rough estimate
            provider=request.provider,
            model_name=model_name
        )
        
        return QueryResponse(
            response=final_message,
            tools=tool_executions,
            response_time=total_response_time,
            token_usage=token_usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

from db_connector.sqlite_connector import SQLiteConnector
_connector_cache: dict[str, SQLiteConnector] = {}

def get_connector(db_path: str, db_type: str = "sqlite") -> SQLiteConnector:
    if db_path not in _connector_cache:
        print(f"Creating new connector for {db_path} with type {db_type}")
        if db_type == "sqlite":
            _connector_cache[db_path] = SQLiteConnector(db_path=db_path)
        else:
            raise ValueError(f"Unsupported database type: {db_type}")
    return _connector_cache[db_path]

@app.post("/sql-query", response_model=SQLQueryResponse)
async def generate_sql_query(request: SQLQueryRequest):
    """Generate SQL query without execution for benchmarking"""
    print("Request received for SQL query generation", request)
    
    try:
        # tmp = os.environ.get("SQLITE_DB_PATH")
        # print("Current SQLITE_DB_PATH:", tmp)
        # Create a fresh SQLite connector for this specific database
        # new_db_path = tmp.replace("db_id", request.db_id)
        print("Creating SQLite connector for db_id:", request.db_path)
        sqlite_connector = get_connector(request.db_path, request.db_type)

        # Create SQL-only agent with provider for LLM
        # Use connection_id from request, or default to db_id
        connection_id = request.connection_id or request.db_id
        agent = create_sql_only_agent(request.provider, request.use_qdrant_hints, connection_id, sqlite_connector)
        
        # Convert conversation history to LangChain messages
        langchain_messages = []
        for msg in request.conversation_history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Add current query
        langchain_messages.append(HumanMessage(content=request.query))
        
        # Initialize tracking variables
        response_start_time = time.time()
        tool_executions = []
        sql_query = ""
        hit_limit = False
        
        # Process the query
        for chunk in agent.stream({"messages": langchain_messages}, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                # Check if we hit the limit
                if "hit_limit" in node_output and node_output["hit_limit"]:
                    hit_limit = True
                
                if "messages" in node_output:
                    for msg in node_output["messages"]:
                        # Track tool calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_start_time = time.time()
                                
                        # Track tool results
                        elif hasattr(msg, 'content') and node_name == "tool_node":
                            tool_end_time = time.time()
                            # Extract tool name from previous tool call
                            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                                # Find the tool name from the tool call
                                tool_name = "unknown"
                                for prev_msg in langchain_messages[-5:]:  # Look at recent messages
                                    if hasattr(prev_msg, 'tool_calls'):
                                        for tc in prev_msg.tool_calls:
                                            if tc.get('id') == msg.tool_call_id:
                                                tool_name = tc.get('name', 'unknown')
                                                break
                                
                                tool_executions.append(ToolExecution(
                                    tool_name=tool_name,
                                    tool_response=msg.content,
                                    tool_execution_time=round(tool_end_time - tool_start_time, 2)
                                ))
                        
                        # Capture final SQL query
                        elif hasattr(msg, 'content') and node_name == "llm_call" and not msg.tool_calls:
                            content = msg.content.strip()
                            # Extract SQL from response - look for SQL keywords and extract from there
                            sql_upper = content.upper()
                            valid_starts = ['SELECT', 'WITH', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']

                            # Find where SQL starts
                            sql_start_idx = -1
                            for start in valid_starts:
                                idx = sql_upper.find(start)
                                if idx != -1 and (sql_start_idx == -1 or idx < sql_start_idx):
                                    sql_start_idx = idx

                            if sql_start_idx != -1:
                                # Extract SQL from the first SQL keyword found
                                sql_query = content[sql_start_idx:].strip()
                            else:
                                # LLM likely output refusal text instead of SQL
                                print(f"Invalid SQL output detected: {content[:100]}...")
                                sql_query = "SELECT 1 -- LLM refused to generate valid SQL"
        
        response_end_time = time.time()
        total_response_time = round(response_end_time - response_start_time, 2)
        
        # Handle hit limit case
        if hit_limit:
            sql_query = "SELECT 1 -- Hit tool call limit, unable to generate query"
        
        # Get model name from environment variables
        model_name = ""
        if request.provider == "Azure":
            model_name = os.getenv("AZURE_DEPLOYMENT_NAME", "")
        elif request.provider == "Claude":
            model_name = os.getenv("CLAUDE_MODEL_NAME", "")
        elif request.provider == "XAI":
            model_name = os.getenv("XAI_MODEL_NAME", "")
        elif request.provider == "Ollama":
            model_name = os.getenv("OLLAMA_MODEL_NAME", "")
        
        # For now, we'll use placeholder token counts since LangChain doesn't easily expose them
        # In a real implementation, you'd track these from the model responses
        token_usage = TokenUsage(
            input_tokens=len(request.query.split()) * 2,  # Rough estimate
            output_tokens=len(sql_query.split()) * 2,  # Rough estimate
            provider=request.provider,
            model_name=model_name
        )
        print("SQL Query Response:", sql_query)
        return SQLQueryResponse(
            sql_query=sql_query,
            tools=tool_executions,
            response_time=total_response_time,
            token_usage=token_usage
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SQL query: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Database Query Assistant API", "version": "1.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4747)