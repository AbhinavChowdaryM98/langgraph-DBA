"""
Common agent module to eliminate code duplication between app.py and streamlit-app."""

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from tool_definitions import query_db, get_schemas_with_tables, get_create_table_statements, python_code_execution, all_tools
from common_config import db_connector
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from typing import Literal
import os

# Agent setup
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    hit_limit: bool

def model_init(provider: str = "Ollama"):
    """Initialize model based on provider."""
    if provider == "Azure":
        model = init_chat_model(
            os.getenv("AZURE_DEPLOYMENT_NAME"),
            model_provider="azure_openai"
        )
    elif provider == "Claude":
        model = init_chat_model(
            os.getenv("CLAUDE_MODEL_NAME"),
            model_provider="anthropic_bedrock"
        )
    elif provider == "XAI":
        model = init_chat_model(
            os.getenv("XAI_MODEL_NAME"),
            model_provider="xai"
        )
    elif provider == "Ollama":
        model = init_chat_model(
            os.getenv("OLLAMA_MODEL_NAME"),
            model_provider="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")
    return model

def llm_call(state: dict, model_with_tools):
    """LLM decides whether to call a tool or not."""
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
                        - After successfully solving a complex query, you may use add_hint_simple to store the pattern
                        - Only add hints when the solution required significant effort or the hints were not helpful
                        - IMPORTANT: After using add_hint_simple, you MUST continue to provide your final answer to the user
                        - Hint tools are optional side operations - they do NOT replace your final response
                        - ALWAYS provide a clear, natural language answer to the user's original question
                        
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
    """Performs the tool call."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop."""
    messages = state["messages"]
    last_message = messages[-1]
    llm_calls = state.get('llm_calls', 0)

    # Stop if we've made too many calls (prevent infinite loops)
    if llm_calls >= 10:
        state["hit_limit"] = True
        return END

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END

def create_agent(provider: str = "XAI"):
    """Create and compile the agent."""
    model = model_init(provider)
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
    """LLM generates SQL queries without execution."""
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

def create_sql_only_agent(provider: str = "XAI"):
    """Create and compile the SQL-only agent."""
    model = model_init(provider)
    
    # Create SQLite-specific tools
    from db_connector.factory import DBType
    from db_connector.sqlite_connector import SQLiteConnector
    sqlite_connector = SQLiteConnector()
    
    # Create SQLite-specific tool functions
    @tool
    def sqlite_get_schemas_with_tables() -> str:
        """Get all schemas with their table names for SQLite."""
        schemas_with_tables = sqlite_connector.get_schemas_with_tables()
        if not schemas_with_tables:
            return json.dumps({"error": "Failed to get schemas with tables"})
        return json.dumps(schemas_with_tables, default=str)

    @tool  
    def sqlite_get_create_table_statements(table_name: str | list[str]) -> str:
        """Get the CREATE TABLE statement for a given table in SQLite."""
        create_table_statement = sqlite_connector.get_create_table_statements(table_name)
        if not create_table_statement:
            return json.dumps({"error": "Failed to get create table statement"})
        return json.dumps({"create_table_statement": create_table_statement}, default=str)
    
    tools = [sqlite_get_schemas_with_tables, sqlite_get_create_table_statements]
    tools_by_name = {tool.name: tool for tool in tools}
    model_with_tools = model.bind_tools(tools)

    # Build workflow
    agent_builder = StateGraph(MessagesState)
    agent_builder.add_node("llm_call", lambda state: llm_call_sql_only(state, model_with_tools))
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
