# Langraph Database Query Assistant

A sophisticated AI-powered database query assistant built with LangGraph, Streamlit, and LangChain. This application enables natural language interaction with databases, supporting both SQL and DAX queries with advanced visualization capabilities.

## Features

- **Multi-Model Support**: Supports PostgreSQL and Microsoft Fabric Semantic Models
- **Natural Language Interface**: Ask questions in plain English and get AI-generated queries
- **Advanced Visualizations**: Built-in support for matplotlib and plotly charts
- **Real-time Chat Interface**: Interactive Streamlit-based UI with execution tracking
- **Multiple LLM Providers**: Support for Azure OpenAI, Claude (Anthropic Bedrock), XAI, and Ollama
- **Tool-based Architecture**: LangGraph workflow with intelligent tool selection
- **Query Optimization**: Smart query generation with schema awareness

## Architecture

### Core Components

- **`app.py`**: Main Streamlit application with LangGraph agent workflow
- **`tool_definitions.py`**: LangChain tools for database operations and code execution
- **`db_connector/`**: Database abstraction layer with multiple connector implementations
- **`app_utils.py`**: Utility functions for safety checks and validation

### Database Connectors

- **PostgreSQL Connector**: Full SQL support with schema introspection
- **Semantic Model Connector**: DAX query support for Microsoft Fabric semantic models
- **Legacy Connectors**: Support for Azure SQL, MySQL, Snowflake, and more

## Prerequisites

### System Requirements

- Python 3.8+
- ODBC Driver 18 for MS SQL Server (required for SQL Server connections)
- Azure CLI (required for semantic model authentication - see note below)

### Authentication Requirements

⚠️ **Important**: For semantic model connections, Azure CLI login is required as execute queries do not work with service principal-based authentication.

To authenticate:
```bash
az login
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd langraph
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with the following variables:

   ```env
   # Database Configuration
   POSTGRES_HOST=localhost
   POSTGRES_DB_NAME=your_database
   POSTGRES_USER=your_username
   POSTGRES_PASSWORD=your_password
   
   # Semantic Model Configuration
   SEMANTIC_DB_NAME=your_semantic_model
   SEMANTIC_DB_ID=your_dataset_id
   SEMANTIC_WORKSPACE_ID=your_workspace_id
   
   # LLM Provider Configuration (choose one or more)
   # Azure OpenAI
   AZURE_DEPLOYMENT_NAME=your_deployment_name
   AZURE_OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_endpoint
   
   # Anthropic Bedrock
   CLAUDE_MODEL_NAME=your_model_name
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_region
   
   # XAI
   XAI_MODEL_NAME=your_model_name
   XAI_API_KEY=your_api_key
   
   # Ollama (optional)
   OLLAMA_MODEL_NAME=your_model_name
   OLLAMA_BASE_URL=http://localhost:11434
   ```

## Usage

1. **Start the application**:
   ```bash
   streamlit run app.py
   ```

2. **Interact with the assistant**:
   - Type your questions in natural language
   - The AI will generate and execute appropriate queries
   - View results in tables, charts, or interactive visualizations
   - Monitor execution steps and timing information

### Supported Query Types

- **PostgreSQL**: Standard SQL queries (SELECT, JOIN, aggregation, etc.)
- **Semantic Models**: DAX queries for Microsoft Fabric datasets

### Visualization Capabilities

- **Matplotlib**: Static charts and plots
- **Plotly**: Interactive graphs and dashboards
- **Base64 Encoding**: Direct image embedding in responses

## Development

### Project Structure

```
langraph/
├── app.py                          # Main Streamlit application
├── app_utils.py                    # Utility functions
├── tool_definitions.py             # LangChain tool definitions
├── requirements.txt                # Python dependencies
├── db_connector/                   # Database connector modules
│   ├── __init__.py
│   ├── base_connector.py          # Abstract base connector
│   ├── factory.py                 # Connector factory
│   ├── config.py                  # Configuration utilities
│   ├── pg_connector.py            # PostgreSQL connector
│   ├── semantic_model_connector.py # Semantic model connector
│   ├── ms_sql_server_connector.py # SQL Server connector
│   └── legacy_connectors/         # Legacy connector implementations
├── Temp/                          # Temporary files and testing
└── readme.md                      # This file
```

### Adding New Connectors

1. Create a new connector class inheriting from `BaseDBConnector`
2. Implement required methods:
   - `execute_query()`
   - `get_schemas_with_tables()`
   - `get_create_table_statements()`
   - `get_query_generation_instructions()`
3. Register the connector in `factory.py`

### Customizing LLM Behavior

Modify the system prompt in `app.py` (line 71-83) to adjust:
- Query generation instructions
- Tool usage behavior
- Response formatting

## Troubleshooting

### Common Issues

1. **Semantic Model Authentication Errors**:
   - Ensure Azure CLI is installed and you're logged in (`az login`)
   - Verify workspace and dataset IDs are correct

2. **ODBC Driver Issues**:
   - Install ODBC Driver 18 for MS SQL Server
   - Verify driver installation: `odbcinst -q -d -n "ODBC Driver 18 for SQL Server"`

3. **Connection Timeouts**:
   - Check network connectivity
   - Verify firewall settings
   - Increase timeout values in connector configuration

4. **Memory Issues with Large Datasets**:
   - The application automatically limits result sets to prevent memory issues
   - Consider adding WHERE clauses to reduce data volume

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions:
- Check the troubleshooting section
- Review the code comments
- Open an issue in the repository