from dotenv import load_dotenv
import streamlit as st
import os
import time
import json
import logging
import warnings

load_dotenv()

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*__path__.*")
from tool_definitions import query_db, get_schemas_with_tables, get_create_table_statements, python_code_execution
from common_config import db_connector
from agent_common import (
    MessagesState, model_init, llm_call, tool_node, 
    should_continue, create_agent
)


# Create agent using common module
agent = create_agent(provider="XAI")




# Streamlit Chat Interface
st.title("Database Query Assistant")
st.write("Ask questions about your database and I'll help you analyze the data.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Display tool results (images) first
        if message["role"] == "assistant" and "tool_results" in message and message["tool_results"]:
            import json
            for tool_result in message["tool_results"]:
                try:
                    result_dict = json.loads(tool_result)
                    if "result" in result_dict:
                        result_content = result_dict["result"]
                        # Handle different formats
                        if result_content.startswith("<html"):
                            # Full HTML (plotly interactive graph)
                            st.iframe(srcdoc=result_content, height=600, scrolling=True)
                        elif result_content.startswith("<img"):
                            # HTML img tag (matplotlib)
                            st.markdown(result_content, unsafe_allow_html=True)
                        elif result_content.startswith("data:image/"):
                            # Base64 data URI
                            st.markdown(f'<img src="{result_content}" style="max-width: 100%; height: auto;">', unsafe_allow_html=True)
                except json.JSONDecodeError:
                    if tool_result.startswith("<html"):
                        st.iframe(srcdoc=tool_result, height=600, scrolling=True)
                    elif tool_result.startswith("<img"):
                        st.markdown(tool_result, unsafe_allow_html=True)
                    elif tool_result.startswith("data:image/"):
                        st.markdown(f'<img src="{tool_result}" style="max-width: 100%; height: auto;">', unsafe_allow_html=True)
        
        st.markdown(message["content"], unsafe_allow_html=True)
        # Show execution steps and timings for assistant messages
        if message["role"] == "assistant":
            # Show total duration
            if "total_duration" in message:
                st.caption(f"⏱️ Response time: {message['total_duration']}s")
            # Show tool timings
            if "tool_timings" in message and message["tool_timings"]:
                with st.expander("View tool execution times"):
                    for tool_name, timing in message["tool_timings"].items():
                        if "duration" in timing:
                            st.markdown(f"- {tool_name}: {timing['duration']}s")
            # Show execution steps
            if "execution_steps" in message and message["execution_steps"]:
                with st.expander("View execution steps"):
                    for step in message["execution_steps"]:
                        st.markdown(step)

# Chat input
if prompt := st.chat_input("Ask a question about your database:"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from agent with streaming
    with st.chat_message("assistant"):
        from langchain.messages import HumanMessage, AIMessage
        
        # Convert chat history to LangChain messages (already includes current prompt from line 147)
        langchain_messages = []
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
        
        # Create a status container for showing steps
        with st.status("Processing...", expanded=True) as status:
            step_count = 0
            final_message = ""
            execution_steps = []
            tool_results = []
            tool_timings = {}
            response_start_time = time.time()
            hit_limit = False
            
            for chunk in agent.stream({"messages": langchain_messages}, stream_mode="updates"):
                for node_name, node_output in chunk.items():
                    # Check if we hit the limit
                    if "hit_limit" in node_output and node_output["hit_limit"]:
                        hit_limit = True
                        status.write("⚠️ **Reached maximum tool call limit (10)**")
                    
                    step_count += 1
                    node_start_time = time.time()
                    step_text = f"**Step {step_count}:** Executing `{node_name}`"
                    status.write(step_text)
                    execution_steps.append(step_text)
                    
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_text = f"  🛠️ Calling tool: `{tool_call['name']}`"
                                    args_text = f"  📝 Args: `{tool_call['args']}`"
                                    status.write(tool_text)
                                    status.write(args_text)
                                    execution_steps.append(tool_text)
                                    execution_steps.append(args_text)
                                    tool_timings[tool_call['name']] = {"start": node_start_time}
                            elif hasattr(msg, 'content'):
                                # Capture tool results (from tool_node)
                                if node_name == "tool_node":
                                    tool_results.append(msg.content)
                                    node_end_time = time.time()
                                    # Update timing for the tool that was called
                                    for tool_name in tool_timings:
                                        if tool_timings[tool_name].get("end") is None:
                                            tool_timings[tool_name]["end"] = node_end_time
                                            tool_timings[tool_name]["duration"] = round(node_end_time - tool_timings[tool_name]["start"], 2)
                                    status.write(f"  📦 Tool result: {msg.content[:100]}...")
                                    execution_steps.append(f"  📦 Tool result: {msg.content[:100]}...")
                                elif msg.content and not msg.content.startswith('{'):
                                    response_text = f"  💬 Response: `{msg.content[:100]}...`"
                                    status.write(response_text)
                                    execution_steps.append(response_text)
                                    # Capture the final AI response
                                    if node_name == "llm_call" and not msg.tool_calls:
                                        final_message = msg.content
            
            response_end_time = time.time()
            total_duration = round(response_end_time - response_start_time, 2)
            status.write(f"⏱️ **Total response time:** {total_duration}s")
            
            # Display tool timings
            if tool_timings:
                status.write("**Tool execution times:**")
                for tool_name, timing in tool_timings.items():
                    if "duration" in timing:
                        status.write(f"  - {tool_name}: {timing['duration']}s")
            
            status.update(label="Complete!", state="complete", expanded=False)
        
        # If limit was hit, generate a summary response
        if hit_limit:
            final_message = """I've reached the maximum number of tool calls (10) to prevent infinite loops. 

Here's what I was able to gather so far:
"""
            # Add tool results summary
            if tool_results:
                for i, tool_result in enumerate(tool_results[-3:], 1):  # Show last 3 results
                    try:
                        result_dict = json.loads(tool_result)
                        if "data" in result_dict:
                            final_message += f"\n- Query result {i}: {result_dict.get('data', 'No data')}"
                        elif "error" in result_dict:
                            final_message += f"\n- Query error {i}: {result_dict['error']}"
                        elif "result" in result_dict:
                            final_message += f"\n- Tool result {i}: {str(result_dict['result'])[:100]}..."
                    except:
                        final_message += f"\n- Result {i}: {tool_result[:100]}..."
            
            final_message += "\n\nPlease ask a more specific question or provide additional guidance to continue the analysis."
        
        # Display tool results (images from python_code_execution)
        for tool_result in tool_results:
            try:
                result_dict = json.loads(tool_result)
                if "result" in result_dict:
                    result_content = result_dict["result"]
                    # Handle different formats
                    if result_content.startswith("<html"):
                        # Full HTML (plotly interactive graph)
                        st.iframe(srcdoc=result_content, height=600, scrolling=True)
                    elif result_content.startswith("<img"):
                        # Already an HTML img tag (matplotlib)
                        st.markdown(result_content, unsafe_allow_html=True)
                    elif result_content.startswith("data:image/"):
                        # Base64 data URI - wrap in img tag
                        st.markdown(f'<img src="{result_content}" style="max-width: 100%; height: auto;">', unsafe_allow_html=True)
            except json.JSONDecodeError:
                # If not JSON, display as-is
                if tool_result.startswith("<html"):
                    st.iframe(srcdoc=tool_result, height=600, scrolling=True)
                elif tool_result.startswith("<img"):
                    st.markdown(tool_result, unsafe_allow_html=True)
                elif tool_result.startswith("data:image/"):
                    st.markdown(f'<img src="{tool_result}" style="max-width: 100%; height: auto;">', unsafe_allow_html=True)
        
        # Display final message as markdown (supports base64 images in HTML img tags)
        st.markdown(final_message.replace("$", r"\$"), unsafe_allow_html=True)
        
        # Show timings for current message
        st.caption(f"⏱️ Response time: {total_duration}s")
        if tool_timings:
            with st.expander("View tool execution times"):
                for tool_name, timing in tool_timings.items():
                    if "duration" in timing:
                        st.markdown(f"- {tool_name}: {timing['duration']}s")
    
    # Add assistant response, execution steps, tool results, and timings to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": final_message, 
        "execution_steps": execution_steps,
        "tool_results": tool_results,
        "total_duration": total_duration,
        "tool_timings": tool_timings
    })