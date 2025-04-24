# internet_of_happiness.py
# Description: Orchestrates HSI analysis using LangGraph for workflow,
#              FastAPI for API endpoint, LangChain for LLM/tool integration,
#              incorporating Mamba-based preprocessing and Llama 4 Scout BioMedical.
#              Conceptual integration of MCP (client-side) and A2A.
# Target Model: Llama 4 Scout BioMedical (via Bedrock/SageMaker)
# HSI Models: SpectralGPT, MambaLG, MambaHSI (as tools/functions)
# Data Source: Voyage81 API (via UI API/MCP Tool)
# Workflow: LangGraph
# API Framework: FastAPI
# Cloud Platform: AWS (SageMaker, Bedrock)
# API Integration: Universal Informatics API (via MCP Tool)

import json
import os
import datetime
import time
import logging
import uuid
from typing import TypedDict, Annotated, Sequence, Optional
import operator

# --- Core Framework Imports ---
from fastapi import FastAPI, HTTPException, Depends, Request # API Framework
import uvicorn # ASGI Server
from pydantic import BaseModel # Data validation for API

# --- LangChain & LangGraph Imports ---
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain.tools import Tool # For wrapping functions as tools
# Assume Bedrock integration for Scout (adapt if using SageMaker endpoint)
# from langchain_aws import ChatBedrock # Example
from langchain_core.prompts import ChatPromptTemplate # For Scout prompts
# Placeholder for LangChain's MCP Client (adapt based on actual library)
# from langchain_community.mcp import MultiServerMCPClient, MCPTool # Example
from langgraph.graph import StateGraph, END, START # Workflow orchestration
from langgraph.checkpoint.sqlite import SqliteSaver # Example checkpointer for memory

# Placeholder for AWS SDK
try:
    import boto3
except ImportError:
    print("AWS SDK (boto3) not installed. AWS interactions will be simulated.")
    boto3 = None

# Placeholders for HSI/Mamba/Hyper-Seq modules (as before)
# Assume these are adapted into functions callable by LangChain Tools

# --- Configuration ---
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
SCOUT_BIOMEDICAL_MODEL_ID = "meta.llama4-scout-17b-instruct-v1:0" # Example Bedrock ID
# UNIVERSAL_INFORMATICS_API_MCP_CONFIG = '{"url": "http://mcp.ui.example.com", "token_secret": "UiApiMcpTokenSecretName"}' # Example config JSON for UI API's MCP server
# Assume the UI API has an MCP Server generated (e.g., by SpeakEasy from its OpenAPI spec)
# This server exposes functions like 'get_hsi_data', 'log_event' via MCP.

# --- Logging Setup (Basic) ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# In prod, leverage UI API for centralized logging (via MCP tool)

# --- Helper Functions / Placeholder HSI Functions ---
# (Keep placeholder functions like run_spectral_gpt, run_mambalg, run_mambahsi from previous script)
# ...

def run_spectral_gpt(hsi_data):
    logging.info("Running SpectralGPT processing (Placeholder)...")
    time.sleep(0.1); patterns = {"spectral_patterns": [0.1, 0.5, 0.9], "confidence": 0.85}; return patterns
def run_mambalg(hsi_data):
    logging.info("Running MambaLG classification (Placeholder)...")
    time.sleep(0.1); classification_result = {"class": "probable_autofluorescence_region", "score": 0.75}; return classification_result
def run_mambahsi(hsi_data):
    logging.info("Running MambaHSI processing (Placeholder)...")
    time.sleep(0.1); features = {"spatial_spectral_features": "encoded_feature_vector_mambahsi"}; return features
def run_hyperseq_analysis(hsi_data, scout_results):
    logging.info("Running Hyper-Seq analysis/validation (Placeholder)...")
    time.sleep(0.1); validation_score = 0.88; return {"hyperseq_validation": validation_score}


# --- LangChain Tool Definitions ---

# Tool for combined HSI Mamba preprocessing
def mamba_hsi_preprocessing_tool_func(hsi_data_cube: str) -> dict:
    """
    Runs a sequence of Mamba-based HSI processing steps (SpectralGPT, MambaLG, MambaHSI)
    on the provided HSI data cube (represented as a string/placeholder here).
    Returns combined features suitable for the Scout model.
    """
    logging.info("Executing Mamba HSI Preprocessing Tool...")
    # Simulate processing the 'hsi_data_cube'
    try:
        spectral_gpt_results = run_spectral_gpt(hsi_data_cube)
        mambalg_results = run_mambalg(hsi_data_cube)
        mambahsi_features = run_mambahsi(hsi_data_cube)

        combined_features = {
            "status": "success",
            "spectral_patterns": spectral_gpt_results.get("spectral_patterns"),
            "classification_hint": mambalg_results.get("class"),
            "mambahsi_encoded_features": mambahsi_features.get("spatial_spectral_features"),
        }
        logging.info("Mamba HSI Preprocessing Tool finished successfully.")
        return combined_features
    except Exception as e:
        logging.error(f"Error in Mamba HSI Preprocessing Tool: {e}")
        return {"status": "error", "message": str(e)}

mamba_hsi_tool = Tool(
    name="MambaHSIPreprocessor",
    func=mamba_hsi_preprocessing_tool_func,
    description="Processes Hyperspectral Imaging (HSI) data using specialized Mamba-based models (SpectralGPT, MambaLG, MambaHSI) to extract relevant features for gene expression analysis.",
)

# Tool(s) for Universal Informatics API via MCP (Conceptual)
# Assumes the UI API exposes an MCP server (e.g., generated via SpeakEasy)
# We'd use LangChain's MCP client integration to create tools dynamically.
def setup_ui_api_mcp_tools():
    """
    Placeholder function to set up LangChain tools by connecting to the
    Universal Informatics API's MCP Server.
    """
    logging.info("Setting up Universal Informatics API MCP Tools (Conceptual)...")
    # In a real implementation:
    # 1. Parse UNIVERSAL_INFORMATICS_API_MCP_CONFIG
    # 2. Fetch auth token using get_secret() if needed by the client setup itself
    # 3. Instantiate MCP Client (e.g., MultiServerMCPClient)
    # 4. Use client.get_tools() or similar to get LangChain Tool objects for 'get_hsi_data', 'log_event', etc.
    # mcp_client = MultiServerMCPClient(config=json.loads(UNIVERSAL_INFORMATICS_API_MCP_CONFIG))
    # ui_api_tools = mcp_client.get_tools() # This is conceptual

    # --- Simulation ---
    def get_hsi_data_sim(user_id: str, session_id: str) -> dict:
        """Simulates fetching HSI data via UI API/MCP."""
        logging.info(f"[MCP Tool Sim] Getting HSI data for user {user_id}")
        return {"status": "success", "data": {"hsi_cube": "dummy_hsi_data_array", "metadata": {"source": "Voyage81_simulated", "user_id": user_id}}}

    def log_event_sim(level: str, message: str, details: Optional[dict] = None) -> dict:
        """Simulates logging via UI API/MCP."""
        logging.info(f"[MCP Tool Sim] Logging event: Level={level}, Msg='{message}', Details={details}")
        return {"status": "success", "message": "Event logged."}

    def report_result_sim(analysis_payload: dict) -> dict:
         """Simulates reporting final result via UI API/MCP."""
         logging.info(f"[MCP Tool Sim] Reporting analysis result for user {analysis_payload.get('user_id')}")
         return {"status": "success", "message": "Result reported."}

    get_hsi_tool = Tool(name="GetHSIDataViaUIAPI", func=get_hsi_data_sim, description="Fetches HSI data cube from Voyage81 for a given user via the Universal Informatics API.")
    log_event_tool = Tool(name="LogEventViaUIAPI", func=log_event_sim, description="Logs an event (INFO, ERROR, CRITICAL) with details to the centralized logging system via the Universal Informatics API.")
    report_result_tool = Tool(name="ReportAnalysisResultViaUIAPI", func=report_result_sim, description="Reports the final gene expression analysis result via the Universal Informatics API.")

    return [get_hsi_tool, log_event_tool, report_result_tool]
    # --- End Simulation ---

ui_api_tools = setup_ui_api_mcp_tools()
all_tools = [mamba_hsi_tool] + ui_api_tools

# --- LLM Setup (Llama 4 Scout BioMedical via Bedrock) ---
def get_scout_llm():
    """Initializes the LangChain LLM client for Scout."""
    logging.info(f"Initializing LLM: {SCOUT_BIOMEDICAL_MODEL_ID} via Bedrock")
    if not boto3:
         logging.warning("Boto3 not available, cannot initialize Bedrock LLM.")
         return None # Cannot proceed without AWS SDK in real scenario

    try:
        # Example using ChatBedrock - adjust model_kwargs as needed
        # llm = ChatBedrock(
        #     client=boto3.client("bedrock-runtime", region_name=AWS_REGION),
        #     model_id=SCOUT_BIOMEDICAL_MODEL_ID,
        #     model_kwargs={"temperature": 0.6, "top_p": 0.9, "max_gen_len": 512},
        #     # May need streaming=True depending on LangGraph usage
        # )
        # return llm
        # --- Simulation ---
        class SimulatedScoutLLM:
            def invoke(self, prompt):
                logging.info(f"Simulating Scout LLM invocation...")
                time.sleep(0.5)
                # Simulate response based on looking for keywords in prompt
                prompt_str = str(prompt)
                content = "Simulation Error: Could not generate response."
                if "HSI Analysis Summary" in prompt_str and "OXTR" in prompt_str:
                    content = "Based on the simulated HSI features and Hyper-Seq context, the inferred OXTR gene expression level is moderately high (0.75 on a normalized scale). Autofluorescence patterns consistent with target TFBS binding detected."
                elif "Error" in prompt_str:
                     content = "Acknowledged error state."

                return AIMessage(content=content)

            async def ainvoke(self, prompt): # Async version for LangGraph
                logging.info(f"Simulating Scout LLM async invocation...")
                await asyncio.sleep(0.5)
                prompt_str = str(prompt)
                content = "Simulation Error: Could not generate async response."
                if "HSI Analysis Summary" in prompt_str and "OXTR" in prompt_str:
                    content = "Based on the simulated HSI features and Hyper-Seq context, the inferred OXTR gene expression level is moderately high (0.75 on a normalized scale). Autofluorescence patterns consistent with target TFBS binding detected."
                elif "Error" in prompt_str:
                    content = "Acknowledged error state."
                return AIMessage(content=content)

        logging.info("Using Simulated Scout LLM.")
        return SimulatedScoutLLM()
        # --- End Simulation ---
    except Exception as e:
        logging.error(f"Failed to initialize Bedrock LLM: {e}")
        # Log via UI API if possible, then raise or return None
        # log_event_via_ui_api("CRITICAL", f"Failed to initialize LLM: {e}") # Needs tool access
        return None

scout_llm = get_scout_llm()

# --- LangGraph State Definition ---
class HsiAnalysisState(TypedDict):
    """Represents the state of our HSI analysis graph."""
    user_id: str
    session_id: str
    hsi_data: Optional[dict] # Raw data from Voyage81/UI API
    processed_hsi_features: Optional[dict] # Output from Mamba preprocessing
    hyper_seq_context: str # Context for analysis
    scout_input_payload: Optional[dict] # Formatted input for Scout
    scout_raw_response: Optional[str] # Raw response from Scout
    analysis_result: Optional[dict] # Interpreted result
    error_message: Optional[str] # To capture errors during execution
    # LangGraph manages message history implicitly when using checkpointers and standard LCEL chains
    # messages: Annotated[Sequence[BaseMessage], operator.add]

# --- LangGraph Node Functions ---

async def get_hsi_data_node(state: HsiAnalysisState):
    """Node to fetch HSI data using the UI API tool."""
    logging.info("Node: get_hsi_data_node")
    get_hsi_tool = next(t for t in all_tools if t.name == "GetHSIDataViaUIAPI")
    try:
        # Assume tool execution happens via an agent executor or similar mechanism
        # Simplified direct call for pseudocode:
        result = await get_hsi_tool.ainvoke({"user_id": state["user_id"], "session_id": state["session_id"]})
        if result and result.get("status") == "success":
            return {"hsi_data": result.get("data")}
        else:
            return {"error_message": f"Failed to get HSI data: {result.get('message', 'Unknown error')}"}
    except Exception as e:
        return {"error_message": f"Exception in get_hsi_data_node: {e}"}

async def preprocess_hsi_node(state: HsiAnalysisState):
    """Node to preprocess HSI data using the Mamba tool."""
    logging.info("Node: preprocess_hsi_node")
    if state.get("error_message"): return {} # Skip if already errored
    if not state.get("hsi_data"): return {"error_message": "HSI data missing for preprocessing."}

    hsi_cube_placeholder = state["hsi_data"].get("hsi_cube", "missing_hsi_cube")
    mamba_tool = next(t for t in all_tools if t.name == "MambaHSIPreprocessor")
    try:
        result = await mamba_tool.ainvoke({"hsi_data_cube": hsi_cube_placeholder})
        if result and result.get("status") == "success":
            return {"processed_hsi_features": result}
        else:
            return {"error_message": f"Mamba preprocessing failed: {result.get('message', 'Unknown error')}"}
    except Exception as e:
        return {"error_message": f"Exception in preprocess_hsi_node: {e}"}

async def invoke_scout_node(state: HsiAnalysisState):
    """Node to format input and invoke the Scout BioMedical LLM."""
    logging.info("Node: invoke_scout_node")
    if state.get("error_message"): return {}
    if not state.get("processed_hsi_features"): return {"error_message": "Processed HSI features missing."}
    if not scout_llm: return {"error_message": "Scout LLM not initialized."}

    # Format the prompt for Scout (Simplified - Adapt based on fine-tuning)
    features = state["processed_hsi_features"]
    context = state["hyper_seq_context"]
    feature_summary = (
        f"HSI Analysis Summary:\n"
        f"- Spectral Patterns: {features.get('spectral_patterns', 'N/A')}\n"
        f"- Region Classification Hint: {features.get('classification_hint', 'N/A')}\n"
        f"Hyper-Seq Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI assistant specialized in interpreting hyperspectral imaging features for biomedical analysis, specifically inferring gene expression based on autofluorescence patterns defined by Hyper-Seq context."),
        ("human", f"Analyze the following HSI data summary to infer OXTR gene expression level. Data: {feature_summary}")
    ])

    # Using LangChain Expression Language (LCEL) to invoke
    chain = prompt | scout_llm
    try:
        response_message = await chain.ainvoke({}) # Input context is in the prompt template
        raw_response = response_message.content
        return {"scout_raw_response": raw_response}
    except Exception as e:
        return {"error_message": f"Exception invoking Scout LLM: {e}"}

async def interpret_result_node(state: HsiAnalysisState):
    """Node to interpret the raw response from Scout."""
    logging.info("Node: interpret_result_node")
    if state.get("error_message"): return {}
    if not state.get("scout_raw_response"): return {"error_message": "Scout response missing for interpretation."}

    raw_response = state["scout_raw_response"]
    # Reuse interpretation logic from previous script (Placeholder)
    try:
        # --- Simple Extraction Logic ---
        expression_level = None
        if "expression level is" in raw_response:
             try:
                parts = raw_response.split("expression level is")[1].split()
                level_desc = parts[0].lower()
                # ... (rest of parsing logic from previous script) ...
                for part in parts:
                    if part.startswith("(") and part.endswith(")"):
                        num_str = part.strip("()")
                        if "scale" not in num_str:
                           expression_level = float(num_str); break
             except Exception: expression_level = "Parsing Error" # Simplified
        # --- End Simple Extraction ---

        result = {
            "inferred_oxtr_expression": expression_level,
            "raw_response": raw_response,
            "interpretation_confidence": 0.9 # Placeholder
        }
        return {"analysis_result": result}
    except Exception as e:
        return {"error_message": f"Exception interpreting result: {e}"}

async def report_result_node(state: HsiAnalysisState):
    """Node to report the final result using the UI API tool."""
    logging.info("Node: report_result_node")
    if state.get("error_message"):
        # Log the error instead of reporting success
        log_tool = next((t for t in all_tools if t.name == "LogEventViaUIAPI"), None)
        if log_tool:
            await log_tool.ainvoke({"level": "ERROR", "message": "Workflow ended with error", "details": {"error": state["error_message"], "user_id": state["user_id"]}})
        return {} # End the graph

    if not state.get("analysis_result"): return {"error_message": "Analysis result missing for reporting."}

    report_tool = next((t for t in all_tools if t.name == "ReportAnalysisResultViaUIAPI"), None)
    if not report_tool: return {"error_message": "Reporting tool not found."}

    final_payload = {
        "user_id": state["user_id"],
        "session_id": state["session_id"],
        "analysis_result": state["analysis_result"],
        "hsi_metadata": state.get("hsi_data", {}).get("metadata"),
        "timestamp": datetime.datetime.utcnow().isoformat()
        # Add validation results if HyperSeq node was added
    }
    try:
        reporting_response = await report_tool.ainvoke({"analysis_payload": final_payload})
        if reporting_response.get("status") != "success":
            # Log failure to report
            log_tool = next((t for t in all_tools if t.name == "LogEventViaUIAPI"), None)
            if log_tool: await log_tool.ainvoke({"level": "ERROR", "message": "Failed to report final result", "details": {"response": reporting_response, "user_id": state["user_id"]}})
        # Even if reporting fails, we might end the graph successfully from analysis perspective
        return {}
    except Exception as e:
         return {"error_message": f"Exception reporting result: {e}"} # Error state

def should_continue(state: HsiAnalysisState) -> str:
    """Conditional edge logic: End if there's an error, otherwise continue."""
    if state.get("error_message"):
        logging.error(f"Workflow error detected: {state['error_message']}")
        # Potentially go to a specific error handling node first
        return "report_result_node" # Go to report/log the error then END
    else:
        # Determine next step based on current node completion
        # For a linear graph, just check if the required input for the next node exists
        if not state.get("hsi_data"): return "get_hsi_data_node" # Should not happen if START -> get_data
        if not state.get("processed_hsi_features"): return "preprocess_hsi_node"
        if not state.get("scout_raw_response"): return "invoke_scout_node"
        if not state.get("analysis_result"): return "interpret_result_node"
        return "report_result_node" # If analysis is done, report it

# --- Build LangGraph ---
workflow_builder = StateGraph(HsiAnalysisState)

workflow_builder.add_node("get_hsi_data", get_hsi_data_node)
workflow_builder.add_node("preprocess_hsi", preprocess_hsi_node)
workflow_builder.add_node("invoke_scout", invoke_scout_node)
workflow_builder.add_node("interpret_result", interpret_result_node)
workflow_builder.add_node("report_result", report_result_node) # Handles both success reporting and error logging before ending

# Define Edges
workflow_builder.add_edge(START, "get_hsi_data") # Entry point

# Conditional edges after each step to check for errors or decide next step
# Using a single conditional function for simplicity here
workflow_builder.add_conditional_edges(
    "get_hsi_data",
    should_continue,
    {"preprocess_hsi_node": "preprocess_hsi", "report_result_node": "report_result"} # Route to next step or error reporting
)
workflow_builder.add_conditional_edges(
    "preprocess_hsi",
    should_continue,
    {"invoke_scout_node": "invoke_scout", "report_result_node": "report_result"}
)
workflow_builder.add_conditional_edges(
    "invoke_scout",
    should_continue,
    {"interpret_result_node": "interpret_result", "report_result_node": "report_result"}
)
workflow_builder.add_conditional_edges(
    "interpret_result",
    should_continue,
    {"report_result_node": "report_result"} # Always go to report after interpretation (success or error in interpretation)
)
workflow_builder.add_edge("report_result", END) # Final step

# --- Memory / Checkpointer ---
# Using SqliteSaver for simple persistence. Use PostgreSQL for production (as in templates [15]).
memory = SqliteSaver.from_conn_string(":memory:") # In-memory for demo; use DB URL for persistence

# Compile the graph
hsi_analysis_graph = workflow_builder.compile(checkpointer=memory)

# --- FastAPI Application ---
app = FastAPI(
    title="Internet of Happiness - HSI Gene Expression Analysis",
    description="API using LangGraph, Scout BioMedical, and Mamba HSI models for gene expression inference.",
    version="0.2.0"
)

# --- API Request/Response Models ---
class HSIAnalysisRequest(BaseModel):
    user_id: str
    session_id: Optional[str] = None # Can be generated if not provided
    hyper_seq_context: str = "Default OXTR TFBS autofluorescence context" # Example context

class HSIAnalysisResponse(BaseModel):
    workflow_status: str # e.g., "completed", "error"
    analysis_result: Optional[dict] = None
    error_message: Optional[str] = None
    thread_id: str # To track the conversation/workflow instance

# --- FastAPI Endpoint ---
@app.post("/process_hsi", response_model=HSIAnalysisResponse)
async def process_hsi_endpoint(request: HSIAnalysisRequest):
    """
    Receives user info and context, triggers the LangGraph workflow
    for HSI analysis, and returns the result.
    """
    thread_id = request.session_id or str(uuid.uuid4())
    logging.info(f"Received request for user {request.user_id}, thread_id: {thread_id}")

    initial_state = HsiAnalysisState(
        user_id=request.user_id,
        session_id=thread_id,
        hyper_seq_context=request.hyper_seq_context,
        hsi_data=None,
        processed_hsi_features=None,
        scout_input_payload=None,
        scout_raw_response=None,
        analysis_result=None,
        error_message=None
    )

    # Configuration for LangGraph invocation, including thread_id for memory
    config = {"configurable": {"thread_id": thread_id}}

    try:
        # Asynchronously invoke the graph
        final_state = await hsi_analysis_graph.ainvoke(initial_state, config=config)

        if final_state.get("error_message"):
             logging.error(f"Workflow completed with error for thread {thread_id}: {final_state['error_message']}")
             return HSIAnalysisResponse(
                 workflow_status="error",
                 error_message=final_state["error_message"],
                 thread_id=thread_id
             )
        else:
             logging.info(f"Workflow completed successfully for thread {thread_id}")
             return HSIAnalysisResponse(
                 workflow_status="completed",
                 analysis_result=final_state.get("analysis_result"),
                 thread_id=thread_id
             )

    except Exception as e:
        logging.exception(f"Unhandled exception during graph invocation for thread {thread_id}: {e}")
        # Log critical failure via UI API if possible
        log_tool = next((t for t in all_tools if t.name == "LogEventViaUIAPI"), None)
        if log_tool:
            try:
                 await log_tool.ainvoke({"level": "CRITICAL", "message": "Unhandled FastAPI endpoint exception", "details": {"error": str(e), "thread_id": thread_id}})
            except: pass # Avoid secondary failure loop
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Root Redirect for Docs ---
from fastapi.responses import RedirectResponse
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# --- Main Execution ---
if __name__ == "__main__":
    # Conceptual Notes:
    # - SpeakEasy: Used externally to generate the MCP server for the Universal Informatics API from its OpenAPI spec [10, 11].
    # - MCP: Used *internally* via a LangChain MCP client/tool to interact with the UI API's MCP server [19]. Standardizes tool calls.
    # - A2A: Not directly used in this internal workflow. Would be relevant if this agent needed to coordinate *with other external agents* [2, 8, 9, 18]. Future enhancement.
    # - OpenAI Protocol: Refers generally to function calling/tool usage patterns popularized by OpenAI, which MCP standardizes further.

    logging.info("Starting FastAPI server for Internet of Happiness v2...")
    # Use reload=True only for development
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
