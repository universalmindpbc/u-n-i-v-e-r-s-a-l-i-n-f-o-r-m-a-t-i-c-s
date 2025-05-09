# atomic_prompt_generation.py

# import os

os.environ["LANGCHAIN_TRACING"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your_langsmith_key_here"
os.environ["LANGCHAIN_PROJECT"] = "your_project_name_here"

# Import your custom LangChain API wrappers.

# In production, these should wrap the actual endpoints and handle error management.
from my_langchain_tools import run_gpt, run_gemini, run_claude, unify_prompts

# === Replace these with your securely stored API keys ===
OPENAI_API_KEY = "your_openai_api_key_here"
GEMINI_API_KEY = "your_gemini_api_key_here"
CLAUDE_API_KEY = "your_claude_api_key_here"
PERPLEXITY_API_KEY = "your_perplexity_api_key_here"

def collect_human_input():
    """
    STEP 1: HUMAN INPUT
    Prompts the user to enter their bioinformatics or systems biology hypothesis.
    """
    hypothesis = input("Enter your bioinformatics/systems biology hypothesis:\n")
    return hypothesis

def universal_agentic_gateway():
    """
    STEP 1a: UNIVERSAL AGENTIC GATEWAY
    Registers this service as a publicly available agent by:
      - Publishing tool.json (MCP) and agent.json (A2A) metadata.
      - Establishing autonomous connections via MPC, A2A, or LangChain.
      - Enabling integration with connectors such as Zapier, Make, and n2n.
    Replace the following stub with actual registration calls.
    """
    print("Registering Universal Agentic Gateway... (stub)")
    # TODO: Implement agent registration using your agent metadata and API endpoints.
    # For example, a function call such as:
    # register_agent(api_key=YOUR_REGISTRATION_KEY, metadata=agent_metadata)

def generate_atomic_prompts(hypothesis):
    """
    STEP 2: ATOMIC PROMPT GENERATION
    In parallel, the following LLMs generate atomic super prompts:
      - GPT-4o (#deepresearch) generates 10 atomic prompts.
      - Perplexity Pro (Gemini 2.0 Flash or Gemini 2.5 Pro) generates 10 atomic prompts.
      - Claude Sonnet 3.7 generates 10 atomic prompts.
    This produces a total of 30 atomic prompts.
    """
    prompts_gpt = run_gpt(
        model="GPT-4o", 
        prompt=hypothesis, 
        num_prompts=10, 
        api_key=OPENAI_API_KEY
    )
    
    prompts_gemini = run_gemini(
        model="Gemini 2.0 Flash",  # or "Gemini 2.5 Pro", as applicable
        prompt=hypothesis, 
        num_prompts=10, 
        api_key=GEMINI_API_KEY
    )
    
    prompts_claude = run_claude(
        model="Claude Sonnet 3.7", 
        prompt=hypothesis, 
        num_prompts=10, 
        api_key=CLAUDE_API_KEY
    )
    
    # Combine all atomic prompts (total = 30)
    all_prompts = prompts_gpt + prompts_gemini + prompts_claude
    return all_prompts

def unify_prompts_to_wolfram(prompts):
    """
    STEP 3: UNIFICATION & LANGUAGE CONVERSION
    Uses Gemini 2.5 Pro Experimental to review the 30 atomic prompts
    and rewrite them into 10 unified Atomic Super Prompts in Wolfram Language.
    
    The function 'unify_prompts' accepts the list of prompts, the target output format,
    and the number of prompts to generate.
    """
    unified_prompts = unify_prompts(
        model="Gemini 2.5 Pro Experimental",
        prompts=prompts,
        output_format="Wolfram Language",
        num_prompts=10
        # Optionally, pass in additional API keys/configuration if required.

import json

with open("gemini_few_shot_examples.json", "r") as f:
    data = json.load(f)
    
few_shot_prompt = data["combined"]

# Then pass few_shot_prompt into your API call:
prompts_gemini = run_gemini(
    model="Gemini 2.5 Pro Experimental", 
    prompt=hypothesis,
    few_shot_example=few_shot_prompt,
    num_prompts=10,
    api_key=GEMINI_API_KEY
)
    return unified_prompts

def main():
    # Optional: Register the service with the Universal Agentic Gateway.
    universal_agentic_gateway()
    
    # STEP 1: Collect the bioinformatics hypothesis from the user.
    hypothesis = collect_human_input()
    
    # STEP 2: Generate atomic prompts from multiple LLMs in parallel.
    atomic_prompts = generate_atomic_prompts(hypothesis)
    print("Generated Atomic Prompts (Total = {}):".format(len(atomic_prompts)))
    for prompt in atomic_prompts:
        print(prompt)
    
    # STEP 3: Unify the atomic prompts into unified atomic super prompts in Wolfram Language.
    unified_prompts = unify_prompts_to_wolfram(atomic_prompts)
    print("\nUnified Atomic Super Prompts in Wolfram Language:")
    for up in unified_prompts:
        print(up)

if __name__ == "__main__":
    main()