from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np

# Just a simple test to verify imports are working
def test_imports():
    print("✅ All imports successful!")
    print("Environment is ready for bioinformatics agent development.")

if __name__ == "__main__":
    test_imports()
    
    pip install --upgrade --ignore-installed "pydantic>=2.6" "pydantic-core>=2.16"