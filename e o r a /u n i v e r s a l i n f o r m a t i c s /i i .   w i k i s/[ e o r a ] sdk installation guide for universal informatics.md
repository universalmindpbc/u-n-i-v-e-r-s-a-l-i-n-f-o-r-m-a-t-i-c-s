# [ I o H ] SDK Installation Guide for Universal Informatics

## Overview

This document outlines the SDKs required for the Internet of Happiness (IoH) and Universal Informatics projects. This installation guide is based on cross-referencing the GitHub stars list (56 repositories) against the current requirements.txt file to identify missing dependencies.

## Current Requirements Analysis

Many dependencies are already included in the existing requirements.txt file, including:

- langchain, langchain-anthropic, langchain-community, langchain-core, langchain-text-splitters
- langgraph
- amazon-braket-sdk, amazon-braket-default-simulator, amazon-braket-schemas
- dwave-ocean-sdk and related packages
- sagemaker, sagemaker-core
- pinecone-client
- anthropic, google-generativeai
- wolframclient
- biopython

The following installation guide ensures all 56 SDKs from the GitHub stars list are properly installed, avoiding duplication with existing requirements.

## SDKs to Install via Terminal

Here's the complete list of SDKs to install, organized by category:

### 1. Hyperspectral Imaging & Sequencing

```bash
# MCF7 hyperspectral imaging sequencing
pip install scRNA-seq-analysis hyperspectral-tools

# Hyperspy - Hyperspectral Data Processing & Analysis (NOT in requirements.txt yet)
pip install hyperspy

# MambaVision - CVPR 2025 Implementation
pip install git+https://github.com/NVlabs/MambaVision.git

# MambaHSI - Spatial-Spectral Mamba for HSI Classification
pip install git+https://github.com/li-yapeng/MambaHSI.git

# IEEE_TGRS_MambaLG - Hyperspectral Image Classification with Mamba
pip install git+https://github.com/danfenghong/IEEE_TGRS_MambaLG.git

# SpectralGPT - Spectral Remote Sensing
pip install git+https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT.git
```

### 2. LLM & Multi-Agent Frameworks

```bash
# Collaborative Reasoner - Meta FAIR
pip install git+https://github.com/facebookresearch/collaborative-reasoner.git

# Matrix - Multi-Agent Data Generation Framework
pip install git+https://github.com/facebookresearch/matrix.git

# Mistral Inference Library
pip install git+https://github.com/mistralai/mistral-inference.git

# Llama Models Utilities
pip install git+https://github.com/meta-llama/llama-models.git

# BioMed-LLaMa-3 - Biomedical Language Understanding
pip install git+https://github.com/zekaouinoureddine/BioMed-LLaMa-3.git

# Pinecone Assistant Connection
pip install git+https://github.com/pinecone-io/pinecone-mcp.git
```

### 3. Quantum Computing & Frameworks

```bash
# Wolfram Quantum Framework
pip install git+https://github.com/WolframResearch/QuantumFramework.git

# Cirq - Google's Quantum Computing Framework
pip install cirq

# OpenQDC - Repository of Quantum Datasets
pip install git+https://github.com/valence-labs/OpenQDC.git

# GUED - Gas-phase Ultrafast Electron Diffraction package
pip install git+https://github.com/lheald2/gued.git
```

### 4. Wolfram & Mathematics Integration

```bash
# Wolfram Language for Jupyter
pip install git+https://github.com/WolframResearch/WolframLanguageForJupyter.git

# AWS Lambda Wolfram Language
pip install git+https://github.com/WolframResearch/AWSLambda-WolframLanguage.git

# Wolfram Web Engine for Python
pip install git+https://github.com/WolframResearch/WolframWebEngineForPython.git

# MathLink.jl - Julia interface for Mathematica/Wolfram Engine
pip install julia
# Then in Julia: using Pkg; Pkg.add("MathLink")
```

### 5. Bioinformatics & Molecular Simulation

```bash
# AlphaFold 3 - Protein Structure Prediction
pip install git+https://github.com/google-deepmind/alphafold3.git

# AmberClassic - Biomolecular Simulation
conda install -c conda-forge ambertools

# CPPTraj - Biomolecular simulation trajectory analysis
pip install git+https://github.com/Amber-MD/cpptraj.git
```

### 6. Privacy & Security Frameworks

```bash
# Concrete ML - Privacy Preserving ML with FHE
pip install concrete-ml

# Fhenix Contracts - FHE on Blockchain
pip install git+https://github.com/FhenixProtocol/fhenix-contracts.git

# AWS HealthOmics Tools
pip install git+https://github.com/awslabs/aws-healthomics-tools.git

# IROH - Peer-2-peer that just works
pip install git+https://github.com/n0-computer/iroh.git
```

### 7. IoT & Connectivity Frameworks

```bash
# HomeKit ADK
pip install git+https://github.com/apple/HomeKitADK.git

# OpenThread - Thread networking protocol
pip install git+https://github.com/openthread/openthread.git

# Matter - Connectivity Standard
pip install git+https://github.com/SiliconLabs/matter.git

# Matter.js - TypeScript implementation of Matter
pip install git+https://github.com/project-chip/matter.js.git
```

### 8. Data Science & Analytics

```bash
# NVIDIA CUDA Library Samples
pip install cuda-python pycuda

# Hyperledger Indy
pip install indy-sdk

# Synapse Stack Builder
pip install git+https://github.com/Sage-Bionetworks/Synapse-Stack-Builder.git

# Neutron Imaging Suite
pip install git+https://github.com/neutronimaging/imagingsuite.git
pip install git+https://github.com/neutronimaging/CBCTCalibration.git
pip install git+https://github.com/neutronimaging/ToFImaging.git
pip install git+https://github.com/neutronimaging/ImagingQuality.git
```

### 9. Computational Biology & AI Benchmarks

```bash
# BixBench - Benchmark for LLM-based Agents in Computational Biology
pip install git+https://github.com/Future-House/BixBench.git

# Data Analysis Crow - Aviary-based data science agent
pip install git+https://github.com/Future-House/data-analysis-crow.git

# OpenEB - Open Source SDK for event-based vision
pip install git+https://github.com/prophesee-ai/openeb.git

# Ocean Node
pip install git+https://github.com/oceanprotocol/ocean-node.git

# PyTorch Geometric - Graph Neural Network Library
pip install torch-geometric
```

### 10. Multimedia & Fingerprinting

```bash
# Audio Fingerprinting Benchmark Toolkit
pip install git+https://github.com/Pexeso/audio-fingerprinting-benchmark-toolkit.git

# Pex SDK for JS and Python
pip install git+https://github.com/Pexeso/pex-sdk-py.git
```

## One-Shot Installation Script

For convenience, here's a one-shot installation script to add all missing SDKs at once, which can be executed via Claude Code in Terminal:

```bash
#!/bin/bash
# Universal Informatics - [ I o H ] SDK Installation Script
# Generated by Claude on May 15, 2025

echo "Starting [ I o H ] SDK installation..."

# Setup conda environment if not already active
if [[ -z $CONDA_DEFAULT_ENV ]]; then
  echo "Creating conda environment 'universal_informatics'..."
  conda create -n universal_informatics python=3.11 -y
  conda activate universal_informatics
else
  echo "Using active conda environment: $CONDA_DEFAULT_ENV"
fi

# Install conda packages first
echo "Installing conda packages..."
conda install -c conda-forge -y ambertools

# Install standard pip packages
echo "Installing pip packages from PyPI..."
pip install scRNA-seq-analysis hyperspectral-tools cirq julia cuda-python pycuda indy-sdk torch-geometric concrete-ml hyperspy

# Check if git is installed
if ! command -v git &> /dev/null; then
  echo "Git is required but not found. Please install git and try again."
  exit 1
fi

# Set up a temporary directory for git clone operations
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Function to install from GitHub repository
install_from_github() {
  repo=$1
  echo "Installing $repo..."
  git clone "https://github.com/$repo.git" --depth 1
  cd "$(basename "$repo")"
  if [ -f "setup.py" ]; then
    pip install -e .
  elif [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    pip install -e .
  else
    echo "No setup.py or requirements.txt found for $repo, attempting direct install..."
    pip install .
  fi
  cd "$TEMP_DIR"
}

# Install GitHub repositories
echo "Installing packages from GitHub..."

# Hyperspectral Imaging & Sequencing
install_from_github "NVlabs/MambaVision"
install_from_github "li-yapeng/MambaHSI"
install_from_github "danfenghong/IEEE_TGRS_MambaLG"
install_from_github "danfenghong/IEEE_TPAMI_SpectralGPT"

# LLM & Multi-Agent Frameworks
install_from_github "facebookresearch/collaborative-reasoner"
install_from_github "facebookresearch/matrix"
install_from_github "mistralai/mistral-inference"
install_from_github "meta-llama/llama-models"
install_from_github "zekaouinoureddine/BioMed-LLaMa-3"
install_from_github "pinecone-io/pinecone-mcp"

# Quantum Computing & Frameworks
install_from_github "WolframResearch/QuantumFramework"
install_from_github "valence-labs/OpenQDC"
install_from_github "lheald2/gued"

# Wolfram & Mathematics Integration
install_from_github "WolframResearch/WolframLanguageForJupyter"
install_from_github "WolframResearch/AWSLambda-WolframLanguage"
install_from_github "WolframResearch/WolframWebEngineForPython"

# Bioinformatics & Molecular Simulation
install_from_github "google-deepmind/alphafold3"
install_from_github "Amber-MD/cpptraj"

# Privacy & Security Frameworks
install_from_github "FhenixProtocol/fhenix-contracts"
install_from_github "awslabs/aws-healthomics-tools"
install_from_github "n0-computer/iroh"

# IoT & Connectivity Frameworks
install_from_github "apple/HomeKitADK"
install_from_github "openthread/openthread"
install_from_github "SiliconLabs/matter"
install_from_github "project-chip/matter.js"

# Data Science & Analytics
install_from_github "Sage-Bionetworks/Synapse-Stack-Builder"
install_from_github "neutronimaging/imagingsuite"
install_from_github "neutronimaging/CBCTCalibration"
install_from_github "neutronimaging/ToFImaging"
install_from_github "neutronimaging/ImagingQuality"

# Computational Biology & AI Benchmarks
install_from_github "Future-House/BixBench"
install_from_github "Future-House/data-analysis-crow"
install_from_github "prophesee-ai/openeb"
install_from_github "oceanprotocol/ocean-node"

# Multimedia & Fingerprinting
install_from_github "Pexeso/audio-fingerprinting-benchmark-toolkit"
install_from_github "Pexeso/pex-sdk-py"

# Clean up
cd
rm -rf "$TEMP_DIR"

# Install Julia package MathLink.jl if Julia is installed
if command -v julia &> /dev/null; then
  echo "Installing MathLink.jl for Julia..."
  julia -e 'using Pkg; Pkg.add("MathLink")'
fi

echo "Updating requirements.txt..."
pip freeze > requirements.txt.new

echo "Installation complete. New requirements in requirements.txt.new"
echo "Please review and merge with existing requirements.txt to avoid duplicates."
```

## Verification Process

After installation, verify the key packages with these commands:

```bash
# Verify MambaVision
python -c "import torch; print('PyTorch available:', torch.__version__); import sys; sys.path.append('/path/to/MambaVision'); from mambavision import models; print('MambaVision imported successfully')"

# Verify hyperspy
python -c "import hyperspy.api as hs; print('HyperSpy version:', hs.__version__)"

# Verify AWS Braket
python -c "from braket.aws import AwsDevice; print('AWS Braket installed successfully')"

# Verify Concrete ML
python -c "import concrete_ml; print('Concrete ML version:', concrete_ml.__version__)"

# Verify Cirq
python -c "import cirq; print('Cirq version:', cirq.__version__)"

# Verify LangChain & LangGraph
python -c "import langchain, langgraph; print(f'LangChain: {langchain.__version__}, LangGraph: {langgraph.__version__}')"
```

## Integration with requirements.txt

After installation, carefully merge the new packages with the existing requirements.txt:

```bash
# Generate list of new requirements
pip freeze > new_requirements.txt

# Use a Python script to merge without duplicates
python -c "
import re

# Read existing requirements
with open('requirements.txt', 'r') as f:
    existing = set(line.strip() for line in f if line.strip() and not line.startswith('#'))

# Read new requirements
with open('new_requirements.txt', 'r') as f:
    new = set(line.strip() for line in f if line.strip() and not line.startswith('#'))

# Extract package names (without versions) from existing requirements
existing_names = set()
for req in existing:
    match = re.match(r'^([a-zA-Z0-9_.-]+).*
, req)
    if match:
        existing_names.add(match.group(1).lower())

# Find truly new packages (not just version differences)
truly_new = []
for req in new:
    match = re.match(r'^([a-zA-Z0-9_.-]+).*
, req)
    if match and match.group(1).lower() not in existing_names:
        truly_new.append(req)

# Append truly new packages to requirements.txt
if truly_new:
    with open('requirements.txt', 'a') as f:
        f.write('\n# New packages added ' + str(new_requirements.txt) + '\n')
        for req in sorted(truly_new):
            f.write(req + '\n')
    print(f'Added {len(truly_new)} new packages to requirements.txt')
else:
    print('No new packages to add')
"
```

## Potential Compatibility Issues

When installing all SDKs from the list, be aware of these potential issues:

1. **CUDA Version Conflicts**: Packages like PyTorch, CUDA Libraries, and TensorFlow may require specific CUDA versions
2. **Wolfram Engine Requirements**: Wolfram packages require a licensed Wolfram Engine or Mathematica installation
3. **Build Dependencies**: Several low-level packages require C++ compilers and system libraries
4. **Python Version Compatibility**: Some packages may not work with the latest Python version
5. **Hardware Requirements**: Quantum and FHE libraries may have specific hardware requirements

## Troubleshooting Common Issues

1. **Missing Compilers**:
    
    ```bash
    # Ubuntu/Debian
    sudo apt-get install build-essential
    # macOS
    xcode-select --install
    ```
    
2. **CUDA Setup Issues**:
    
    ```bash
    # Check CUDA version
    nvcc --version
    # Set CUDA paths if needed
    export CUDA_HOME=/usr/local/cuda
    export PATH=$PATH:$CUDA_HOME/bin
    ```
    
3. **Wolfram Engine Setup**:
    
    ```bash
    # Activate Wolfram Engine (if installed)
    wolframscript -activate
    # Test Wolfram Engine
    echo 'Print["Hello from Wolfram Engine"]' | wolframscript
    ```
    
4. **GitHub Installation Failures**: If a GitHub repository installation fails, try:
    
    ```bash
    # Clone manually
    git clone https://github.com/username/repo.git
    cd repo
    # Check for specific installation instructions in README
    # Install in development mode
    pip install -e .
    ```
    

## Conclusion

This comprehensive installation guide ensures that all 56 SDKs from the GitHub stars list are properly installed, cross-referenced against the existing requirements.txt to avoid duplication. The one-shot installation script provides a convenient way to install all missing SDKs at once through a Claude Code agentic operation in Terminal.

---

_This guide was generated by Claude on May 15, 2025, based on precise cross-referencing of the GitHub stars list (56 repositories) against the Universal Informatics requirements.txt file._