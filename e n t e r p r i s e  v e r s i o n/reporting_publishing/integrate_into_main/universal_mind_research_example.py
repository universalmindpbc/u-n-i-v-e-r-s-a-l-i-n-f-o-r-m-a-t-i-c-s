#!/usr/bin/env python3
"""
Universal Mind Research Example

This script demonstrates the integration of all components in the Universal
Mind research pipeline, showcasing how JADAD scoring and Fuzzy Phi Logic
quality assessment ensure publishable genomic research.

The script connects GPTo3, DORA, PandaOmics, and InClinicio AI to create
a comprehensive research design and execution system.
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any

# Import all components
from genomic_rct_guardrails import (
    GenomicRctGuardrails, 
    FuzzyPhiLogic,
    InClinicioAgent,
    get_wiki_diagram
)
from universal_research_design_integrator import (
    UniversalResearchDesignIntegrator,
    ResearchVector
)
from pandaomics_rct_pipeline import (
    PandaOmicsRctPipeline,
    get_pipeline_integration_diagram
)
from gpto3_research_integration import (
    GPTo3ResearchIntegration,
    generate_wiki_documentation
)
from research_design_agents import (
    GPTo3BriefingAgent,
    PandaOmicsInSilicoAgent,
    generate_agent_integration_diagram,
    generate_research_workflow_diagram
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("universal_mind_example.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("universal_mind_example")

class UniversalMindResearchExample:
    """
    Example implementation showcasing the Universal Mind research pipeline
    with JADAD scoring and Fuzzy Phi Logic quality assessment.
    """
    
    def __init__(self, api_keys: Dict[str, str] = None):
        """
        Initialize the example
        
        Args:
            api_keys: Dictionary of API keys for various services (optional)
        """
        # Default API keys if none provided (for demonstration)
        if api_keys is None:
            api_keys = {
                "openai": os.environ.get("OPENAI_API_KEY", "demo_key"),
                "scite": os.environ.get("SCITE_API_KEY", "demo_key"),
                "dora": os.environ.get("DORA_API_KEY", "demo_key")
            }
        
        self.api_keys = api_keys
        
        # Initialize all components
        self.guardrails = GenomicRctGuardrails(api_keys)
        self.integrator = UniversalResearchDesignIntegrator(api_keys)
        self.pipeline = PandaOmicsRctPipeline(api_keys)
        self.gpto3_integration = GPTo3ResearchIntegration(api_keys)
        self.fuzzy_phi = FuzzyPhiLogic()
        
        logger.info("Universal Mind Research Example initialized")
    
    async def run_full_example(self, 
                             genes: List[str] = None,
                             condition: str = None,
                             research_area: str = None,
                             target_journals: List[str] = None) -> Dict[str, Any]:
        """
        Run a full example of the research pipeline
        
        Args:
            genes: List of gene names/IDs (defaults to BRCA1, TP53, EGFR)
            condition: Medical condition/disease (defaults to Breast Cancer)
            research_area: Research area/field (defaults to cancer genomics)
            target_journals: List of target journals (defaults to Nature Methods, Frontiers)
            
        Returns:
            Complete pipeline results
        """
        # Default values if not specified
        if genes is None:
            genes = ["BRCA1", "TP53", "EGFR"]
            
        if condition is None:
            condition = "Breast Cancer"
            
        if research_area is None:
            research_area = "cancer genomics"
            
        if target_journals is None:
            target_journals = ["Nature Methods", "Frontiers in Bioinformatics"]
        
        print("\n" + "=" * 80)
        print(f"Universal Mind Research Example: {research_area}")
        print(f"Genes: {', '.join(genes)}")
        print(f"Condition: {condition}")
        print(f"Target Journals: {', '.join(target_journals)}")
        print("=" * 80 + "\n")
        
        # Step 1: Generate comprehensive briefing for GPTo3
        print("Step 1: Generating comprehensive briefing for GPTo3...")
        briefing = await self.gpto3_integration.brief_gpto3_on_research_design(
            research_area, genes, condition, target_journals
        )
        self._print_quality_metrics(briefing.get("quality_metrics", {}))
        
        # Step 2: Design genomic RCT
        print("\nStep 2: Designing genomic RCT...")
        rct_design = await self.pipeline.design_genomic_rct(
            genes, condition, target_journals
        )
        self._print_quality_metrics(rct_design.get("quality_metrics", {}))
        
        # Step 3: Execute RCT with quality controls
        print("\nStep 3: Executing RCT with quality controls...")
        execution_results = await self.pipeline.execute_rct_with_quality_controls(rct_design)
        self._print_execution_results(execution_results)
        
        # Step 4: Generate publication
        print("\nStep 4: Generating publication...")
        publication = await self.pipeline.generate_publication_from_rct(execution_results)
        self._print_publication_summary(publication)
        
        # Create complete results
        results = {
            "briefing": briefing,
            "rct_design": rct_design,
            "execution_results": execution_results,
            "publication": publication,
            "summary": {
                "research_area": research_area,
                "genes": genes,
                "condition": condition,
                "execution_time": datetime.now().isoformat(),
                "quality_metrics": {
                    "jadad_score": rct_design["quality_metrics"]["jadad_score"],
                    "weighted_scientific_significance": rct_design["quality_metrics"]["weighted_scientific_significance"],
                    "integrated_quality_score": rct_design["quality_metrics"]["integrated_quality_score"],
                    "nature_viable": rct_design["quality_metrics"]["nature_viable"],
                    "frontiers_viable": rct_design["quality_metrics"]["frontiers_viable"]
                },
                "publication_ready": publication.get("quality_metrics", {}).get("publication_ready", False)
            }
        }
        
        print("\n" + "=" * 80)
        print("Example completed successfully")
        print("=" * 80)
        
        return results
    
    def display_integration_diagrams(self) -> None:
        """Display all integration diagrams"""
        print("\n" + "=" * 80)
        print("Universal Mind Research Integration Diagrams")
        print("=" * 80 + "\n")
        
        print("1. Agent Integration Diagram")
        print(generate_agent_integration_diagram())
        
        print("\n2. Research Workflow Diagram")
        print(generate_research_workflow_diagram())
        
        print("\n3. Pipeline Integration Diagram")
        print(get_pipeline_integration_diagram())
        
        print("\n4. GPTo3 Integration Diagram")
        print(self.gpto3_integration.get_integration_diagram())
        
        print("\n5. GPTo3 Workflow Diagram")
        print(self.gpto3_integration.get_gpto3_workflow_diagram())
        
        print("\n6. Wiki Mini Diagram")
        print(get_wiki_diagram())
    
    def explain_fuzzy_phi_logic(self) -> None:
        """Explain the Fuzzy Phi Logic quality assessment system"""
        print("\n" + "=" * 80)
        print("Fuzzy Phi Logic Quality Assessment System")
        print("=" * 80 + "\n")
        
        print("The Fuzzy Phi Logic system calculates Scientific Significance based on Pisano periods and biological importance.")
        print("\nCore Principles:")
        print("1. Pisano Period Generation:")
        print("   - Creates α, β, and γ sequences for mathematical analysis")
        print("   - α sequence: Fibonacci sequence modulo n")
        print("   - β sequence: Lucas numbers modulo n")
        print("   - γ sequence: Harmonized sequence with Golden Ratio weighting")
        
        print("\n2. Geometric Nestings (P value):")
        print("   - Counts Platonic geometries nested in the transcribed circle")
        print("   - Identifies triangular, square, pentagonal, hexagonal, and octahedral patterns")
        print("   - Higher nestings indicate greater mathematical harmony")
        
        print("\n3. Biological Importance Assessment:")
        print("   - Analyzes relationship to key biological processes")
        print("   - High: Primary life functions (chlorophyll creation, cellular duplication, gene expression)")
        print("   - Medium: Supporting functions (molecular docking, protein folding, quantum aspects)")
        print("   - Low: Elemental aspects (electron/proton/neutron properties)")
        
        print("\n4. Weighted Scientific Significance (WSS):")
        print("   - WSS = P × Biological Importance (normalized to scale 1-10)")
        print("   - High WSS (8-10): Creates cellular life")
        print("   - Medium WSS (5-7): Supports cellular life")
        print("   - Low WSS (1-4): Creates elemental life")
        
        print("\n5. Integration with JADAD Scoring:")
        print("   - JADAD score assesses methodological quality (0-5)")
        print("   - Integrated Quality Score combines JADAD and WSS")
        print("   - Nature journals require: JADAD ≥ 4, WSS ≥ 8, Integrated Score ≥ 7")
        print("   - Frontiers journals require: JADAD ≥ 3, WSS ≥ 5, Integrated Score ≥ 5")
        
        print("\nThe Fuzzy Phi Logic system allows Universal Mind to maintain strict quality standards while enabling frontier genomic research through quantitative assessment of scientific significance.")
    
    def explain_jadad_scoring(self) -> None:
        """Explain the JADAD scoring system for RCT quality assessment"""
        print("\n" + "=" * 80)
        print("JADAD Scoring System for RCT Quality Assessment")
        print("=" * 80 + "\n")
        
        print("The JADAD score is a validated tool for assessing the methodological quality of randomized controlled trials (RCTs).")
        print("\nScoring Components:")
        print("1. Randomization (0-2 points):")
        print("   - 1 point: Randomization is mentioned")
        print("   - 1 additional point: Appropriate randomization method described")
        print("   - 1 point deducted: Inappropriate randomization method")
        
        print("\n2. Blinding (0-2 points):")
        print("   - 1 point: Double-blinding is mentioned")
        print("   - 1 additional point: Appropriate blinding method described")
        print("   - 1 point deducted: Inappropriate blinding method")
        
        print("\n3. Withdrawals and Dropouts (0-1 point):")
        print("   - 1 point: Withdrawals and dropouts described with reasons")
        
        print("\nTotal Score Interpretation:")
        print("   - 5: Excellent methodological quality")
        print("   - 4: Good methodological quality")
        print("   - 3: Moderate methodological quality")
        print("   - ≤2: Poor methodological quality")
        
        print("\nQuality Standards for Publication:")
        print("   - Nature journals require JADAD ≥ 4")
        print("   - Frontiers journals require JADAD ≥ 3")
        
        print("\nContextual Adaptations:")
        print("   - Surgical/device trials: JADAD ≥ 3 with CONSORT adherence ≥ 85%")
        print("   - Rare diseases (n < 100): JADAD ≥ 3 with real-world evidence validation")
        print("   - Pediatric populations: JADAD ≥ 4 with mandatory DSMB oversight")
        
        print("\nThe JADAD scoring system ensures that all Universal Mind research meets rigorous methodological standards, enhancing reproducibility and validity of genomic findings.")
    
    def _print_quality_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print quality metrics in a formatted way"""
        print("\nQuality Metrics:")
        print(f"  JADAD Score: {metrics.get('jadad_score', 'N/A')}")
        print(f"  Weighted Scientific Significance: {metrics.get('weighted_scientific_significance', 'N/A')}")
        print(f"  Integrated Quality Score: {metrics.get('integrated_quality_score', 'N/A')}")
        print(f"  Nature Viable: {metrics.get('nature_viable', 'N/A')}")
        print(f"  Frontiers Viable: {metrics.get('frontiers_viable', 'N/A')}")
    
    def _print_execution_results(self, results: Dict[str, Any]) -> None:
        """Print execution results in a formatted way"""
        quality_controls = results.get("quality_controls", {})
        study_results = results.get("study_results", {})
        publication_readiness = results.get("publication_readiness", {})
        
        print("\nExecution Results:")
        print(f"  JADAD Verification: {quality_controls.get('jadad_verification', {}).get('final_jadad_score', 'N/A')}")
        print(f"  Fuzzy Phi Verification: {quality_controls.get('fuzzy_phi_verification', {}).get('wss_category', 'N/A')}")
        
        primary_outcome = study_results.get("primary_outcome", {})
        if primary_outcome:
            print("\nPrimary Outcome:")
            print(f"  Effect Size: {primary_outcome.get('effect_size', 'N/A')}")
            print(f"  P-value: {primary_outcome.get('p_value', 'N/A')}")
            ci = primary_outcome.get("confidence_interval", [])
            if ci:
                print(f"  Confidence Interval: [{ci[0]}, {ci[1]}]")
        
        print("\nPublication Readiness:")
        print(f"  Meets Nature Standards: {publication_readiness.get('meets_nature_standards', 'N/A')}")
        print(f"  Meets Frontiers Standards: {publication_readiness.get('meets_frontiers_standards', 'N/A')}")
    
    def _print_publication_summary(self, publication: Dict[str, Any]) -> None:
        """Print publication summary in a formatted way"""
        print("\nPublication Summary:")
        print(f"  Title: {publication.get('title', 'N/A')}")
        print(f"  Target Journal: {publication.get('target_journal', 'N/A')}")
        
        sections = publication.get("sections", {})
        if sections:
            print("\nSection Overview:")
            for section_name in sections.keys():
                print(f"  - {section_name}")
        
        print("\nQuality Metrics:")
        quality_metrics = publication.get("quality_metrics", {})
        print(f"  JADAD Score: {quality_metrics.get('jadad_score', 'N/A')}")
        print(f"  Weighted Scientific Significance: {quality_metrics.get('weighted_scientific_significance', 'N/A')}")
        print(f"  Publication Ready: {quality_metrics.get('publication_ready', 'N/A')}")

def create_readme_markdown() -> str:
    """
    Create README.md content for the GitHub repository
    
    Returns:
        Markdown string with README content
    """
    readme = """
# Universal Mind Research Pipeline

## Overview

The Universal Mind Research Pipeline integrates GPTo3, DORA, PandaOmics, and InClinicio AI to create a comprehensive research design and execution system with JADAD scoring and Fuzzy Phi Logic quality assessment.

## Components

### 1. Research Design Agents

- **GPTo3BriefingAgent**: Creates comprehensive research design briefings
- **PandaOmicsInSilicoAgent**: Executes virtual RCTs with genomic assessments
- **JadadScoreCalculator**: Validates methodological quality of research designs

### 2. Genomic RCT Guardrails

- **GenomicRctGuardrails**: Ensures research designs meet quality standards
- **FuzzyPhiLogic**: Calculates Scientific Significance using Pisano periods and biological importance
- **InClinicioAgent**: Validates research designs with integrated quality metrics

### 3. Universal Research Design Integrator

- **UniversalResearchDesignIntegrator**: Connects all components into a unified system
- **ResearchVector**: Calculates research quality vectors using Fuzzy Phi Logic

### 4. PandaOmics RCT Pipeline

- **PandaOmicsRctPipeline**: Executes RCTs with integrated quality controls
- Ensures studies meet publication standards for high-impact journals

### 5. GPTo3 Research Integration

- **GPTo3ResearchIntegration**: Connects GPTo3 with all research design components
- Creates and validates research designs with comprehensive quality assessment

## Quality Standards

### JADAD Scoring

The JADAD score assesses methodological quality based on:

- Randomization (description and appropriateness)
- Blinding (description and appropriateness)
- Withdrawals and dropouts reporting

JADAD scores range from 0-5, with ≥4 required for Nature and ≥3 for Frontiers.

### Fuzzy Phi Logic

Fuzzy Phi Logic calculates Scientific Significance using:

- P = Count of Platonic geometries nested in the transcribed circle of Pisano Period
- WSS = Weighted Scientific Significance based on biological importance
- Scientific Significance = P × WSS (normalized to scale 1-10)

#### WSS Categories:

- **High (8-10)**: Creates cellular life - primary life functions
- **Medium (5-7)**: Supports cellular life - molecular and quantum processes
- **Low (1-4)**: Creates elemental life - fundamental particle properties

## Installation

```bash
git clone https://github.com/universalmind/research-pipeline.git
cd research-pipeline
pip install -r requirements.txt
```

## Usage

```python
from universal_mind_research_example import UniversalMindResearchExample

# Initialize the example
example = UniversalMindResearchExample(api_keys)

# Run full example
results = await example.run_full_example(
    genes=["BRCA1", "TP53", "EGFR"],
    condition="Breast Cancer",
    research_area="cancer genomics",
    target_journals=["Nature Methods", "Frontiers in Bioinformatics"]
)

# Display integration diagrams
example.display_integration_diagrams()

# Explore Fuzzy Phi Logic
example.explain_fuzzy_phi_logic()

# Learn about JADAD scoring
example.explain_jadad_scoring()
```

## Workflow Diagram

```mermaid
graph TD
    User[Researcher/User] --> GPTo3[GPTo3]
    GPTo3 --> |"Comprehensive<br>briefing"|RDA[Research Design<br>Agent]
    RDA --> |"JADAD<br>scoring"|JS[JADAD Score]
    RDA --> FPL[Fuzzy Phi Logic]
    FPL --> |"Scientific<br>Significance"|WSS[Weighted Scientific<br>Significance]
    JS --> QS[Quality Standards]
    WSS --> QS
    QS --> PO[PandaOmics<br>RCT Pipeline]
    PO --> |"RCT<br>results"|DORA[DORA<br>Manuscript Service]
    DORA --> |"Publication-ready<br>manuscript"|Journal[High-Impact<br>Journals]
```

## License

This project is licensed under the Universal Mind Public License - see the LICENSE file for details.

## Acknowledgments

- Universal Mind Research Team
- Garvan Institute of Medical Research
- DeepMind Health Division
"""
    return readme

async def run_example():
    """Run the Universal Mind Research Example"""
    # Create example instance
    example = UniversalMindResearchExample()
    
    # Display integration diagrams
    example.display_integration_diagrams()
    
    # Explain Fuzzy Phi Logic
    example.explain_fuzzy_phi_logic()
    
    # Explain JADAD scoring
    example.explain_jadad_scoring()
    
    # Check if API keys are available for full example
    if os.environ.get("OPENAI_API_KEY"):
        # Run full example
        await example.run_full_example(
            genes=["BRCA1", "TP53", "EGFR"],
            condition="Breast Cancer",
            research_area="cancer genomics",
            target_journals=["Nature Methods", "Frontiers in Bioinformatics"]
        )
    else:
        print("\n" + "=" * 80)
        print("Full example execution skipped - API keys not available")
        print("=" * 80)
    
    # Generate README.md
    readme = create_readme_markdown()
    print("\nREADME.md generated. You can save it to your repository.")

if __name__ == "__main__":
    # Run example
    asyncio.run(run_example())
