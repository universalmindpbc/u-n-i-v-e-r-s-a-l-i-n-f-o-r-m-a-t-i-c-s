#!/usr/bin/env python3
"""
Universal Research Design Integrator for Universal Informatics

This module integrates the JADAD and IF scoring system with Fuzzy Phi Logic
to ensure research designs meet Universal Mind's strict guardrails while
allowing discretion for frontier genomic research through quality scoring.

The module connects DORA, GPTo3, PandaOmics, and InClinicio AI to form
a comprehensive research design validation pipeline.
"""

import os
import json
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Set
from datetime import datetime
from enum import Enum
from pathlib import Path

# Import research design components
from research_design_agents import (
    ResearchDesignAgent, 
    PandaOmicsInSilicoAgent, 
    GPTo3BriefingAgent,
    JadadScoreCalculator,
    ImpactFactorCalculator,
    generate_agent_integration_diagram,
    generate_research_workflow_diagram
)

# Import Genomic RCT Guardrails
from genomic_rct_guardrails import (
    GenomicRctGuardrails,
    FuzzyPhiLogic,
    InClinicioAgent,
    get_wiki_diagram
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("universal_research_design.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("universal_research_design")

class ResearchVector:
    """
    Research vector calculation for genomic research quality assessment using
    the Platonic geometry and biological relationship concepts from Fuzzy Phi Logic.
    """
    
    def __init__(self):
        """Initialize Research Vector calculator"""
        self.fuzzy_phi = FuzzyPhiLogic()
    
    async def calculate_vector(self, 
                             genomic_data: Dict[str, Any],
                             research_design: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate the research vector for a genomic study
        
        Args:
            genomic_data: Genomic research data
            research_design: Research design parameters
            
        Returns:
            Research vector components and quality score
        """
        # Get scientific significance using Fuzzy Phi Logic
        significance = self.fuzzy_phi.calculate_scientific_significance(genomic_data)
        
        # Calculate quality components
        p_value = significance["geometric_nestings"]  # Platonic geometries
        wss_value = significance["weighted_scientific_significance"]  # WSS
        
        # Calculate Scientific Significance using the Fuzzy Phi Logic formula
        scientific_significance = p_value * wss_value
        
        # Normalize to scale of 1-10
        normalized_ss = min(10, max(1, round(scientific_significance)))
        
        return {
            "platonic_geometries": p_value,
            "weighted_scientific_significance": wss_value,
            "scientific_significance": normalized_ss,
            "research_vector_components": {
                "alpha_sequence": significance["pisano_period"]["alpha_sequence"],
                "beta_sequence": significance["pisano_period"]["beta_sequence"],
                "gamma_sequence": significance["pisano_period"]["gamma_sequence"]
            },
            "wss_category": significance["wss_category"],
            "wss_description": significance["wss_description"]
        }

class UniversalResearchDesignIntegrator:
    """
    Integrates all research design components to ensure publishable research
    through a comprehensive quality assessment system.
    
    This integrator connects:
    1. DORA for manuscript generation
    2. GPTo3 for research design briefing
    3. PandaOmics for InSilico AI pipeline
    4. InClinicio AI for RCT validation
    5. Fuzzy Phi Logic for quality assessment
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        """
        Initialize the Universal Research Design Integrator
        
        Args:
            api_keys: Dictionary of API keys for various services
        """
        self.api_keys = api_keys
        
        # Initialize core components
        self.research_design_agent = ResearchDesignAgent(api_keys)
        self.pandaomics_agent = PandaOmicsInSilicoAgent(api_keys)
        self.gpto3_agent = GPTo3BriefingAgent(
            api_keys, 
            self.research_design_agent, 
            self.pandaomics_agent
        )
        
        # Initialize guardrail system
        self.guardrails = GenomicRctGuardrails(api_keys)
        
        # Initialize research vector calculator
        self.research_vector = ResearchVector()
        
        logger.info("Universal Research Design Integrator initialized")
    
    async def create_integrated_research_design(self,
                                             research_area: str,
                                             genes: List[str],
                                             condition: str,
                                             target_journals: List[str] = None) -> Dict[str, Any]:
        """
        Create an integrated research design with all quality controls
        
        Args:
            research_area: Research area/field
            genes: List of gene names/IDs
            condition: Medical condition/disease
            target_journals: List of target journals (defaults to Nature and Frontiers)
            
        Returns:
            Comprehensive research design with quality metrics
        """
        logger.info(f"Creating integrated research design for {research_area}")
        
        # Default journals if not specified
        if target_journals is None:
            target_journals = ["Nature Methods", "Frontiers in Bioinformatics"]
        
        # Step 1: Get initial study design from ResearchDesignAgent
        study_design = await self.research_design_agent.design_study(research_area, target_journals)
        logger.info(f"Initial study design created with JADAD score: {study_design.get('current_jadad_score', 0)}")
        
        # Step 2: Execute In Silico RCT with PandaOmics
        in_silico_results = await self.pandaomics_agent.execute_in_silico_rct(genes, condition)
        logger.info(f"In Silico RCT executed with JADAD score: {in_silico_results.get('jadad_score', 0)}")
        
        # Step 3: Create genomic data structure
        genomic_data = {
            "genes": genes,
            "biological_processes": [
                "Gene expression regulation",
                "DNA methylation",
                "Transcription",
                "Cell division",
                "Signal transduction"
            ],
            "molecular_functions": [
                "Protein binding",
                "DNA binding",
                "Enzyme activity",
                "Molecular docking",
                "Protein folding"
            ],
            "cellular_components": [
                "Nucleus",
                "Cytoplasm",
                "Membrane",
                "Mitochondrion",
                "Endoplasmic reticulum"
            ]
        }
        
        # Step 4: Calculate research vector
        vector = await self.research_vector.calculate_vector(genomic_data, study_design)
        logger.info(f"Research vector calculated with WSS: {vector['weighted_scientific_significance']}")
        
        # Step 5: Validate with guardrails
        validation = await self.guardrails.validate_research_design(
            study_design.get("research_design", {}),
            genomic_data
        )
        logger.info(f"Research design validated with quality score: {validation.get('integrated_quality_score', 0)}")
        
        # Step 6: Create briefing for GPTo3
        briefing = await self.guardrails.brief_gpto3(
            research_area,
            genes,
            condition,
            target_journals
        )
        logger.info("Comprehensive briefing created for GPTo3")
        
        # Step 7: Integrate all components into a unified research design
        integrated_design = {
            "research_metadata": {
                "area": research_area,
                "genes": genes,
                "condition": condition,
                "target_journals": target_journals,
                "created_at": datetime.now().isoformat()
            },
            "quality_metrics": {
                "jadad_score": validation["jadad_score"],
                "impact_factor_potential": validation["potential_impact_factor"],
                "weighted_scientific_significance": vector["weighted_scientific_significance"],
                "integrated_quality_score": validation["integrated_quality_score"],
                "nature_viable": validation["nature_viable"],
                "frontiers_viable": validation["frontiers_viable"]
            },
            "research_design": study_design.get("research_design", {}),
            "in_silico_validation": {
                "rct_design": in_silico_results.get("rct_design", {}),
                "primary_outcome": in_silico_results.get("primary_outcome", {}),
                "gene_specific_results": in_silico_results.get("gene_specific_results", {})
            },
            "fuzzy_phi_assessment": {
                "research_vector": vector,
                "pisano_period": validation["scientific_significance"]["pisano_period"],
                "geometric_nestings": validation["scientific_significance"]["geometric_nestings"],
                "wss_category": vector["wss_category"],
                "wss_description": vector["wss_description"]
            },
            "publication_strategy": {
                "viable_journals": validation["viable_journal_targets"],
                "key_points": briefing.get("publication_strategy", {}).get("key_points_for_discussion", []),
                "figures_to_prepare": briefing.get("publication_strategy", {}).get("figures_to_prepare", [])
            },
            "gpto3_briefing": {
                "comprehensive_briefing": briefing,
                "rct_guardrails": briefing.get("rct_guardrails", {})
            }
        }
        
        logger.info(f"Integrated research design created for {research_area}")
        return integrated_design
    
    async def get_integrated_workflow_diagram(self) -> str:
        """
        Generate a comprehensive Markdown diagram showing the full integration
        
        Returns:
            Markdown string with integrated workflow diagram
        """
        diagram = """
```mermaid
graph TD
    subgraph "Universal Mind Research Design System"
        User[Researcher/User] --> GPTo3[GPTo3]
        GPTo3 --> |"Designs research<br>with quality controls"|URDI[Universal Research<br>Design Integrator]
        
        subgraph "Research Design Components"
            URDI --> RDA[Research Design Agent]
            URDI --> PO[PandaOmics InSilico]
            URDI --> GRG[Genomic RCT Guardrails]
            
            RDA --> |"JADAD<br>scoring"|JS[JADAD Score]
            RDA --> |"IF<br>analysis"|IF[Impact Factor]
            
            PO --> |"In Silico<br>RCT execution"|ISRCT[Virtual RCT Results]
            
            GRG --> FPL[Fuzzy Phi Logic]
            FPL --> |"Calculates"|RV[Research Vector]
            RV --> |"P × WSS"|SS[Scientific Significance]
            
            JS --> |"QS calculation"|QS[Quality Score]
            SS --> |"QS calculation"|QS
        end
        
        URDI --> |"Validated<br>research design"|DORA[DORA Manuscript<br>Service]
        DORA --> |"Publishable<br>manuscript"|Journal[High-Impact<br>Journals]
    end
    
    subgraph "Fuzzy Phi Logic Workflow"
        PP[Pisano Period<br>α, β, γ sequences] --> |"Count of<br>Platonic geometries"|P[P value]
        BioRel[Biological<br>Relationships] --> |"Process<br>importance"|WSS[Weighted Scientific<br>Significance]
        P --> |"P × WSS"|SS
        
        SS --> |"Category<br>determination"|HiWSS[High WSS 8-10<br>Creates cellular life]
        SS --> |"Category<br>determination"|MedWSS[Medium WSS 5-7<br>Supports cellular life]
        SS --> |"Category<br>determination"|LowWSS[Low WSS 1-4<br>Creates elemental life]
    end
    
    subgraph "Quality Verification"
        QS --> |"≥7 required for<br>Nature journals"|NatQ[Nature Quality<br>Threshold]
        QS --> |"≥5 required for<br>Frontiers journals"|FrontQ[Frontiers Quality<br>Threshold]
        JS --> |"≥4 required for<br>Nature journals"|NatJ[Nature JADAD<br>Threshold]
        JS --> |"≥3 required for<br>Frontiers journals"|FrontJ[Frontiers JADAD<br>Threshold]
    end
```
"""
        return diagram
    
    def get_rct_design_implementation_diagram(self) -> str:
        """
        Generate a Markdown diagram showing the RCT design implementation
        
        Returns:
            Markdown string with RCT design implementation diagram
        """
        diagram = """
```mermaid
sequenceDiagram
    participant User as Researcher/User
    participant URDI as Universal Research Design Integrator
    participant RDA as Research Design Agent
    participant GPTo3 as GPTo3 Agent
    participant PO as PandaOmics InSilico
    participant FPL as Fuzzy Phi Logic
    participant DORA as DORA Manuscript Service
    
    User->>URDI: Request publishable research design
    URDI->>RDA: Design study for Nature/Frontiers
    RDA->>RDA: Calculate JADAD requirements
    RDA->>RDA: Analyze target journal IF
    RDA->>URDI: Return initial design (JADAD ≥ 4)
    
    URDI->>PO: Execute In Silico RCT
    PO->>PO: Query genomic databases
    PO->>PO: Simulate RCT with virtual participants
    PO->>PO: Validate against JADAD criteria
    PO->>URDI: Return In Silico RCT results
    
    URDI->>FPL: Calculate Research Vector
    FPL->>FPL: Generate Pisano periods (α, β, γ)
    FPL->>FPL: Count geometric nestings
    FPL->>FPL: Calculate biological importance
    FPL->>FPL: Determine WSS category
    FPL->>URDI: Return Scientific Significance
    
    URDI->>URDI: Calculate Quality Score
    URDI->>URDI: Validate publishability
    URDI->>GPTo3: Create comprehensive briefing
    
    GPTo3->>GPTo3: Apply RCT guardrails
    GPTo3->>GPTo3: Format for Nature/Frontiers requirements
    GPTo3->>URDI: Return enhanced briefing
    
    URDI->>DORA: Submit validated design
    DORA->>DORA: Generate publication-ready manuscript
    DORA->>User: Deliver publishable research design
```
"""
        return diagram
    
    async def get_implementation_code_snippet(self) -> str:
        """
        Generate a Python code snippet showing how to use the integrator
        
        Returns:
            Python code snippet as string
        """
        code = """
# Example implementation of the Universal Research Design Integrator

import asyncio
from universal_research_design_integrator import UniversalResearchDesignIntegrator

async def design_publishable_research():
    # Initialize with API keys
    api_keys = {
        "openai": "your_openai_key", 
        "scite": "your_scite_key",
        "dora": "your_dora_key"
    }
    
    # Create integrator
    integrator = UniversalResearchDesignIntegrator(api_keys)
    
    # Define research parameters
    research_area = "genomics"
    genes = ["BRCA1", "TP53", "EGFR"]
    condition = "Breast Cancer"
    target_journals = ["Nature Methods", "Frontiers in Bioinformatics"]
    
    # Create integrated research design
    design = await integrator.create_integrated_research_design(
        research_area=research_area,
        genes=genes,
        condition=condition,
        target_journals=target_journals
    )
    
    # Check publishability
    if design["quality_metrics"]["nature_viable"]:
        print(f"Research design meets Nature publication standards!")
        print(f"JADAD Score: {design['quality_metrics']['jadad_score']}")
        print(f"WSS: {design['quality_metrics']['weighted_scientific_significance']}")
        print(f"Quality Score: {design['quality_metrics']['integrated_quality_score']}")
        
        # Print research vector components
        print("\nResearch Vector Components:")
        print(f"Platonic Geometries: {design['fuzzy_phi_assessment']['research_vector']['platonic_geometries']}")
        print(f"WSS Category: {design['fuzzy_phi_assessment']['research_vector']['wss_category']}")
        
        # Access the GPTo3 briefing
        briefing = design["gpto3_briefing"]["comprehensive_briefing"]
        print(f"\nGPTo3 Briefing generated for {len(briefing['study_context']['target_journals'])} target journals")
        
        # Generate manuscript with DORA (in a real implementation)
        print("\nReady to generate manuscript with DORA")
    else:
        print("Research design needs improvement to meet Nature standards")
        print(f"Current JADAD Score: {design['quality_metrics']['jadad_score']} (needs ≥4)")
        print(f"Current WSS: {design['quality_metrics']['weighted_scientific_significance']} (needs ≥8)")
        print(f"Current Quality Score: {design['quality_metrics']['integrated_quality_score']} (needs ≥7)")
        
        if design["quality_metrics"]["frontiers_viable"]:
            print("\nResearch design meets Frontiers publication standards")
            print("Consider targeting Frontiers journals instead, or improving design")
        else:
            print("\nResearch design needs improvement for any high-impact publication")
            
        # Get improvement suggestions
        print("\nImprovement suggestions:")
        for i, suggestion in enumerate(design.get("quality_metrics", {}).get("jadad_improvements", []), 1):
            print(f"{i}. {suggestion}")

# Run the example
if __name__ == "__main__":
    asyncio.run(design_publishable_research())
"""
        return code
    
    async def run_example(self) -> None:
        """Run a complete example of the research design process"""
        # Print header
        print("=" * 80)
        print("Universal Mind Research Design Integrator Example")
        print("=" * 80)
        
        # Define example parameters
        research_area = "genomics"
        genes = ["BRCA1", "TP53", "EGFR"]
        condition = "Breast Cancer"
        target_journals = ["Nature Methods", "Frontiers in Bioinformatics"]
        
        # Print parameters
        print(f"\nResearch Area: {research_area}")
        print(f"Target Genes: {', '.join(genes)}")
        print(f"Condition: {condition}")
        print(f"Target Journals: {', '.join(target_journals)}")
        
        # Generate workflow diagrams
        print("\nWorkflow Diagrams:")
        print("1. Agent Integration Diagram")
        print(generate_agent_integration_diagram())
        print("\n2. Research Workflow Diagram")
        print(generate_research_workflow_diagram())
        print("\n3. Integrated Workflow Diagram")
        print(await self.get_integrated_workflow_diagram())
        
        # Create integrated research design
        print("\nGenerating integrated research design...")
        design = await self.create_integrated_research_design(
            research_area=research_area,
            genes=genes,
            condition=condition,
            target_journals=target_journals
        )
        
        # Print quality metrics
        print("\nQuality Metrics:")
        print(f"JADAD Score: {design['quality_metrics']['jadad_score']}")
        print(f"Weighted Scientific Significance: {design['quality_metrics']['weighted_scientific_significance']}")
        print(f"Integrated Quality Score: {design['quality_metrics']['integrated_quality_score']}")
        print(f"Nature Viable: {design['quality_metrics']['nature_viable']}")
        print(f"Frontiers Viable: {design['quality_metrics']['frontiers_viable']}")
        
        # Print Fuzzy Phi Logic assessment
        print("\nFuzzy Phi Logic Assessment:")
        print(f"Pisano Period Modulus: {design['fuzzy_phi_assessment']['pisano_period']['modulus']}")
        print(f"Geometric Nestings: {design['fuzzy_phi_assessment']['geometric_nestings']}")
        print(f"WSS Category: {design['fuzzy_phi_assessment']['research_vector']['wss_category']}")
        print(f"WSS Description: {design['fuzzy_phi_assessment']['research_vector']['wss_description']}")
        
        # Print publication strategy
        print("\nPublication Strategy:")
        print(f"Viable Journals: {len(design['publication_strategy']['viable_journals'])}")
        if design['publication_strategy']['viable_journals']:
            top_journal = design['publication_strategy']['viable_journals'][0]
            print(f"Top Journal: {top_journal['name']} (IF: {top_journal['impact_factor']})")
        
        # Print conclusion
        print("\nConclusion:")
        if design["quality_metrics"]["nature_viable"]:
            print("Research design meets Nature publication standards!")
            print("Ready to generate manuscript with DORA")
        elif design["quality_metrics"]["frontiers_viable"]:
            print("Research design meets Frontiers publication standards")
            print("Consider improving design or proceeding with Frontiers submission")
        else:
            print("Research design needs improvement for high-impact publication")
            print("Follow JADAD and WSS improvement suggestions")
        
        print("\nExample completed successfully")
        print("=" * 80)

# -------------------------------------------------------------------------------
# Helper functions for integrations with other components
# -------------------------------------------------------------------------------

async def integrate_dora_with_research_design(
    api_keys: Dict[str, str],
    research_design: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Integrate DORA manuscript service with the research design
    
    Args:
        api_keys: Dictionary of API keys
        research_design: Integrated research design
        
    Returns:
        DORA manuscript generation results
    """
    from Integrated_Publication_Pipeline import DoraManuscriptService
    
    # Initialize DORA service
    if "dora" in api_keys:
        dora_url = "https://api.pharma.ai/dora/v1"
        dora_service = DoraManuscriptService(api_keys["dora"], dora_url)
        
        # Convert research design to format expected by DORA
        dora_data = {
            "title": f"Impact of {', '.join(research_design['research_metadata']['genes'][:2])} in {research_design['research_metadata']['condition']}",
            "authors": [
                {"name": "Universal Mind Research Team", "email": "research@universalmind.ai", "affiliation": "Universal Mind PBC"}
            ],
            "abstract": f"A genomic study investigating the role of {', '.join(research_design['research_metadata']['genes'])} in {research_design['research_metadata']['condition']} using an integrated research design approach.",
            "results": research_design["in_silico_validation"],
            "target_journals": research_design["research_metadata"]["target_journals"]
        }
        
        # Generate manuscript
        manuscript = await dora_service.generate_manuscript(
            research_data=dora_data,
            template="scientific",
            target_journal=research_design["research_metadata"]["target_journals"][0]
        )
        
        return manuscript
    else:
        return {"error": "DORA API key not provided"}

async def brief_gpto3_on_research_design(
    api_keys: Dict[str, str],
    research_design: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create a comprehensive briefing for GPTo3 based on the research design
    
    Args:
        api_keys: Dictionary of API keys
        research_design: Integrated research design
        
    Returns:
        GPTo3 briefing results
    """
    # Initialize the Universal Research Design Integrator
    integrator = UniversalResearchDesignIntegrator(api_keys)
    
    # Extract metadata
    research_area = research_design["research_metadata"]["area"]
    genes = research_design["research_metadata"]["genes"]
    condition = research_design["research_metadata"]["condition"]
    target_journals = research_design["research_metadata"]["target_journals"]
    
    # Generate briefing
    briefing = await integrator.guardrails.brief_gpto3(
        research_area,
        genes,
        condition,
        target_journals
    )
    
    return briefing

if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize with API keys
        api_keys = {
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "scite": os.environ.get("SCITE_API_KEY", ""),
            "dora": os.environ.get("DORA_API_KEY", "")
        }
        
        # Create integrator
        integrator = UniversalResearchDesignIntegrator(api_keys)
        
        # Run example
        await integrator.run_example()
        
        # Print code snippet
        print("\nImplementation Code Snippet:")
        print(await integrator.get_implementation_code_snippet())
    
    # Run example
    asyncio.run(main())
