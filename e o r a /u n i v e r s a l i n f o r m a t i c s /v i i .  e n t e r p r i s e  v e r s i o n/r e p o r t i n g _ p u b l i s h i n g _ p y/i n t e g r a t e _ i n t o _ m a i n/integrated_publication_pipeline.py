#!/usr/bin/env python3
# Integrated Publication Pipeline
# Part of Universal Informatics API - Module 4: reporting_publishing.py
# Author: Claude, based on Universal Informatics Team design
# Date: May 8, 2025

import os
import json
import time
import logging
import requests
import aiohttp
import asyncio
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid
import re
from enum import Enum

# For PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    print("Warning: ReportLab not available. PDF generation will be limited.")

# For bibliographic management
try:
    import pybtex
    from pybtex.database import BibliographyData, Entry
    HAS_PYBTEX = True
except ImportError:
    HAS_PYBTEX = False
    print("Warning: PyBTeX not available. Bibliography management will be limited.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("publication_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("publication_pipeline")

# Import Research Design Agents for JADAD scoring and RCT design
try:
    from research_design_agents import ResearchDesignAgent, PandaOmicsInSilicoAgent, GPTo3BriefingAgent
    _HAS_RESEARCH_AGENTS = True
except ImportError:
    _HAS_RESEARCH_AGENTS = False
    print("Warning: Research Design Agents not available. RCT design capabilities will be limited.")

class PublicationStatus(Enum):
    """Status of a publication in the pipeline"""
    DRAFT = "draft"
    JOURNAL_SUBMITTED = "journal_submitted"
    UNDER_REVIEW = "under_review"
    REVISION_REQUESTED = "revision_requested"
    ACCEPTED = "accepted"
    PUBLISHED = "published"
    REJECTED = "rejected"

class JournalTier(Enum):
    """Journal tiers based on impact factor and prestige"""
    TIER_1 = "tier_1"  # Nature, Science, Cell, etc.
    TIER_2 = "tier_2"  # Frontiers, PLOS, etc.
    TIER_3 = "tier_3"  # Field-specific journals
    TIER_4 = "tier_4"  # Open access, newer journals

@dataclass
class Citation:
    """A citation for a reference"""
    key: str
    authors: List[str]
    title: str
    year: int
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    publisher: Optional[str] = None
    type: str = "article"  # article, book, conference, preprint
    citation_count: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_bibtex(self) -> str:
        """Convert citation to BibTeX format"""
        if not HAS_PYBTEX:
            # Basic BibTeX generation without pybtex
            bibtex = f"@{self.type}{{{self.key},\n"
            bibtex += f"  author = {{{' and '.join(self.authors)}}},\n"
            bibtex += f"  title = {{{self.title}}},\n"
            bibtex += f"  year = {{{self.year}}},\n"
            
            if self.journal:
                bibtex += f"  journal = {{{self.journal}}},\n"
            if self.volume:
                bibtex += f"  volume = {{{self.volume}}},\n"
            if self.issue:
                bibtex += f"  number = {{{self.issue}}},\n"
            if self.pages:
                bibtex += f"  pages = {{{self.pages}}},\n"
            if self.doi:
                bibtex += f"  doi = {{{self.doi}}},\n"
            if self.url:
                bibtex += f"  url = {{{self.url}}},\n"
            if self.publisher:
                bibtex += f"  publisher = {{{self.publisher}}},\n"
                
            bibtex += "}\n"
            return bibtex
        else:
            # Use pybtex for proper BibTeX generation
            entry = Entry(self.type)
            
            # Add fields
            fields = {
                'title': self.title,
                'year': str(self.year)
            }
            
            if self.journal:
                fields['journal'] = self.journal
            if self.volume:
                fields['volume'] = self.volume
            if self.issue:
                fields['number'] = self.issue
            if self.pages:
                fields['pages'] = self.pages
            if self.doi:
                fields['doi'] = self.doi
            if self.url:
                fields['url'] = self.url
            if self.publisher:
                fields['publisher'] = self.publisher
                
            entry.fields = fields
            
            # Add persons
            persons = {'author': []}
            for author in self.authors:
                first_last = author.split(', ')
                if len(first_last) == 2:
                    last, first = first_last
                else:
                    parts = author.split(' ')
                    last = parts[-1]
                    first = ' '.join(parts[:-1])
                
                persons['author'].append({'first': first, 'last': last})
            
            entry.persons = persons
            
            # Create bibliography data
            bib_data = BibliographyData({self.key: entry})
            
            # Format as BibTeX
            return bib_data.to_string('bibtex')

@dataclass
class Figure:
    """A figure for a publication"""
    id: str
    title: str
    caption: str
    file_path: str
    format: str = "png"  # png, jpg, svg, pdf
    width: Optional[float] = None
    height: Optional[float] = None
    dpi: int = 300
    data: Optional[Any] = None
    source_code: Optional[str] = None
    is_generated: bool = False
    
    def __post_init__(self):
        """Validate figure data"""
        if not os.path.exists(self.file_path) and not self.is_generated:
            logger.warning(f"Figure file not found: {self.file_path}")

@dataclass
class Table:
    """A table for a publication"""
    id: str
    title: str
    caption: str
    data: Union[str, pd.DataFrame]
    format: str = "csv"  # csv, tsv, dataframe
    
    def to_markdown(self) -> str:
        """Convert table to markdown format"""
        if isinstance(self.data, str):
            if self.format == "csv":
                df = pd.read_csv(self.data)
            elif self.format == "tsv":
                df = pd.read_csv(self.data, sep='\t')
            else:
                raise ValueError(f"Unsupported table format: {self.format}")
        else:
            df = self.data
            
        return df.to_markdown()

@dataclass
class Author:
    """An author of a publication"""
    name: str
    email: str
    affiliation: str
    orcid: Optional[str] = None
    is_corresponding: bool = False
    contributions: List[str] = field(default_factory=list)
    bio: Optional[str] = None

@dataclass
class ReviewComment:
    """A review comment"""
    id: str
    reviewer_id: str
    timestamp: str
    text: str
    section: Optional[str] = None
    response: Optional[str] = None
    status: str = "open"  # open, addressed, disputed

@dataclass
class PublicationMetadata:
    """Metadata for a publication"""
    title: str
    authors: List[Author]
    abstract: str
    keywords: List[str]
    journal_targets: List[str]
    preprint_targets: List[str]
    acknowledgments: Optional[str] = None
    funding_sources: List[str] = field(default_factory=list)
    conflicts_of_interest: List[str] = field(default_factory=list)
    ethical_statements: List[str] = field(default_factory=list)
    data_availability: Optional[str] = None
    code_availability: Optional[str] = None

@dataclass
class Publication:
    """A scientific publication"""
    id: str
    metadata: PublicationMetadata
    sections: Dict[str, str]
    figures: List[Figure] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    citations: List[Citation] = field(default_factory=list)
    supplementary_materials: List[Dict[str, Any]] = field(default_factory=list)
    status: PublicationStatus = PublicationStatus.DRAFT
    submission_history: List[Dict[str, Any]] = field(default_factory=list)
    review_comments: List[ReviewComment] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert publication to dictionary for serialization"""
        return {
            "id": self.id,
            "metadata": {
                "title": self.metadata.title,
                "authors": [
                    {
                        "name": author.name,
                        "email": author.email,
                        "affiliation": author.affiliation,
                        "orcid": author.orcid,
                        "is_corresponding": author.is_corresponding,
                        "contributions": author.contributions,
                        "bio": author.bio
                    }
                    for author in self.metadata.authors
                ],
                "abstract": self.metadata.abstract,
                "keywords": self.metadata.keywords,
                "journal_targets": self.metadata.journal_targets,
                "preprint_targets": self.metadata.preprint_targets,
                "acknowledgments": self.metadata.acknowledgments,
                "funding_sources": self.metadata.funding_sources,
                "conflicts_of_interest": self.metadata.conflicts_of_interest,
                "ethical_statements": self.metadata.ethical_statements,
                "data_availability": self.metadata.data_availability,
                "code_availability": self.metadata.code_availability
            },
            "sections": self.sections,
            "figures": [
                {
                    "id": fig.id,
                    "title": fig.title,
                    "caption": fig.caption,
                    "file_path": fig.file_path,
                    "format": fig.format,
                    "width": fig.width,
                    "height": fig.height,
                    "dpi": fig.dpi,
                    "is_generated": fig.is_generated
                }
                for fig in self.figures
            ],
            "tables": [
                {
                    "id": table.id,
                    "title": table.title,
                    "caption": table.caption,
                    "format": table.format,
                    "data": table.data if isinstance(table.data, str) else table.data.to_csv()
                }
                for table in self.tables
            ],
            "citations": [
                {
                    "key": citation.key,
                    "authors": citation.authors,
                    "title": citation.title,
                    "year": citation.year,
                    "journal": citation.journal,
                    "volume": citation.volume,
                    "issue": citation.issue,
                    "pages": citation.pages,
                    "doi": citation.doi,
                    "url": citation.url,
                    "publisher": citation.publisher,
                    "type": citation.type,
                    "citation_count": citation.citation_count,
                    "metrics": citation.metrics
                }
                for citation in self.citations
            ],
            "supplementary_materials": self.supplementary_materials,
            "status": self.status.value,
            "submission_history": self.submission_history,
            "review_comments": [
                {
                    "id": comment.id,
                    "reviewer_id": comment.reviewer_id,
                    "timestamp": comment.timestamp,
                    "text": comment.text,
                    "section": comment.section,
                    "response": comment.response,
                    "status": comment.status
                }
                for comment in self.review_comments
            ],
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Publication':
        """Create publication from dictionary"""
        # Create metadata
        metadata = PublicationMetadata(
            title=data["metadata"]["title"],
            authors=[
                Author(
                    name=author["name"],
                    email=author["email"],
                    affiliation=author["affiliation"],
                    orcid=author.get("orcid"),
                    is_corresponding=author.get("is_corresponding", False),
                    contributions=author.get("contributions", []),
                    bio=author.get("bio")
                )
                for author in data["metadata"]["authors"]
            ],
            abstract=data["metadata"]["abstract"],
            keywords=data["metadata"]["keywords"],
            journal_targets=data["metadata"]["journal_targets"],
            preprint_targets=data["metadata"]["preprint_targets"],
            acknowledgments=data["metadata"].get("acknowledgments"),
            funding_sources=data["metadata"].get("funding_sources", []),
            conflicts_of_interest=data["metadata"].get("conflicts_of_interest", []),
            ethical_statements=data["metadata"].get("ethical_statements", []),
            data_availability=data["metadata"].get("data_availability"),
            code_availability=data["metadata"].get("code_availability")
        )
        
        # Create figures
        figures = [
            Figure(
                id=fig["id"],
                title=fig["title"],
                caption=fig["caption"],
                file_path=fig["file_path"],
                format=fig.get("format", "png"),
                width=fig.get("width"),
                height=fig.get("height"),
                dpi=fig.get("dpi", 300),
                is_generated=fig.get("is_generated", False)
            )
            for fig in data.get("figures", [])
        ]
        
        # Create tables
        tables = []
        for table_data in data.get("tables", []):
            table_format = table_data.get("format", "csv")
            if table_format == "csv" or table_format == "tsv":
                table_content = table_data["data"]
            else:
                # Try to convert string to DataFrame
                sep = ',' if table_format == "csv" else '\t'
                try:
                    table_content = pd.read_csv(table_data["data"], sep=sep)
                except:
                    table_content = table_data["data"]
            
            tables.append(Table(
                id=table_data["id"],
                title=table_data["title"],
                caption=table_data["caption"],
                data=table_content,
                format=table_format
            ))
        
        # Create citations
        citations = [
            Citation(
                key=citation["key"],
                authors=citation["authors"],
                title=citation["title"],
                year=citation["year"],
                journal=citation.get("journal"),
                volume=citation.get("volume"),
                issue=citation.get("issue"),
                pages=citation.get("pages"),
                doi=citation.get("doi"),
                url=citation.get("url"),
                publisher=citation.get("publisher"),
                type=citation.get("type", "article"),
                citation_count=citation.get("citation_count"),
                metrics=citation.get("metrics", {})
            )
            for citation in data.get("citations", [])
        ]
        
        # Create review comments
        review_comments = [
            ReviewComment(
                id=comment["id"],
                reviewer_id=comment["reviewer_id"],
                timestamp=comment["timestamp"],
                text=comment["text"],
                section=comment.get("section"),
                response=comment.get("response"),
                status=comment.get("status", "open")
            )
            for comment in data.get("review_comments", [])
        ]
        
        # Create publication
        return cls(
            id=data["id"],
            metadata=metadata,
            sections=data["sections"],
            figures=figures,
            tables=tables,
            citations=citations,
            supplementary_materials=data.get("supplementary_materials", []),
            status=PublicationStatus(data.get("status", "draft")),
            submission_history=data.get("submission_history", []),
            review_comments=review_comments,
            last_updated=data.get("last_updated", datetime.now().isoformat())
        )
    
    def to_markdown(self) -> str:
        """Convert publication to markdown format"""
        md = f"# {self.metadata.title}\n\n"
        
        # Authors
        md += "## Authors\n\n"
        for author in self.metadata.authors:
            md += f"* {author.name} - {author.affiliation}"
            if author.orcid:
                md += f" - ORCID: {author.orcid}"
            if author.is_corresponding:
                md += f" (Corresponding author: {author.email})"
            md += "\n"
        md += "\n"
        
        # Abstract
        md += "## Abstract\n\n"
        md += f"{self.metadata.abstract}\n\n"
        
        # Keywords
        md += "## Keywords\n\n"
        md += ", ".join(self.metadata.keywords) + "\n\n"
        
        # Main sections
        for section_name, section_content in self.sections.items():
            md += f"## {section_name}\n\n"
            md += f"{section_content}\n\n"
        
        # Figures
        if self.figures:
            md += "## Figures\n\n"
            for fig in self.figures:
                md += f"**Figure {fig.id}: {fig.title}**\n\n"
                md += f"![{fig.title}]({fig.file_path})\n\n"
                md += f"*{fig.caption}*\n\n"
        
        # Tables
        if self.tables:
            md += "## Tables\n\n"
            for table in self.tables:
                md += f"**Table {table.id}: {table.title}**\n\n"
                md += table.to_markdown() + "\n\n"
                md += f"*{table.caption}*\n\n"
        
        # References
        if self.citations:
            md += "## References\n\n"
            for i, citation in enumerate(self.citations, 1):
                authors_str = ", ".join(citation.authors)
                journal_info = ""
                if citation.journal:
                    journal_info = f" {citation.journal}"
                    if citation.volume:
                        journal_info += f", {citation.volume}"
                        if citation.issue:
                            journal_info += f"({citation.issue})"
                    if citation.pages:
                        journal_info += f": {citation.pages}"
                
                md += f"{i}. {authors_str}. ({citation.year}). {citation.title}.{journal_info}"
                if citation.doi:
                    md += f" DOI: {citation.doi}"
                md += "\n\n"
        
        # Acknowledgments
        if self.metadata.acknowledgments:
            md += "## Acknowledgments\n\n"
            md += f"{self.metadata.acknowledgments}\n\n"
        
        # Funding sources
        if self.metadata.funding_sources:
            md += "## Funding\n\n"
            for source in self.metadata.funding_sources:
                md += f"* {source}\n"
            md += "\n"
        
        # Data and code availability
        if self.metadata.data_availability or self.metadata.code_availability:
            md += "## Data and Code Availability\n\n"
            if self.metadata.data_availability:
                md += f"**Data availability**: {self.metadata.data_availability}\n\n"
            if self.metadata.code_availability:
                md += f"**Code availability**: {self.metadata.code_availability}\n\n"
        
        return md
    
    def to_bibtex(self) -> str:
        """Generate BibTeX file for all citations"""
        bibtex = ""
        for citation in self.citations:
            bibtex += citation.to_bibtex() + "\n"
        return bibtex
    
    def generate_pdf(self, output_path: str) -> bool:
        """Generate PDF version of the publication
        
        Args:
            output_path: Path to save the PDF
            
        Returns:
            Success status
        """
        if not HAS_REPORTLAB:
            logger.error("ReportLab is required for PDF generation")
            return False
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Add custom styles
            styles.add(ParagraphStyle(
                name='Title',
                parent=styles['Title'],
                fontSize=16,
                spaceAfter=12
            ))
            
            styles.add(ParagraphStyle(
                name='Abstract',
                parent=styles['Normal'],
                fontSize=10,
                leftIndent=36,
                rightIndent=36,
                spaceAfter=12
            ))
            
            # Build content
            content = []
            
            # Title
            content.append(Paragraph(self.metadata.title, styles['Title']))
            content.append(Spacer(1, 12))
            
            # Authors
            authors_text = ""
            for i, author in enumerate(self.metadata.authors):
                if i > 0:
                    authors_text += ", "
                authors_text += author.name
                authors_text += "<super>%d</super>" % (i + 1)
            
            content.append(Paragraph(authors_text, styles['Normal']))
            content.append(Spacer(1, 12))
            
            # Affiliations
            for i, author in enumerate(self.metadata.authors):
                content.append(Paragraph(
                    "<super>%d</super> %s" % (i + 1, author.affiliation),
                    styles['Normal']
                ))
            
            content.append(Spacer(1, 12))
            
            # Abstract
            content.append(Paragraph("Abstract", styles['Heading2']))
            content.append(Paragraph(self.metadata.abstract, styles['Abstract']))
            content.append(Spacer(1, 12))
            
            # Keywords
            content.append(Paragraph(
                "<b>Keywords:</b> " + ", ".join(self.metadata.keywords),
                styles['Normal']
            ))
            content.append(Spacer(1, 24))
            
            # Main sections
            for section_name, section_content in self.sections.items():
                content.append(Paragraph(section_name, styles['Heading1']))
                
                # Split paragraphs
                paragraphs = section_content.split("\n\n")
                for paragraph in paragraphs:
                    content.append(Paragraph(paragraph, styles['Normal']))
                    content.append(Spacer(1, 12))
            
            # Figures
            for fig in self.figures:
                if os.path.exists(fig.file_path):
                    content.append(Paragraph(f"Figure {fig.id}: {fig.title}", styles['Heading3']))
                    content.append(Image(fig.file_path, width=450, height=300))
                    content.append(Paragraph(fig.caption, styles['Caption']))
                    content.append(Spacer(1, 12))
            
            # Tables
            for table in self.tables:
                content.append(Paragraph(f"Table {table.id}: {table.title}", styles['Heading3']))
                
                # Convert table data to ReportLab table
                if isinstance(table.data, pd.DataFrame):
                    df = table.data
                else:
                    if table.format == "csv":
                        df = pd.read_csv(table.data)
                    elif table.format == "tsv":
                        df = pd.read_csv(table.data, sep='\t')
                    else:
                        df = pd.read_csv(table.data)
                
                # Create table data
                table_data = [df.columns.tolist()]
                for _, row in df.iterrows():
                    table_data.append(row.tolist())
                
                # Create table
                report_table = Table(table_data)
                content.append(report_table)
                content.append(Paragraph(table.caption, styles['Caption']))
                content.append(Spacer(1, 12))
            
            # References
            if self.citations:
                content.append(Paragraph("References", styles['Heading1']))
                
                for i, citation in enumerate(self.citations, 1):
                    authors_str = ", ".join(citation.authors)
                    journal_info = ""
                    if citation.journal:
                        journal_info = f" {citation.journal}"
                        if citation.volume:
                            journal_info += f", {citation.volume}"
                            if citation.issue:
                                journal_info += f"({citation.issue})"
                        if citation.pages:
                            journal_info += f": {citation.pages}"
                    
                    ref_text = f"{i}. {authors_str}. ({citation.year}). {citation.title}.{journal_info}"
                    if citation.doi:
                        ref_text += f" DOI: {citation.doi}"
                    
                    content.append(Paragraph(ref_text, styles['Normal']))
                    content.append(Spacer(1, 6))
            
            # Build PDF
            doc.build(content)
            
            logger.info(f"PDF generated successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return False

class JournalFormatAdapter:
    """Adapter for formatting publications for specific journals"""
    
    def __init__(self, journal_name: str, template_path: Optional[str] = None):
        """Initialize journal adapter
        
        Args:
            journal_name: Name of the journal
            template_path: Path to journal template files (optional)
        """
        self.journal_name = journal_name
        self.template_path = template_path
        
        # Load journal specifications
        self.specs = self._load_journal_specs()
        
    def _load_journal_specs(self) -> Dict[str, Any]:
        """Load journal specifications from templates
        
        Returns:
            Journal specifications
        """
        # Default specifications
        default_specs = {
            "word_limit": 8000,
            "abstract_limit": 250,
            "max_figures": 8,
            "max_tables": 8,
            "citation_style": "apa",
            "required_sections": [
                "Introduction", "Methods", "Results", "Discussion"
            ],
            "formatting": {
                "title_case": "title",
                "headings_style": "sentence",
                "figure_placement": "end",
                "table_placement": "end"
            }
        }
        
        # Try to load journal-specific specifications
        if self.template_path:
            spec_path = os.path.join(self.template_path, f"{self.journal_name.lower()}_specs.json")
            if os.path.exists(spec_path):
                try:
                    with open(spec_path, 'r') as f:
                        journal_specs = json.load(f)
                    
                    # Merge with defaults
                    for key, value in journal_specs.items():
                        if isinstance(value, dict) and key in default_specs and isinstance(default_specs[key], dict):
                            # Merge nested dictionaries
                            default_specs[key].update(value)
                        else:
                            # Replace or add value
                            default_specs[key] = value
                    
                    logger.info(f"Loaded journal specifications for {self.journal_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading journal specifications: {e}")
        
        # Special case handling for known journals
        if self.journal_name.lower() == "nature":
            default_specs["word_limit"] = 4000
            default_specs["abstract_limit"] = 150
            default_specs["max_figures"] = 4
            default_specs["required_sections"] = [
                "Introduction", "Results", "Discussion", "Methods"
            ]
        elif self.journal_name.lower().startswith("frontiers"):
            default_specs["word_limit"] = 12000
            default_specs["abstract_limit"] = 350
            default_specs["max_figures"] = 15
            default_specs["required_sections"] = [
                "Introduction", "Materials and Methods", "Results", "Discussion"
            ]
        
        return default_specs
    
    def format_publication(self, publication: Publication) -> Publication:
        """Format publication according to journal specifications
        
        Args:
            publication: Publication to format
            
        Returns:
            Formatted publication
        """
        # Create a copy of the publication to avoid modifying the original
        formatted_pub = Publication.from_dict(publication.to_dict())
        
        # Format title case if needed
        if self.specs["formatting"]["title_case"] == "title":
            formatted_pub.metadata.title = self._to_title_case(formatted_pub.metadata.title)
        elif self.specs["formatting"]["title_case"] == "sentence":
            formatted_pub.metadata.title = self._to_sentence_case(formatted_pub.metadata.title)
        
        # Format section headings
        formatted_sections = {}
        for section_name, section_content in formatted_pub.sections.items():
            if self.specs["formatting"]["headings_style"] == "title":
                formatted_section_name = self._to_title_case(section_name)
            elif self.specs["formatting"]["headings_style"] == "sentence":
                formatted_section_name = self._to_sentence_case(section_name)
            elif self.specs["formatting"]["headings_style"] == "uppercase":
                formatted_section_name = section_name.upper()
            else:
                formatted_section_name = section_name
            
            formatted_sections[formatted_section_name] = section_content
        
        formatted_pub.sections = formatted_sections
        
        # Check for required sections
        for required_section in self.specs["required_sections"]:
            found = False
            for section_name in formatted_pub.sections.keys():
                if section_name.lower() == required_section.lower():
                    found = True
                    break
            
            if not found:
                logger.warning(f"Required section '{required_section}' not found in publication")
        
        # Truncate abstract if needed
        if len(formatted_pub.metadata.abstract.split()) > self.specs["abstract_limit"]:
            logger.warning(f"Abstract exceeds limit of {self.specs['abstract_limit']} words")
            
            # Simple truncation (could be improved)
            words = formatted_pub.metadata.abstract.split()
            formatted_pub.metadata.abstract = " ".join(words[:self.specs["abstract_limit"]])
        
        # Check word count
        total_words = sum(len(content.split()) for content in formatted_pub.sections.values())
        if total_words > self.specs["word_limit"]:
            logger.warning(f"Publication exceeds word limit of {self.specs['word_limit']} words")
        
        # Check figure and table limits
        if len(formatted_pub.figures) > self.specs["max_figures"]:
            logger.warning(f"Publication exceeds figure limit of {self.specs['max_figures']}")
        
        if len(formatted_pub.tables) > self.specs["max_tables"]:
            logger.warning(f"Publication exceeds table limit of {self.specs['max_tables']}")
        
        # Format citations according to journal style
        # This would be a more complex implementation in a real system
        
        return formatted_pub
    
    def _to_title_case(self, text: str) -> str:
        """Convert text to title case
        
        Args:
            text: Input text
            
        Returns:
            Title-cased text
        """
        # Skip small words
        small_words = {'a', 'an', 'and', 'as', 'at', 'but', 'by', 'en', 'for',
                      'if', 'in', 'of', 'on', 'or', 'the', 'to', 'via', 'vs'}
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            # Always capitalize first and last word
            if i == 0 or i == len(words) - 1:
                result.append(word.capitalize())
            # Don't capitalize small words
            elif word.lower() in small_words:
                result.append(word.lower())
            # Capitalize other words
            else:
                result.append(word.capitalize())
        
        return " ".join(result)
    
    def _to_sentence_case(self, text: str) -> str:
        """Convert text to sentence case
        
        Args:
            text: Input text
            
        Returns:
            Sentence-cased text
        """
        if not text:
            return text
        
        # Capitalize first letter, lowercase the rest
        return text[0].upper() + text[1:].lower()

class PublicationAPI:
    """API client for interacting with journal platforms"""
    
    def __init__(self, api_keys: Dict[str, str], base_urls: Dict[str, str]):
        """Initialize Publication API client
        
        Args:
            api_keys: Dictionary of API keys for different platforms
            base_urls: Dictionary of base URLs for different platforms
        """
        self.api_keys = api_keys
        self.base_urls = base_urls
        self.session = requests.Session()
        
        # Set timeout for requests
        self.timeout = 30
    
    async def submit_to_journal(self, 
                              publication: Publication, 
                              journal: str,
                              cover_letter: str) -> Dict[str, Any]:
        """Submit publication to journal
        
        Args:
            publication: Publication to submit
            journal: Journal name
            cover_letter: Cover letter text
            
        Returns:
            Response data
        """
        if journal.lower() not in self.api_keys or journal.lower() not in self.base_urls:
            raise ValueError(f"Unknown journal: {journal}")
        
        # Get API key and base URL
        api_key = self.api_keys[journal.lower()]
        base_url = self.base_urls[journal.lower()]
        
        # Format publication for journal
        adapter = JournalFormatAdapter(journal)
        formatted_pub = adapter.format_publication(publication)
        
        # Create submission payload
        payload = self._create_journal_payload(formatted_pub, journal, cover_letter)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/submissions",
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status in [200, 201, 202]:
                    data = await response.json()
                    
                    # Update publication status and history
                    publication.status = PublicationStatus.JOURNAL_SUBMITTED
                    publication.submission_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "platform": journal,
                        "status": "submitted",
                        "submission_id": data.get("submission_id", "unknown")
                    })
                    
                    logger.info(f"Successfully submitted to {journal}: {data.get('submission_id', 'unknown')}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error submitting to {journal}: {response.status} - {error_text}")
                    raise RuntimeError(f"Submission failed: {response.status} - {error_text}")
    
    async def check_submission_status(self, 
                                    publication_id: str, 
                                    submission_id: str,
                                    platform: str) -> Dict[str, Any]:
        """Check status of a submission
        
        Args:
            publication_id: Publication ID
            submission_id: Submission ID
            platform: Platform or journal name
            
        Returns:
            Status data
        """
        if platform.lower() not in self.api_keys or platform.lower() not in self.base_urls:
            raise ValueError(f"Unknown platform: {platform}")
        
        # Get API key and base URL
        api_key = self.api_keys[platform.lower()]
        base_url = self.base_urls[platform.lower()]
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/api/submissions/{submission_id}",
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Retrieved status from {platform}: {submission_id}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error checking status on {platform}: {response.status} - {error_text}")
                    raise RuntimeError(f"Status check failed: {response.status} - {error_text}")
    
    async def get_reviewer_comments(self, 
                                  submission_id: str,
                                  platform: str) -> List[Dict[str, Any]]:
        """Get reviewer comments for a submission
        
        Args:
            submission_id: Submission ID
            platform: Platform or journal name
            
        Returns:
            List of reviewer comments
        """
        if platform.lower() not in self.api_keys or platform.lower() not in self.base_urls:
            raise ValueError(f"Unknown platform: {platform}")
        
        # Get API key and base URL
        api_key = self.api_keys[platform.lower()]
        base_url = self.base_urls[platform.lower()]
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/api/submissions/{submission_id}/reviews",
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Retrieved {len(data.get('reviews', []))} reviewer comments from {platform}")
                    return data.get("reviews", [])
                else:
                    error_text = await response.text()
                    logger.error(f"Error getting reviews from {platform}: {response.status} - {error_text}")
                    raise RuntimeError(f"Review retrieval failed: {response.status} - {error_text}")
    
    async def submit_revision(self, 
                            publication: Publication,
                            submission_id: str,
                            platform: str,
                            response_to_reviewers: str) -> Dict[str, Any]:
        """Submit a revision to a previous submission
        
        Args:
            publication: Revised publication
            submission_id: Original submission ID
            platform: Platform or journal name
            response_to_reviewers: Response letter to reviewers
            
        Returns:
            Response data
        """
        if platform.lower() not in self.api_keys or platform.lower() not in self.base_urls:
            raise ValueError(f"Unknown platform: {platform}")
        
        # Get API key and base URL
        api_key = self.api_keys[platform.lower()]
        base_url = self.base_urls[platform.lower()]
        
        # Create revision payload
        payload = self._create_revision_payload(publication, submission_id, response_to_reviewers)
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.put(
                f"{base_url}/api/submissions/{submission_id}",
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status in [200, 202]:
                    data = await response.json()
                    
                    # Update publication history
                    publication.submission_history.append({
                        "timestamp": datetime.now().isoformat(),
                        "platform": platform,
                        "status": "revision_submitted",
                        "submission_id": submission_id
                    })
                    
                    logger.info(f"Successfully submitted revision to {platform}: {submission_id}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error submitting revision to {platform}: {response.status} - {error_text}")
                    raise RuntimeError(f"Revision submission failed: {response.status} - {error_text}")
    

    
    def _create_journal_payload(self, 
                              publication: Publication, 
                              journal: str,
                              cover_letter: str) -> Dict[str, Any]:
        """Create payload for journal submission
        
        Args:
            publication: Publication to submit
            journal: Journal name
            cover_letter: Cover letter text
            
        Returns:
            Submission payload
        """
        # Convert corresponding author flag to object
        corresponding_authors = []
        for author in publication.metadata.authors:
            if author.is_corresponding:
                corresponding_authors.append({
                    "name": author.name,
                    "email": author.email
                })
        
        # Create base payload
        payload = {
            "title": publication.metadata.title,
            "abstract": publication.metadata.abstract,
            "authors": [
                {
                    "given_name": author.name.split()[0],
                    "family_name": author.name.split()[-1],
                    "email": author.email,
                    "affiliation": author.affiliation,
                    "orcid_id": author.orcid,
                    "is_corresponding": author.is_corresponding
                }
                for author in publication.metadata.authors
            ],
            "corresponding_authors": corresponding_authors,
            "keywords": publication.metadata.keywords,
            "cover_letter": cover_letter,
            "manuscript_text": publication.to_markdown(),
            "funding_statement": "\n".join(publication.metadata.funding_sources),
            "competing_interests": "\n".join(publication.metadata.conflicts_of_interest),
            "data_availability": publication.metadata.data_availability,
            "acknowledgments": publication.metadata.acknowledgments
        }
        
        # Journal-specific adjustments
        if journal.lower().startswith("nature"):
            payload["manuscript_type"] = "Article"
            payload["subject_areas"] = ["Bioinformatics", "Genomics"]
        elif journal.lower().startswith("frontiers"):
            payload["manuscript_type"] = "Original Research"
            payload["subject_areas"] = ["Computational Genomics", "Bioinformatics"]
            payload["research_topics"] = ["Genomics", "Machine Learning in Bioinformatics"]
        
        return payload
    
    def _create_revision_payload(self, 
                               publication: Publication,
                               submission_id: str,
                               response_to_reviewers: str) -> Dict[str, Any]:
        """Create payload for revision submission
        
        Args:
            publication: Revised publication
            submission_id: Original submission ID
            response_to_reviewers: Response letter to reviewers
            
        Returns:
            Revision payload
        """
        # Base payload similar to journal submission
        payload = self._create_journal_payload(publication, "generic", "")
        
        # Add revision-specific fields
        payload["submission_id"] = submission_id
        payload["response_to_reviewers"] = response_to_reviewers
        
        # Track changes - this would be more sophisticated in a real system
        payload["changes_summary"] = "Revised manuscript addressing reviewer comments"
        
        return payload

class DoraManuscriptService:
    """Service for generating manuscript drafts using DORA"""
    
    def __init__(self, api_key: str, base_url: str = "https://api.pharma.ai/dora/v1"):
        """Initialize DORA manuscript service
        
        Args:
            api_key: DORA API key
            base_url: Base URL for DORA API
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        
        # Set timeout for requests
        self.timeout = 60  # Longer timeout for manuscript generation
    
    async def generate_manuscript(self, research_data: Dict[str, Any], 
                                template: str = "scientific",
                                target_journal: Optional[str] = None) -> Dict[str, Any]:
        """Generate manuscript draft from research data
        
        Args:
            research_data: Research data structure
            template: Manuscript template (scientific, clinical, review)
            target_journal: Target journal for formatting (optional)
            
        Returns:
            Generated manuscript data
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Prepare payload
        payload = {
            "research_data": research_data,
            "template": template,
            "options": {
                "include_figures": True,
                "include_tables": True,
                "include_references": True,
                "language_style": "academic"
            }
        }
        
        # Add target journal if provided
        if target_journal:
            payload["options"]["target_journal"] = target_journal
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Generated manuscript draft with DORA")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error generating manuscript with DORA: {response.status} - {error_text}")
                    raise RuntimeError(f"Manuscript generation failed: {response.status} - {error_text}")
                    
    async def revise_manuscript(self, manuscript_id: str, 
                              review_comments: List[Dict[str, Any]],
                              revision_instructions: str) -> Dict[str, Any]:
        """Revise manuscript based on review comments
        
        Args:
            manuscript_id: ID of manuscript to revise
            review_comments: List of review comments
            revision_instructions: Specific instructions for revision
            
        Returns:
            Revised manuscript data
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Prepare payload
        payload = {
            "manuscript_id": manuscript_id,
            "review_comments": review_comments,
            "revision_instructions": revision_instructions,
            "options": {
                "include_response_letter": True,
                "track_changes": True
            }
        }
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/revise",
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Generated manuscript revision with DORA")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error revising manuscript with DORA: {response.status} - {error_text}")
                    raise RuntimeError(f"Manuscript revision failed: {response.status} - {error_text}")
                    
    async def generate_response_letter(self, manuscript_id: str,
                                    review_comments: List[Dict[str, Any]]) -> str:
        """Generate response letter to reviewers
        
        Args:
            manuscript_id: ID of manuscript
            review_comments: List of review comments with responses
            
        Returns:
            Formatted response letter
        """
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Prepare payload
        payload = {
            "manuscript_id": manuscript_id,
            "review_comments": review_comments,
            "options": {
                "letter_style": "professional",
                "include_summary": True
            }
        }
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/response_letter",
                json=payload,
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Generated response letter with DORA")
                    return data.get("response_letter", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Error generating response letter with DORA: {response.status} - {error_text}")
                    raise RuntimeError(f"Response letter generation failed: {response.status} - {error_text}")


class CitationService:
    """Service for managing citations and bibliographic data"""
    
    def __init__(self, api_keys: Dict[str, str]):
        """Initialize citation service
        
        Args:
            api_keys: Dictionary of API keys for citation services
        """
        self.api_keys = api_keys
        self.session = requests.Session()
        
        # Set timeout for requests
        self.timeout = 30
    
    async def query_scite(self, doi: str) -> Dict[str, Any]:
        """Query Scite.ai for citation context
        
        Args:
            doi: DOI to query
            
        Returns:
            Citation context data
        """
        if "scite" not in self.api_keys:
            raise ValueError("Scite API key not provided")
        
        # Prepare headers
        headers = {
            "x-api-key": self.api_keys['scite'],
            "Accept": "application/json"
        }
        
        # Scite API endpoint
        url = f"https://api.scite.ai/v1/citations/doi/{doi}"
        
        # Use aiohttp for async request
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                headers=headers,
                timeout=self.timeout
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Retrieved Scite metrics for DOI: {doi}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Error querying Scite: {response.status} - {error_text}")
                    return {"error": f"Scite query failed: {response.status}"}
    
    async def enrich_citation(self, citation: Citation) -> Citation:
        """Enrich citation with metrics from citation services
        
        Args:
            citation: Citation to enrich
            
        Returns:
            Enriched citation
        """
        # Skip if no DOI
        if not citation.doi:
            logger.warning(f"No DOI provided for citation: {citation.key}")
            return citation
        
        # Query Scite
        scite_result = await self.query_scite(citation.doi)
        
        # Update citation with Scite metrics
        if "error" not in scite_result:
            citation.metrics["supporting_citations"] = scite_result.get("supporting", 0)
            citation.metrics["mentioning_citations"] = scite_result.get("mentioning", 0)
            citation.metrics["contrasting_citations"] = scite_result.get("contrasting", 0)
            citation.metrics["trending_score"] = scite_result.get("trending_score", 0)
        
        logger.info(f"Enriched citation: {citation.key}")
        return citation
    
    async def enrich_all_citations(self, citations: List[Citation]) -> List[Citation]:
        """Enrich multiple citations with metrics
        
        Args:
            citations: List of citations to enrich
            
        Returns:
            List of enriched citations
        """
        tasks = [self.enrich_citation(citation) for citation in citations]
        return await asyncio.gather(*tasks)

# LangChain and LangGraph integration for publication agents
class PublicationAgentSystem:
    """Autonomous agent system for managing publication workflows using LangChain and LangGraph.
    
    This class enables:
    1. AI-driven communication with journals, reviewers, and co-authors
    2. Autonomous handling of submission processes
    3. Intelligent revision management based on reviewer feedback
    4. Strategic journal selection and formatting
    """
    
    def __init__(self, 
                 api_keys: Dict[str, str], 
                 model_name: str = "gpt-4",
                 temperature: float = 0.2):
        """Initialize the Publication Agent system
        
        Args:
            api_keys: Dictionary of API keys for various services
            model_name: LLM model to use for agents
            temperature: Temperature for LLM generation
        """
        self.api_keys = api_keys
        self.model_name = model_name
        self.temperature = temperature
        self.initialized = False
        self.agents = {}
        
        # Initialize components if LangChain is available
        if _HAS_LANGCHAIN:
            self._initialize_components()
        else:
            logger.warning("LangChain not available - agent capabilities will be limited")
    
    def _initialize_components(self):
        """Initialize LangChain components for agent system"""
        # Initialize LLM
        if "openai" in self.api_keys:
            self.llm = ChatOpenAI(
                openai_api_key=self.api_keys["openai"],
                model_name=self.model_name,
                temperature=self.temperature
            )
            
            # Initialize agent memory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create agent tools
            self.tools = self._create_tools()
            
            # Initialize agent system if LangGraph is available
            if _HAS_LANGGRAPH:
                self._initialize_graph()
            else:
                # Fallback to basic agent without graph
                self._initialize_basic_agent()
                
            self.initialized = True
                
        else:
            logger.warning("OpenAI API key not provided - agent capabilities will be limited")
            self.initialized = False
    
    def _create_tools(self) -> List[Tool]:
        """Create tools for the publication agent"""
        tools = [
            Tool(
                name="format_for_journal",
                func=self._tool_format_for_journal,
                description="Format a publication for a specific journal's requirements"
            ),
            Tool(
                name="check_submission_status",
                func=self._tool_check_submission_status,
                description="Check the status of a submitted publication"
            ),
            Tool(
                name="generate_response_letter",
                func=self._tool_generate_response_letter,
                description="Generate a response letter to reviewer comments"
            ),
            Tool(
                name="analyze_reviewer_feedback",
                func=self._tool_analyze_reviewer_feedback,
                description="Analyze reviewer comments to identify key issues to address"
            ),
            Tool(
                name="recommend_journal",
                func=self._tool_recommend_journal,
                description="Recommend target journals based on publication content and impact goals"
            )
        ]
        return tools
    
    def _initialize_graph(self):
        """Initialize LangGraph for multi-step agent workflows"""
        # Define node functions for the publication workflow graph
        def journal_selection_node(state):
            """Select appropriate journals for submission"""
            publication_data = state["publication_data"]
            recommended_journals = self._recommend_journals(publication_data)
            state["recommended_journals"] = recommended_journals
            return state
        
        def manuscript_formatting_node(state):
            """Format manuscript for selected journal"""
            publication_data = state["publication_data"]
            target_journal = state["selected_journal"]
            formatted_manuscript = self._format_for_journal(publication_data, target_journal)
            state["formatted_manuscript"] = formatted_manuscript
            return state
        
        def submission_node(state):
            """Submit manuscript to journal"""
            formatted_manuscript = state["formatted_manuscript"]
            target_journal = state["selected_journal"]
            cover_letter = state.get("cover_letter", "")
            submission_result = self._submit_to_journal(formatted_manuscript, target_journal, cover_letter)
            state["submission_result"] = submission_result
            return state
        
        def revision_node(state):
            """Handle reviewer feedback and revisions"""
            publication_data = state["publication_data"]
            reviewer_comments = state["reviewer_comments"]
            revision_strategy = self._analyze_reviewer_feedback(reviewer_comments)
            response_letter = self._generate_response_letter(publication_data, reviewer_comments)
            revised_manuscript = self._revise_manuscript(publication_data, revision_strategy)
            state["revised_manuscript"] = revised_manuscript
            state["response_letter"] = response_letter
            return state
        
        # Define the workflow graph
        workflow = StateGraph(state_type=Dict)
        
        # Add nodes
        workflow.add_node("journal_selection", journal_selection_node)
        workflow.add_node("manuscript_formatting", manuscript_formatting_node)
        workflow.add_node("submission", submission_node)
        workflow.add_node("revision", revision_node)
        
        # Add conditional routing
        workflow.add_conditional_edges(
            "journal_selection",
            lambda x: "manuscript_formatting" if x.get("selected_journal") else END
        )
        workflow.add_edge("manuscript_formatting", "submission")
        workflow.add_conditional_edges(
            "submission",
            lambda x: "revision" if x.get("reviewer_comments") else END
        )
        workflow.add_edge("revision", END)
        
        # Compile the graph
        self.workflow = workflow.compile()
        logger.info("Initialized LangGraph workflow for publication pipeline")
    
    def _initialize_basic_agent(self):
        """Initialize basic LangChain agent without graph capability"""
        # Create prompt template
        template = """You are an expert in scientific publishing helping researchers with their publications.

Current state of the publication process:
{current_state}

Chat history:
{chat_history}

User query: {user_input}

Think step-by-step about how to help the user with their publication needs."""
        
        prompt = PromptTemplate(
            input_variables=["current_state", "chat_history", "user_input"],
            template=template
        )
        
        # Create the LLM chain
        llm_chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            verbose=True,
            memory=self.memory
        )
        
        # Create the agent
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            tools=self.tools,
            verbose=True
        )
        
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            memory=self.memory
        )
        
        logger.info("Initialized basic LangChain agent for publication pipeline")
    
    async def execute_workflow(self, publication_data: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Execute the publication workflow
        
        Args:
            publication_data: Data for the publication
            task: Task to perform (e.g., "submit", "revise")
            
        Returns:
            Result of the workflow execution
        """
        if not self.initialized:
            logger.error("Agent system not initialized properly")
            return {"error": "Agent system not initialized"}
        
        # Prepare initial state
        initial_state = {
            "publication_data": publication_data,
            "task": task,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            if _HAS_LANGGRAPH:
                # Execute the workflow graph
                logger.info(f"Executing LangGraph workflow for task: {task}")
                result = await self.workflow.ainvoke(initial_state)
                return result
            else:
                # Use basic agent
                logger.info(f"Executing basic agent for task: {task}")
                result = await self.agent_executor.arun(
                    current_state=json.dumps(initial_state),
                    user_input=f"Help me {task} this publication"
                )
                return {"agent_response": result}
        except Exception as e:
            logger.error(f"Error executing publication workflow: {e}")
            return {"error": str(e)}
    
    # Tool implementation methods
    def _tool_format_for_journal(self, publication_id: str, journal: str) -> str:
        """Tool to format a publication for a specific journal"""
        # This would call the appropriate formatting service
        return f"Formatted publication {publication_id} for {journal}"
    
    def _tool_check_submission_status(self, publication_id: str, submission_id: str, journal: str) -> str:
        """Tool to check submission status"""
        # This would call the appropriate API to check status
        return f"Checking status of submission {submission_id} for publication {publication_id} at {journal}"
    
    def _tool_generate_response_letter(self, publication_id: str, reviewer_comments: List[Dict[str, Any]]) -> str:
        """Tool to generate a response letter to reviewers"""
        # This would use LLM or specialized service to generate responses
        return f"Generated response letter addressing {len(reviewer_comments)} reviewer comments"
    
    def _tool_analyze_reviewer_feedback(self, reviewer_comments: List[Dict[str, Any]]) -> str:
        """Tool to analyze reviewer feedback"""
        # This would use LLM to analyze and categorize feedback
        return f"Analyzed {len(reviewer_comments)} comments and identified key revision needs"
    
    def _tool_recommend_journal(self, publication_data: Dict[str, Any]) -> str:
        """Tool to recommend target journals"""
        # This would analyze content and recommend appropriate journals
        return "Recommended journals based on publication content and impact goals"
    
    # Core agent methods
    def _recommend_journals(self, publication_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Recommend journals based on publication content"""
        # In a real implementation, this would do sophisticated journal matching
        # based on content, citation patterns, impact factors, etc.
        return [
            {"name": "Nature Methods", "impact_factor": 28.5, "acceptance_rate": 0.08},
            {"name": "Genome Biology", "impact_factor": 13.2, "acceptance_rate": 0.15},
            {"name": "Bioinformatics", "impact_factor": 6.9, "acceptance_rate": 0.22}
        ]
    
    def _format_for_journal(self, publication_data: Dict[str, Any], journal: str) -> Dict[str, Any]:
        """Format publication for specific journal"""
        # This would apply journal-specific formatting
        formatted_data = publication_data.copy()
        formatted_data["formatted_for"] = journal
        return formatted_data
    
    def _submit_to_journal(self, publication_data: Dict[str, Any], journal: str, cover_letter: str) -> Dict[str, Any]:
        """Submit publication to journal"""
        # This would handle the actual submission process
        return {
            "submission_id": f"subm-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now().isoformat(),
            "status": "submitted",
            "journal": journal
        }
    
    def _analyze_reviewer_feedback(self, reviewer_comments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze reviewer feedback to create revision strategy"""
        # This would categorize feedback and prioritize revisions
        return {
            "major_concerns": ["methodology clarity", "statistical analysis"],
            "minor_concerns": ["formatting issues", "citation completeness"],
            "priority_actions": ["expand methods section", "add statistical validation"]
        }
    
    def _generate_response_letter(self, publication_data: Dict[str, Any], reviewer_comments: List[Dict[str, Any]]) -> str:
        """Generate response letter to reviewer comments"""
        # This would create a detailed response letter
        return f"Dear Editor and Reviewers,\n\nWe thank you for your thoughtful feedback on our manuscript...\n\n"
    
    def _revise_manuscript(self, publication_data: Dict[str, Any], revision_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Revise manuscript based on feedback and strategy"""
        # This would apply revisions to the manuscript
        revised_data = publication_data.copy()
        revised_data["revision_history"] = {
            "timestamp": datetime.now().isoformat(),
            "strategy_applied": revision_strategy,
            "revision_round": 1
        }
        return revised_data

class IntegratedPublicationPipeline:
    """Integrated pipeline for publication management"""
    
    def __init__(self, 
                api_keys: Dict[str, str], 
                base_urls: Dict[str, str],
                output_dir: str = "publications"):
        """Initialize publication pipeline
        
        Args:
            api_keys: Dictionary of API keys for various services
            base_urls: Dictionary of base URLs for various services
            output_dir: Directory for output files
        """
        self.publication_api = PublicationAPI(api_keys, base_urls)
        self.citation_service = CitationService(api_keys)
        
        # Initialize DORA service if key provided
        if "dora" in api_keys:
            dora_url = base_urls.get("dora", "https://api.pharma.ai/dora/v1")
            self.dora_service = DoraManuscriptService(api_keys["dora"], dora_url)
        else:
            self.dora_service = None
        
        # Initialize Publication Agent if LangChain is available
        if _HAS_LANGCHAIN and "openai" in api_keys:
            self.agent_system = PublicationAgentSystem(api_keys)
        else:
            self.agent_system = None
            
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Track publications
        self.publications: Dict[str, Publication] = {}
        
    async def execute_agent_workflow(self, publication_id: str, task: str, **kwargs) -> Dict[str, Any]:
        """Execute an autonomous agent workflow for a publication task
        
        Args:
            publication_id: ID of the publication to process
            task: Task to perform (e.g., "format", "submit", "revise")
            **kwargs: Additional task-specific parameters
            
        Returns:
            Results from the agent workflow
        """
        if not self.agent_system or not self.agent_system.initialized:
            logger.error("Agent system not available or not properly initialized")
            return {"error": "Agent system not available"}
        
        # Check if publication exists
        if publication_id not in self.publications:
            logger.error(f"Publication {publication_id} not found")
            return {"error": f"Publication {publication_id} not found"}
        
        publication = self.publications[publication_id]
        
        # Prepare publication data for agent
        publication_data = publication.to_dict()
        
        # Add any additional parameters
        task_data = {
            "publication_id": publication_id,
            "publication_data": publication_data,
            "task": task,
            **kwargs
        }
        
        # Execute the agent workflow
        try:
            logger.info(f"Executing agent workflow for publication {publication_id}: {task}")
            result = await self.agent_system.execute_workflow(publication_data, task)
            
            # Handle result based on task type
            if task == "format" and "formatted_manuscript" in result:
                # Update publication with formatted content
                formatted_data = result["formatted_manuscript"]
                target_journal = result.get("selected_journal")
                
                logger.info(f"Updating publication {publication_id} with formatted content for {target_journal}")
                
                # Create a new publication ID for the formatted version
                formatted_id = f"{publication_id}_formatted_{target_journal.lower().replace(' ', '_')}"
                
                # Create a copy with formatted content
                formatted_pub = Publication.from_dict(formatted_data)
                formatted_pub.id = formatted_id
                
                # Store the formatted publication
                self.publications[formatted_id] = formatted_pub
                self._save_publication(formatted_pub)
                
                result["formatted_publication_id"] = formatted_id
                
            elif task == "submit" and "submission_result" in result:
                # Update publication with submission results
                submission_result = result["submission_result"]
                journal = submission_result.get("journal")
                submission_id = submission_result.get("submission_id")
                
                logger.info(f"Updating publication {publication_id} with submission results for {journal}")
                
                # Update publication status
                publication.status = PublicationStatus.JOURNAL_SUBMITTED
                
                # Add to submission history
                publication.submission_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "journal": journal,
                    "submission_id": submission_id,
                    "status": "submitted"
                })
                
                # Save updated publication
                self._save_publication(publication)
                
            elif task == "revise" and "revised_manuscript" in result:
                # Update publication with revision
                revised_data = result["revised_manuscript"]
                response_letter = result.get("response_letter", "")
                
                logger.info(f"Updating publication {publication_id} with revision content")
                
                # Create a new publication ID for the revised version
                revision_id = f"{publication_id}_revision_{len(publication.submission_history)}"
                
                # Create a copy with revised content
                revised_pub = Publication.from_dict(revised_data)
                revised_pub.id = revision_id
                
                # Store the revised publication
                self.publications[revision_id] = revised_pub
                self._save_publication(revised_pub)
                
                result["revised_publication_id"] = revision_id
                
            # Add agent execution metadata
            result["_agent_meta"] = {
                "execution_time": datetime.now().isoformat(),
                "original_publication_id": publication_id,
                "task": task
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing agent workflow: {e}")
            return {"error": str(e)}
    
    async def create_publication_from_results(self, 
                                           title: str,
                                           authors: List[Dict[str, Any]],
                                           abstract: str,
                                           results_data: Dict[str, Any],
                                           template: Optional[str] = None) -> Publication:
        """Create publication from research results
        
        Args:
            title: Publication title
            authors: List of author dictionaries
            abstract: Abstract text
            results_data: Research results data
            template: Template name (optional)
            
        Returns:
            Generated publication
        """
        # Create author objects
        author_objects = []
        for author_data in authors:
            author_objects.append(Author(
                name=author_data["name"],
                email=author_data["email"],
                affiliation=author_data["affiliation"],
                orcid=author_data.get("orcid"),
                is_corresponding=author_data.get("is_corresponding", False),
                contributions=author_data.get("contributions", []),
                bio=author_data.get("bio")
            ))
        
        # Create metadata
        metadata = PublicationMetadata(
            title=title,
            authors=author_objects,
            abstract=abstract,
            keywords=results_data.get("keywords", []),
            journal_targets=results_data.get("journal_targets", ["Frontiers in Bioinformatics"]),
            preprint_targets=[],  # No preprint targets
            acknowledgments=results_data.get("acknowledgments"),
            funding_sources=results_data.get("funding_sources", []),
            conflicts_of_interest=results_data.get("conflicts_of_interest", []),
            ethical_statements=results_data.get("ethical_statements", []),
            data_availability=results_data.get("data_availability"),
            code_availability=results_data.get("code_availability")
        )
        
        # Initialize default content collections
        sections = {}
        figures = []
        tables = []
        citations = []
        
        # Try using DORA for manuscript generation if available
        if self.dora_service is not None:
            try:
                logger.info("Using DORA service for manuscript generation")
                
                # Prepare research data for DORA
                dora_data = {
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "results": results_data,
                    "target_journals": metadata.journal_targets
                }
                
                # Generate manuscript with DORA
                manuscript_data = await self.dora_service.generate_manuscript(
                    research_data=dora_data,
                    template="scientific",
                    target_journal=metadata.journal_targets[0] if metadata.journal_targets else None
                )
                
                # Extract sections from manuscript
                sections = manuscript_data.get("sections", {})
                logger.debug(f"Generated {len(sections)} sections with DORA")
                
                # Extract figures from manuscript
                for fig_data in manuscript_data.get("figures", []):
                    figures.append(Figure(
                        id=fig_data["id"],
                        title=fig_data["title"],
                        caption=fig_data["caption"],
                        file_path=fig_data["file_path"],
                        format=fig_data.get("format", "png")
                    ))
                logger.debug(f"Extracted {len(figures)} figures from DORA manuscript")
                
                # Extract tables from manuscript
                for table_data in manuscript_data.get("tables", []):
                    table_obj = Table(
                        id=table_data["id"],
                        title=table_data["title"],
                        caption=table_data["caption"],
                        data=table_data["data"],
                        format=table_data.get("format", "csv")
                    )
                    tables.append(table_obj)
                logger.debug(f"Extracted {len(tables)} tables from DORA manuscript")
                
                # Extract citations from manuscript
                for citation_data in manuscript_data.get("citations", []):
                    citation = Citation(
                        key=citation_data["key"],
                        authors=citation_data["authors"],
                        title=citation_data["title"],
                        year=citation_data["year"],
                        journal=citation_data.get("journal"),
                        doi=citation_data.get("doi"),
                        url=citation_data.get("url")
                    )
                    citations.append(citation)
                logger.debug(f"Extracted {len(citations)} citations from DORA manuscript")
                
                logger.info("Successfully generated manuscript draft with DORA")
                
            except Exception as e:
                logger.error(f"Error generating manuscript with DORA: {e}")
                logger.info("Falling back to manual manuscript generation")
                
                # Fall back to manual generation if DORA fails
                sections = {}
                figures = []
                tables = []
                citations = []
        else:
            logger.info("DORA service not available, using manual manuscript generation")
        
        # If we don't have sections from DORA (either because it failed or isn't available),
        # generate them manually
        if not sections:
            logger.info("Generating manuscript sections from research results")
            sections = await self._generate_sections_from_results(results_data, template)
        
        # If we don't have figures from DORA, extract them from results
        if not figures:
            logger.info("Extracting figures from research results")
            for fig_data in results_data.get("figures", []):
                figures.append(Figure(
                    id=fig_data["id"],
                    title=fig_data["title"],
                    caption=fig_data["caption"],
                    file_path=fig_data["file_path"],
                    format=fig_data.get("format", "png")
                ))
        
        # If we don't have tables from DORA, extract them from results
        if not tables:
            logger.info("Extracting tables from research results")
            for table_data in results_data.get("tables", []):
                table_obj = Table(
                    id=table_data["id"],
                    title=table_data["title"],
                    caption=table_data["caption"],
                    data=table_data["data"],
                    format=table_data.get("format", "csv")
                )
                tables.append(table_obj)
        
        # If we don't have citations from DORA, extract them from results
        if not citations:
            logger.info("Extracting citations from research results")
            for citation_data in results_data.get("citations", []):
                citation = Citation(
                    key=citation_data["key"],
                    authors=citation_data["authors"],
                    title=citation_data["title"],
                    year=citation_data["year"],
                    journal=citation_data.get("journal"),
                    doi=citation_data.get("doi"),
                    url=citation_data.get("url")
                )
                citations.append(citation)
        
        # Enrich citations with metrics from Scite.ai regardless of citation source
        logger.info(f"Enriching {len(citations)} citations with Scite.ai metrics")
        enriched_citations = await self.citation_service.enrich_all_citations(citations)
        
        # Create publication ID
        pub_id = str(uuid.uuid4())
        
        # Create publication
        publication = Publication(
            id=pub_id,
            metadata=metadata,
            sections=sections,
            figures=figures,
            tables=tables,
            citations=enriched_citations,
            status=PublicationStatus.DRAFT,
            last_updated=datetime.now().isoformat()
        )
        
        # Store publication
        self.publications[pub_id] = publication
        
        # Save publication to file
        self._save_publication(publication)
        
        logger.info(f"Created publication with ID: {pub_id}")
        return publication
    
    async def _generate_sections_from_results(self, 
                                          results_data: Dict[str, Any],
                                          template: Optional[str] = None) -> Dict[str, str]:
        """Generate manuscript sections from research results
        
        Args:
            results_data: Research results data
            template: Template name (optional)
            
        Returns:
            Dictionary of section name -> content
        """
        # Default template sections
        default_sections = {
            "Introduction": "Introduction section content...",
            "Methods": "Methods section content...",
            "Results": "Results section content...",
            "Discussion": "Discussion section content...",
            "Conclusion": "Conclusion section content..."
        }
        
        # Use results data to generate better content
        if "methods" in results_data:
            default_sections["Methods"] = results_data["methods"]
        
        if "results" in results_data:
            default_sections["Results"] = results_data["results"]
        
        if "discussion" in results_data:
            default_sections["Discussion"] = results_data["discussion"]
        
        if "conclusion" in results_data:
            default_sections["Conclusion"] = results_data["conclusion"]
        
        # In a real implementation, this would use a more sophisticated approach
        # like natural language generation or AI templates
        
        # Load template if provided
        if template:
            template_path = os.path.join("templates", f"{template}.json")
            if os.path.exists(template_path):
                try:
                    with open(template_path, 'r') as f:
                        template_data = json.load(f)
                    
                    # Use template sections and placeholders
                    template_sections = template_data.get("sections", {})
                    for section_name, section_template in template_sections.items():
                        # Replace placeholders
                        section_content = section_template
                        for key, value in results_data.items():
                            if isinstance(value, str):
                                section_content = section_content.replace(f"{{{key}}}", value)
                        
                        default_sections[section_name] = section_content
                
                except Exception as e:
                    logger.error(f"Error loading template: {e}")
        
        return default_sections
    
    def _save_publication(self, publication: Publication) -> None:
        """Save publication to file
        
        Args:
            publication: Publication to save
        """
        # Create publication directory
        pub_dir = os.path.join(self.output_dir, publication.id)
        os.makedirs(pub_dir, exist_ok=True)
        
        # Save publication as JSON
        pub_path = os.path.join(pub_dir, "publication.json")
        with open(pub_path, 'w') as f:
            json.dump(publication.to_dict(), f, indent=2)
        
        # Save publication as Markdown
        md_path = os.path.join(pub_dir, "manuscript.md")
        with open(md_path, 'w') as f:
            f.write(publication.to_markdown())
        
        # Save citations as BibTeX
        bib_path = os.path.join(pub_dir, "references.bib")
        with open(bib_path, 'w') as f:
            f.write(publication.to_bibtex())
        
        # Generate PDF if available
        if HAS_REPORTLAB:
            pdf_path = os.path.join(pub_dir, "manuscript.pdf")
            publication.generate_pdf(pdf_path)
        
        logger.info(f"Saved publication to {pub_dir}")
    
    async def format_for_journal(self, 
                              publication_id: str, 
                              journal: str) -> Publication:
        """Format publication for specific journal
        
        Args:
            publication_id: Publication ID
            journal: Journal name
            
        Returns:
            Formatted publication
        """
        if publication_id not in self.publications:
            raise ValueError(f"Unknown publication ID: {publication_id}")
        
        publication = self.publications[publication_id]
        
        # Create journal adapter
        adapter = JournalFormatAdapter(journal)
        
        # Format publication
        formatted_pub = adapter.format_publication(publication)
        
        # Save formatted publication
        formatted_id = f"{publication_id}_{journal.lower()}"
        formatted_pub.id = formatted_id
        self.publications[formatted_id] = formatted_pub
        
        # Save to file