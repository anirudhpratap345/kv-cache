"""
Research Paper Analysis Tool
Helps break down and understand research papers systematically
"""

import json
from typing import Dict, List, Any

class PaperBreakdown:
    """Structured framework for analyzing research papers"""
    
    def __init__(self, title: str, arxiv_id: str = "2305.14314"):
        self.title = title
        self.arxiv_id = arxiv_id
        self.sections = {}
    
    def add_section(self, name: str, content: Dict[str, Any]):
        """Add a section of the paper analysis"""
        self.sections[name] = content
    
    def to_markdown(self) -> str:
        """Generate markdown summary"""
        md = f"# {self.title}\n\n"
        md += f"**ArXiv ID:** {self.arxiv_id}\n\n"
        
        for section_name, section_content in self.sections.items():
            md += f"## {section_name}\n\n"
            
            for key, value in section_content.items():
                if isinstance(value, list):
                    md += f"### {key}\n"
                    for item in value:
                        md += f"- {item}\n"
                    md += "\n"
                else:
                    md += f"**{key}:** {value}\n\n"
        
        return md


def create_analysis_template() -> PaperBreakdown:
    """Create a template for paper analysis"""
    
    paper = PaperBreakdown(
        title="Research Paper on KV Caching / LLM Optimization",
        arxiv_id="2305.14314"
    )
    
    # Abstract & Overview
    paper.add_section("Abstract & Overview", {
        "Main Topic": "[To be filled]",
        "Key Problem": "[What problem does this solve?]",
        "Key Contribution": "[What's novel?]",
        "Impact": "[Why does this matter?]",
    })
    
    # Technical Approach
    paper.add_section("Technical Approach", {
        "Methods": [
            "[Method 1 - description]",
            "[Method 2 - description]",
            "[Method 3 - description]",
        ],
        "Key Algorithms": [
            "[Algorithm name and how it works]",
            "[Key insight]",
        ],
        "Mathematical Foundation": "[Key equations and their meaning]",
    })
    
    # Experiments
    paper.add_section("Experimental Evaluation", {
        "Datasets/Benchmarks": [
            "[Benchmark 1]",
            "[Benchmark 2]",
        ],
        "Baselines Compared": [
            "[Baseline 1]",
            "[Baseline 2]",
        ],
        "Key Results": [
            "[Result 1: X improved by Y%]",
            "[Result 2: Performance improvement]",
        ],
        "Performance Metrics": [
            "[Latency reduction]",
            "[Throughput improvement]",
            "[Memory savings]",
        ],
    })
    
    # Strengths
    paper.add_section("Strengths & Contributions", {
        "Main Strengths": [
            "[Strength 1]",
            "[Strength 2]",
            "[Strength 3]",
        ],
        "Novel Aspects": "[What's new compared to prior work]",
        "Practical Impact": "[Real-world applicability]",
    })
    
    # Weaknesses & Limitations
    paper.add_section("Limitations & Future Work", {
        "Limitations": [
            "[Limitation 1]",
            "[Limitation 2]",
            "[Limitation 3]",
        ],
        "Future Directions": [
            "[Potential extension 1]",
            "[Potential extension 2]",
        ],
        "Unsolved Challenges": "[What still needs to be addressed]",
    })
    
    # Relevance to KV Cache
    paper.add_section("Relevance to KV Caching", {
        "How It Relates": "[Connection to KV cache problem]",
        "Applicable Techniques": [
            "[Technique 1 - applicable to KV cache]",
            "[Technique 2 - applicable]",
        ],
        "Integration Points": "[Where this could be integrated]",
    })
    
    return paper


def print_analysis_guide():
    """Print guide for breaking down the paper"""
    
    guide = """
==========================================
HOW TO BREAK DOWN A RESEARCH PAPER
==========================================

1. READ IN THIS ORDER:
   - Title & Abstract (understand the core idea)
   - Introduction (why this matters)
   - Conclusion (what was achieved)
   - Results/Figures (see the numbers)
   - Methods (understand the approach)

2. IDENTIFY THESE KEY ELEMENTS:
   [*] Problem Statement: What's the issue?
   [*] Proposed Solution: How does it solve it?
   [*] Key Innovation: What's novel?
   [*] Experimental Setup: How was it tested?
   [*] Results: What were the improvements?

3. FOR EACH SECTION, ASK:
   Q: What is this section trying to tell me?
   Q: What are the key findings?
   Q: How does this connect to the overall paper?
   Q: What would a practitioner need to know?

4. CREATE YOUR SUMMARY:
   - 1 sentence: Core idea
   - 3 sentences: Main contributions
   - 5 sentences: Methods and results
   - 2 sentences: Why it matters
   - 3 sentences: How to apply it

5. FOR KV CACHE RELEVANCE:
   [*] Does it improve cache efficiency?
   [*] Does it reduce memory requirements?
   [*] Does it speed up inference?
   [*] Is it applicable to multi-layer caching?
   [*] What's the expected speedup?

==========================================
"""
    
    print(guide)


if __name__ == "__main__":
    # Show the guide
    print_analysis_guide()
    
    # Create template
    paper = create_analysis_template()
    
    # Save template
    print("\n" + "="*70)
    print("Paper Analysis Template created!")
    print("="*70 + "\n")
    
    # Print markdown
    print(paper.to_markdown())
