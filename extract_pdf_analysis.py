#!/usr/bin/env python3
"""
PDF Extraction and Analysis Script
Extracts and analyzes content from research papers
"""

import pdfplumber
import re
from typing import Dict, List

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from PDF file"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += f"\n--- PAGE {page_num + 1} ---\n{page_text}"
    return text

def extract_sections(text: str) -> Dict[str, str]:
    """Extract major sections from paper text"""
    sections = {}
    
    # Extract title (usually at beginning)
    lines = text.split('\n')
    title_candidates = []
    for i, line in enumerate(lines[:50]):  # Check first 50 lines
        if len(line.strip()) > 20 and len(line.strip()) < 200:
            title_candidates.append(line.strip())
    
    if title_candidates:
        sections['title'] = title_candidates[0]
    
    # Try to extract abstract
    abstract_match = re.search(r'(?:ABSTRACT|Abstract)(.*?)(?:Introduction|1\.|INTRODUCTION)', text, re.IGNORECASE | re.DOTALL)
    if abstract_match:
        sections['abstract'] = abstract_match.group(1).strip()[:1000]
    
    # Try to extract introduction
    intro_match = re.search(r'(?:INTRODUCTION|Introduction|1\s+Introduction)(.*?)(?:Related Work|Background|2\.|RELATED|METHOD|Approach)', text, re.IGNORECASE | re.DOTALL)
    if intro_match:
        sections['introduction'] = intro_match.group(1).strip()[:1500]
    
    # Try to extract methodology
    method_match = re.search(r'(?:METHOD|METHODOLOGY|Methodology|Approach|3\..*?Method)(.*?)(?:EXPERIMENT|Evaluation|Result|4\.|RESULT)', text, re.IGNORECASE | re.DOTALL)
    if method_match:
        sections['methodology'] = method_match.group(1).strip()[:1500]
    
    # Try to extract results
    result_match = re.search(r'(?:RESULT|EVALUATION|Evaluation|Experiment|Experimental Result)(.*?)(?:CONCLUSION|Discussion|6\.|FUTURE)', text, re.IGNORECASE | re.DOTALL)
    if result_match:
        sections['results'] = result_match.group(1).strip()[:1500]
    
    # Try to extract conclusion
    conclusion_match = re.search(r'(?:CONCLUSION|Conclusion|FUTURE WORK|Future Work)(.*?)(?:REFERENCES|References|$)', text, re.IGNORECASE | re.DOTALL)
    if conclusion_match:
        sections['conclusion'] = conclusion_match.group(1).strip()[:1000]
    
    return sections

def extract_key_metrics(text: str) -> List[str]:
    """Extract numbers, percentages, and key metrics"""
    metrics = []
    
    # Find percentage improvements
    percent_pattern = r'(\d+\.?\d*)\s*%\s*(?:improvement|faster|better|reduction|increase)'
    for match in re.finditer(percent_pattern, text, re.IGNORECASE):
        metrics.append(match.group(0))
    
    # Find speedup/acceleration metrics
    speedup_pattern = r'(\d+\.?\d*)[×x]\s*(?:faster|speedup|acceleration)'
    for match in re.finditer(speedup_pattern, text, re.IGNORECASE):
        metrics.append(match.group(0))
    
    # Find throughput metrics
    throughput_pattern = r'(\d+\.?\d*)\s*(?:tokens|ops|QPS|requests)(?:/s|/sec)?'
    for match in re.finditer(throughput_pattern, text):
        metrics.append(match.group(0))
    
    return list(set(metrics))[:20]

def main():
    pdf_path = "d:\\KV Cache\\2305.14314v1.pdf"
    
    print("=" * 80)
    print("PDF RESEARCH PAPER ANALYSIS")
    print("=" * 80)
    
    # Extract text
    print("\n[1/3] Extracting text from PDF...")
    full_text = extract_pdf_text(pdf_path)
    print(f"✓ Extracted {len(full_text)} characters")
    
    # Extract sections
    print("\n[2/3] Extracting key sections...")
    sections = extract_sections(full_text)
    print(f"✓ Found {len(sections)} major sections")
    
    # Extract metrics
    print("\n[3/3] Identifying key metrics...")
    metrics = extract_key_metrics(full_text)
    print(f"✓ Found {len(metrics)} key metrics")
    
    # Display results
    print("\n" + "=" * 80)
    print("EXTRACTED CONTENT")
    print("=" * 80)
    
    if 'title' in sections:
        print(f"\n📄 TITLE:\n{sections['title']}")
    
    if 'abstract' in sections:
        print(f"\n📋 ABSTRACT:\n{sections['abstract']}")
    
    if 'introduction' in sections:
        print(f"\n🎯 INTRODUCTION:\n{sections['introduction']}")
    
    if 'methodology' in sections:
        print(f"\n🔬 METHODOLOGY:\n{sections['methodology']}")
    
    if 'results' in sections:
        print(f"\n📊 RESULTS & FINDINGS:\n{sections['results']}")
    
    if 'conclusion' in sections:
        print(f"\n✅ CONCLUSION:\n{sections['conclusion']}")
    
    if metrics:
        print(f"\n📈 KEY METRICS FOUND:")
        for metric in metrics:
            print(f"  - {metric}")
    
    # Save full text to file for reference
    with open("d:\\KV Cache\\extracted_pdf_content.txt", "w", encoding="utf-8") as f:
        f.write("FULL EXTRACTED PDF CONTENT\n")
        f.write("=" * 80 + "\n\n")
        f.write(full_text)
    
    print("\n" + "=" * 80)
    print(f"✓ Full extracted content saved to: extracted_pdf_content.txt")
    print("=" * 80)

if __name__ == "__main__":
    main()
