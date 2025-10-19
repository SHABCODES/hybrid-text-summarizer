#!/usr/bin/env python3
"""
Demo script for Hybrid Text Summarizer
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessor import RobustTextPreprocessor
from src.summarizer import ImprovedHybridSummarizer

def main():
    print(" Hybrid Text Summarizer Demo")
    print("=" * 40)
    
    # Sample article
    sample_article = """
    Artificial intelligence is transforming many industries. Machine learning algorithms 
    can now perform tasks that were previously thought to require human intelligence. 
    Natural language processing has seen significant advances in recent years. 
    Companies are investing heavily in AI research and development. 
    The future of AI looks promising with continued innovation. Many experts believe 
    that AI will create new job opportunities while automating routine tasks. 
    Ethical considerations in AI development are becoming increasingly important.
    """
    
    # Initialize components
    preprocessor = RobustTextPreprocessor()
    summarizer = ImprovedHybridSummarizer(summary_ratio=0.3)
    
    # Process and summarize
    processed = preprocessor.prepare_single_article(sample_article)
    if processed:
        summary = summarizer.summarize(
            processed['original_sentences'],
            processed['preprocessed_sentences']
        )
        
        print(" Original Text:")
        print(sample_article)
        print(f"\n Original sentences: {processed['num_sentences']}")
        print("\n Generated Summary:")
        print(summary)
        print(f"\n Summary length: {len(summary)} characters")
    else:
        print(" Could not process the sample text")

if __name__ == "__main__":
    main()
