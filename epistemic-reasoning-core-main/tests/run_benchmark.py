"""
Generic Benchmark Runner for Episteme
=====================================
Usage: python3 tests/run_benchmark.py [benchmark_file.json]

Supports schema:
[
  {
    "id": "case_id",
    "description": "...",
    "setup": ["fact1", "fact2", ...],
    "queries": [
      { "text": "Question?", "expected": "YES/NO/UNKNOWN/INVALID/..." }
    ]
  }
]
"""

import json
import logging
import sys
import os
from pathlib import Path
from dataclasses import asdict

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ahankara.layer import Ahankara
from manas.layer import Manas
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph
from common.types import Verdict

# Configure Logging
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "benchmark_execution.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)

logger = logging.getLogger("Benchmark")

def run_benchmark(file_path: str):
    print(f"Running Benchmark: {file_path} (Logs: {LOG_FILE})")
    
    with open(file_path, 'r') as f:
        cases = json.load(f)
        
    passed_queries = 0
    total_queries = 0
    
    for case in cases:
        cid = case.get('id', 'unknown')
        desc = case.get('description', '')
        setup = case.get('setup', [])
        queries = case.get('queries', [])
        
        print(f"\nCASE: {cid} - {desc}")
        logger.info(f"CASE: {cid} - {desc}")
        
        # Fresh System
        chitta = ChittaGraph()
        manas = Manas(llm_backend="mock")
        buddhi = Buddhi(chitta)
        ahankara = Ahankara(manas, buddhi, chitta)
        
        # Teach Only Once per Case
        for fact in setup:
            ahankara.process(fact)
            
        # Run Queries
        for q in queries:
            total_queries += 1
            text = q['text']
            expected = q['expected'].upper()
            
            logger.info(f"QUERY: {text} (Expect: {expected})")
            
            # Execute
            parsed = manas.parse(text)
            # Epistemic Hygiene Check for INVALID
            # Normally Buddhi.answer handles logic, but if Manas can flagged it?
            # Manas doesn't natively flag INVALID yet in my implementation, 
            # I relied on Buddhi.answer logic or Manas fallback.
            
            # Let's see what Buddhi answer returns
            answer = buddhi.answer(parsed)
            # AnswerProof verdict is str
            actual = answer.verdict.upper() if isinstance(answer.verdict, str) else answer.verdict.value.upper()
            
            # Match
            match = (actual == expected)
            
            status = "PASS" if match else f"FAIL (Got {actual})"
            print(f"  [{status}] {text}")
            logger.info(f"  RESULT: {status} (Actual: {actual})")
            
            if match:
                passed_queries += 1
        
        # Handle Temporal Updates
        updates = case.get('updates', [])
        for update in updates:
            fact = update.get('fact')
            print(f"  [UPDATE] Teaching: {fact}")
            logger.info(f"  [UPDATE] Teaching: {fact}")
            ahankara.process(fact)
            
            for q in update.get('queries', []):
                total_queries += 1
                text = q['text']
                expected = q['expected'].upper()
                
                logger.info(f"QUERY (Post-Update): {text} (Expect: {expected})")
                
                # Execute
                parsed = manas.parse(text)
                answer = buddhi.answer(parsed)
                actual = answer.verdict.upper() if isinstance(answer.verdict, str) else answer.verdict.value.upper()
                
                match = (actual == expected)
                status = "PASS" if match else f"FAIL (Got {actual})"
                print(f"  [{status}] {text}")
                logger.info(f"  RESULT: {status} (Actual: {actual})")
                
                if match:
                    passed_queries += 1

    print(f"\nSummary: {passed_queries}/{total_queries} queries passed.")
    if passed_queries != total_queries:
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 tests/run_benchmark.py [file.json]")
        sys.exit(1)
    run_benchmark(sys.argv[1])
