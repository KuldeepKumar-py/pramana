"""
Scientific Benchmark Runner for Episteme
========================================

Executes the scientific benchmark suite and logs detailed proof traces.
This allows verification of *how* the system reached its conclusions.

Output Log: `tests/logs/scientific_benchmark_execution.log`
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
LOG_FILE = LOG_DIR / "scientific_benchmark_execution.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'  # Overwrite each run
)

logger = logging.getLogger("ScientificBenchmark")

def log_section(title: str):
    logger.info("=" * 60)
    logger.info(title.center(60))
    logger.info("=" * 60)

def log_item(key: str, value: str):
    logger.info(f"{key:<20}: {value}")

def run_scientific_benchmark():
    print(f"Running Scientific Benchmark... (Logs: {LOG_FILE})")
    
    # Load Data
    data_path = Path(__file__).parent / "data" / "benchmark_scientific.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    cases = data['cases']
    passed = 0
    total = len(cases)
    
    log_section("STARTING BENCHMARK RUN")
    logger.info(f"Total Cases: {total}")
    
    for case in cases:
        cid = case['id']
        scenario = case['scenario']
        facts = case['setup_facts']
        query_text = case['query']
        expected_str = case['expected_verdict']
        
        logger.info("\n" + "-" * 40)
        logger.info(f"CASE: {cid} ({scenario})")
        logger.info(f"Diff: {case['difficulty']}, Desc: {case['description']}")
        
        # Initialize Fresh System (Strict Mode)
        chitta = ChittaGraph()
        manas = Manas(llm_backend="mock")
        buddhi = Buddhi(chitta)
        ahankara = Ahankara(manas, buddhi, chitta)
        
        # 1. Teach Facts
        logger.info(f"TEACHING {len(facts)} FACTS:")
        for fact in facts:
            ahankara.process(fact)
            logger.info(f"  + Taught: {fact}")
            
        # 2. Query
        logger.info(f"QUERY: {query_text}")
        
        # Execute Query manually to access Answer object directly if possible,
        # or capture logs from ahankara.ask()
        
        # We want the FULL answer object with proof trace
        # Manas Parse -> Buddhi Answer
        parsed_query = manas.parse(query_text)
        answer = buddhi.answer(parsed_query)
        
        # 3. Log Proof Trace
        logger.info("PROOF TRACE:")
        for step in answer.steps:
             logger.info(f"  Step: Rule={step.rule}, Out={step.output}")
             
        if answer.conflicts:
            logger.info(f"  CONFLICTS: {answer.conflicts}")
        
        # 4. Verification
        actual_verdict = answer.verdict.upper()  # YES, NO, UNKNOWN
        
        # Map expected format (POSITIVE/NEGATIVE) to Verdict (YES/NO)
        if expected_str == "POSITIVE":
            expected_v = "YES"
        elif expected_str == "NEGATIVE":
            expected_v = "NO"
        elif expected_str in ["REFUSED", "UNKNOWN"]:
            expected_v = "UNKNOWN" # Simplify for now
        else:
            expected_v = expected_str

        match = (actual_verdict == expected_v)
        
        log_item("Expected Verdict", expected_v)
        log_item("Actual Verdict", actual_verdict)
        
        if match:
            logger.info("RESULT: PASS ✅")
            passed += 1
        else:
            logger.info("RESULT: FAIL ❌")
            
    # Final Summary
    log_section("BENCHMARK SUMMARY")
    logger.info(f"Passed: {passed}/{total}")
    logger.info(f"Accuracy: {(passed/total)*100:.1f}%")
    
    print(f"Done. Passed: {passed}/{total}. See logs for details.")

if __name__ == "__main__":
    run_scientific_benchmark()
