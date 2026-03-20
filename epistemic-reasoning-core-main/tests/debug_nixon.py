
import sys
import os
sys.path.append("/Users/jeevan/Desktop/Episteme")

from ahankara.layer import Ahankara
from manas.layer import Manas
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph
import logging

# Configure basic logging to stdout
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def debug_nixon():
    print("=== Debugging Nixon Diamond ===")
    
    chitta = ChittaGraph()
    manas = Manas(llm_backend="mock")
    buddhi = Buddhi(chitta)
    ahankara = Ahankara(manas, buddhi, chitta)
    
    facts = [
      "Quakers are pacifists",
      "Republicans are not pacifists",
      "Nixon is a Quaker",
      "Nixon is a Republican"
    ]
    
    print("\n[TEACHING]")
    for f in facts:
        print(f"Teaching: {f}")
        ahankara.process(f)
        
    query = "Is Nixon a pacifist?"
    print(f"\n[QUERY] {query}")
    
    parsed = manas.parse(query)
    answer = buddhi.answer(parsed)
    
    print(f"\n[RESULT] Verdict: {answer.verdict}")
    print("Proof Trace:")
    for step in answer.steps:
        print(f"  {step.rule} -> {step.output} (Conf: {step.confidence})")
        
    if answer.conflicts:
        print(f"Conflicts: {answer.conflicts}")

if __name__ == "__main__":
    debug_nixon()
