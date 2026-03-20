
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ahankara.layer import Ahankara
from manas.layer import Manas
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph

def test_negative_inheritance():
    print("Initializing Episteme Componets...")
    chitta = ChittaGraph()
    manas = Manas()
    buddhi = Buddhi(chitta)
    episteme = Ahankara(manas, buddhi, chitta)
    
    # helper wrapper
    class EngineWrapper:
        def __init__(self, engine): self.engine = engine
        def teach(self, text): self.engine.process(text)
        def response(self, text): return self.engine.ask(text)
        
    episteme_wrapper = EngineWrapper(episteme)
    
    print("\nTeaching Defaults...")
    # 1. Mammals do not have gills (Class-level negation)
    episteme_wrapper.teach("Mammals do not have gills.")
    
    # 2. Bats are mammals
    episteme_wrapper.teach("Bats are mammals.")
    
    # query
    query = "Do bats have gills?"
    print(f"\nQuery: {query}")
    
    response = episteme_wrapper.response(query)
    print(f"Answer: {response}")
    
    # Check internals
    if "No" in response or "no" in response.lower():
        print("RESULT: NEGATIVE INHERITANCE WORKS (YES!)")
    else:
        print("RESULT: NEGATIVE INHERITANCE FAILED (UNKNOWN)")

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    test_negative_inheritance()
