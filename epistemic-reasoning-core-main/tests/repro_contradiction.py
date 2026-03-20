import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ahankara.layer import Ahankara
from manas.layer import Manas
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph

def run_stress_test():
    print("Initializing Episteme Componets...")
    chitta = ChittaGraph()
    manas = Manas()
    buddhi = Buddhi(chitta)
    episteme = Ahankara(manas, buddhi, chitta)
    
    class EngineWrapper:
        def __init__(self, engine): self.engine = engine
        def teach(self, text): 
            print(f"Teaching: {text}")
            self.engine.process(text)
        def ask(self, text, expect):
            print(f"Query: {text}")
            ans = self.engine.ask(text)
            print(f"Answer: {ans}")
            status = "PASS" if expect.lower() in ans.lower() or (expect=="UNKNOWN" and "not certain" in ans.lower()) else "FAIL"
            print(f"Verdict: {status} (Expected: {expect})\n")
    
    agent = EngineWrapper(episteme)
    
    print("\n--- CASE 1: Classic Penguin (Specific beats General) ---")
    agent.teach("Birds can fly.")
    agent.teach("Penguins are birds.")
    agent.teach("Penguins cannot fly.")
    agent.ask("Can penguins fly?", "No")
    
    print("\n--- CASE 2: Inherited Negation (General beats Silence) ---")
    agent.teach("Mammals do not have gills.")
    agent.teach("Bats are mammals.")
    agent.ask("Do bats have gills?", "No")
    
    print("\n--- CASE 3: Contradiction via Inheritance (Conflict) ---")
    # This setup is tricky.
    # Animal -> breathes
    # Rock -> not breathes
    # Sponge -> Animal (so breathes)
    # Sponge -> Rock (logical error, but let's test conflict)
    # If we teach "Sponge is a Rock", it inherits "not breathes".
    # If we teach "Sponge is an Animal", it inherits "breathes".
    # What happens?
    print("Resetting for Case 3...")
    chitta.clear() 
    
    agent.teach("Animals can move.")
    agent.teach("Rocks cannot move.")
    
    # Create the conflict
    # Note: System might reject "Sponge is a Rock" if "Sponge is an Animal" is already known 
    # and disjointness is checked? (Not yet implemented)
    agent.teach("Sponge is an animal.")
    agent.teach("Sponge is a rock.")
    
    print("Graph state:")
    # Check inheritance
    # Sponge -> Animal -> move (+1)
    # Sponge -> Rock -> move (-1)
    
    # Query
    agent.ask("Can sponge move?", "No") # Expect "No" due to Negation Dominance (Safety First)
    
    return

if __name__ == "__main__":
    run_stress_test()
