
import sys
sys.path.append("/Users/jeevan/Desktop/Episteme")
from chitta.graph import ChittaGraph
from manas.layer import Manas
from buddhi.layer import Buddhi
from ahankara.layer import Ahankara

def inspect_storage():
    chitta = ChittaGraph()
    manas = Manas(llm_backend="mock")
    buddhi = Buddhi(chitta)
    ahankara = Ahankara(manas, buddhi, chitta)
    
    fact = "Republicans are not pacifists"
    ahankara.process(fact)
    
    print(f"Fact: {fact}")
    for bid, belief in chitta.beliefs.items():
        print(f"Belief: {belief.statement_text}")
        print(f"  Entities: {belief.entities}")
        print(f"  Predicates: {belief.predicates}")
        print(f"  Polarity: {belief.polarity}")
        print(f"  Epistemic State: {belief.epistemic_state}")

if __name__ == "__main__":
    inspect_storage()
