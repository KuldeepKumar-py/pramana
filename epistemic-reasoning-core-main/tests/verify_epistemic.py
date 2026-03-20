
import sys
import os
sys.path.append("/Users/jeevan/Desktop/Episteme")

from buddhi.belief import EpistemicType, Belief
from chitta.graph import ChittaGraph
from manas.layer import Manas
from common.types import Verdict, Answer

def test_epistemic_flow():
    print("=== Testing Epistemic Flow ===")
    
    # 1. Setup
    graph = ChittaGraph()
    manas = Manas(llm_backend="mock") 
    
    # 2. Parse "Birds can fly" -> Should be DEFAULT (capability/is_a) or OBSERVATION
    # Manas mock parser maps "can_fly" to relation. 
    # My updated Manas logic: "can_" -> capability -> DEFAULT?
    # Let's see what it does.
    text1 = "Birds can fly"
    proposal1 = manas.parse(text1)
    print(f"Input: '{text1}'")
    print(f"Proposal Type: {proposal1.get('epistemic_type')}")
    
    # 3. Add to Graph
    # Chitta should pick up the type
    bid1 = graph.add_belief_from_proposal(proposal1)
    belief1 = graph.get(bid1)
    print(f"Stored Belief 1 State: {belief1.epistemic_state}")
    
    # 4. Parse "Penguins cannot fly" -> Should be EXCEPTION (Negative polarity)
    text2 = "Penguins cannot fly"
    proposal2 = manas.parse(text2) 
    print(f"\nInput: '{text2}'")
    print(f"Proposal Polarity: {proposal2.get('polarity')}")
    print(f"Proposal Type: {proposal2.get('epistemic_type')}")
    
    bid2 = graph.add_belief_from_proposal(proposal2)
    belief2 = graph.get(bid2)
    print(f"Stored Belief 2 State: {belief2.epistemic_state}")
    
    # 5. Parse "Socrates is a man" -> Should be DEFAULT (is_a)
    text3 = "Socrates is a man"
    proposal3 = manas.parse(text3)
    print(f"\nInput: '{text3}'")
    print(f"Proposal Type: {proposal3.get('epistemic_type')}")
    
    bid3 = graph.add_belief_from_proposal(proposal3)
    belief3 = graph.get(bid3)
    print(f"Stored Belief 3 State: {belief3.epistemic_state}")

    # 6. Test INVALID Verdict
    print("\n=== Testing INVALID Verdict ===")
    ans = Answer(
        query="Garbage",
        verdict=Verdict.INVALID,
        confidence=0.0
    )
    print(f"Verdict: {ans.verdict}")
    print(f"Natural Language: {ans.to_natural_language()}")
    
    # 7. Test UNKNOWN Belief
    print("\n=== Testing UNKNOWN Belief ===")
    proposal_unknown = {"template": "relation", "canonical": {}, "raw_text": "Who knows?"}
    bid_u = graph.add_unknown(proposal_unknown)
    belief_u = graph.get(bid_u)
    print(f"Unknown Belief State: {belief_u.epistemic_state}")
    
    # 8. Test Promotion Logic (Manually trigger)
    from buddhi.layer import Buddhi
    buddhi = Buddhi(graph, learning_mode=False)
    
    # Add a hypothesis
    prop_hyp = {"template": "is_a", "canonical": {"subject": "A", "object": "B"}, "raw_text": "A is B"}
    bid_h = graph.add_hypothetical(prop_hyp, confidence=0.8) # High confidence to trigger promotion
    belief_h = graph.get(bid_h)
    print(f"\nBefore Promotion: {belief_h.epistemic_state}")
    
    buddhi.promote_hypotheses()
    print(f"After Promotion: {belief_h.epistemic_state}")
    # Should be DEFAULT because template is "is_a"

if __name__ == "__main__":
    test_epistemic_flow()
