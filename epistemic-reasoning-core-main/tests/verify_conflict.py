
import sys
import os
import shutil
import unittest
sys.path.append(os.getcwd())

from chitta.graph import ChittaGraph
from ahankara.layer import Ahankara
from buddhi.belief import Belief, EpistemicType, Polarity
from common.types import Verdict

class TestConflictResolution(unittest.TestCase):
    def setUp(self):
        self.graph_path = "tests_conflict_db"
        if os.path.exists(self.graph_path):
            shutil.rmtree(self.graph_path)
        self.ai = Ahankara(persistence_dir=self.graph_path)
    
    def tearDown(self):
        if os.path.exists(self.graph_path):
            shutil.rmtree(self.graph_path)

    def test_horizontal_conflict_nixon(self):
        print("\n--- Test: Nixon Diamond (Horizontal) ---")
        # 1. Inject conflicting rules (Equal Rank)
        # Quakers have gills (Default)
        b1 = Belief(
            id="rule_quaker_gills",
            canonical={"subject": "quaker", "predicate": "has_gills", "entities": ["quaker"]},
            statement_text="Quakers have gills.",
            epistemic_state=EpistemicType.DEFAULT,
            polarity_value=1,
            template="Group Property",
            confidence=0.9
        )
        self.ai.chitta.add_belief(b1)
        
        # Republicans do NOT have gills (Default)
        b2 = Belief(
            id="rule_republican_no_gills",
            canonical={"subject": "republican", "predicate": "has_gills", "entities": ["republican"]},
            statement_text="Republicans do not have gills.",
            epistemic_state=EpistemicType.DEFAULT,
            polarity_value=-1,
            template="Group Property",
            confidence=0.9
        )
        self.ai.chitta.add_belief(b2)
        
        # 2. Teach facts
        self.ai.process("Nixon is a republican.")
        self.ai.process("Nixon is a quaker.")
        
        # 3. Query
        proposal = self.ai.perceive("Does Nixon have gills?")
        proof = self.ai.buddhi.answer(proposal)
        
        print(f"Verdict: {proof.verdict}")
        if proof.conflicts:
            print(f"Conflict: {proof.conflicts[0].predicate}")
            
        self.assertEqual(str(proof.verdict).lower(), "conflict")

    def test_vertical_conflict_specificity(self):
        print("\n--- Test: Specificity (Vertical) ---")
        # A -> B
        # A has X (Default)
        # B has NOT X (Default)
        # Specificity should prefer A (Dist 1 < Dist 2)
        
        # 1. Inject rules (Manual Injection to bypass Parser Negation Bug)
        # Tweety is a Penguin
        self.ai.process("Tweety is a penguin.")
        # Penguin is a Bird
        self.ai.process("Penguins are birds.")
        
        # Bird flies (Default)
        b1 = Belief(
            id="rule_bird_flies",
            canonical={"subject": "bird", "predicate": "behavior_flies", "entities": ["bird"]},
            statement_text="Birds fly.",
            epistemic_state=EpistemicType.DEFAULT,
            polarity_value=1,
            template="Group Property",
            confidence=0.9
        )
        self.ai.chitta.add_belief(b1)
        
        # Penguin does NOT fly (Exception/Negative Default)
        b2 = Belief(
            id="rule_penguin_no_fly",
            canonical={"subject": "penguin", "predicate": "behavior_flies", "entities": ["penguin"]},
            statement_text="Penguins do not fly.",
            epistemic_state=EpistemicType.DEFAULT, 
            polarity_value=-1, # FORCE NEGATIVE
            template="Group Property",
            confidence=0.9
        )
        self.ai.chitta.add_belief(b2)
        
        # DEBUG: Check stored beliefs
        print("\nDEBUG: Stored Beliefs for 'penguin':")
        for bid in self.ai.chitta.entity_index.get("penguin", []):
            b = self.ai.chitta.get(bid)
            print(f"  - {b.statement_text} | Polarity: {b.polarity} | Preds: {b.predicates}")
        
        # 3. Query: Does Tweety fly?
        # Path: Tweety -> Penguin (Dist 1) -> Bird (Dist 2)
        # Penguin belief (Neg) at Dist 1.
        # Bird belief (Pos) at Dist 2.
        # Expectation: NO (Specificity Win)
        
        # 3. Query: Does Tweety fly?
        # Path: Tweety -> Penguin (Dist 1) -> Bird (Dist 2)
        # Penguin belief (Neg) at Dist 1.
        # Bird belief (Pos) at Dist 2.
        # Expectation: NO (Specificity Win)
        
        proposal = self.ai.perceive("Does Tweety fly?")
        print(f"Proposal Predicates: {proposal.get('predicates')}")
        
        proof = self.ai.buddhi.answer(proposal)
        
        print(f"Verdict: {proof.verdict}")
        if proof.steps:
            for s in proof.steps:
                print(f"[{s.rule}] {s.output}")
                    
        self.assertTrue(proof.verdict == "no" or proof.verdict == Verdict.NO)
        
        # Check that specificity rule was applied
        rules = [s.rule for s in proof.steps]
        self.assertIn("specificity_resolution", rules)

if __name__ == "__main__":
    unittest.main()
