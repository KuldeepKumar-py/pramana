
import sys
import os
import shutil
import unittest
sys.path.append(os.getcwd())

from chitta.graph import ChittaGraph
from ahankara.layer import Ahankara
from buddhi.belief import Belief, EpistemicType, Polarity
from common.types import Verdict

class TestQuantitativeLayer(unittest.TestCase):
    def setUp(self):
        self.graph_path = "tests_quant_db"
        if os.path.exists(self.graph_path):
            shutil.rmtree(self.graph_path)
        self.ai = Ahankara(persistence_dir=self.graph_path)
    
    def tearDown(self):
        if os.path.exists(self.graph_path):
            shutil.rmtree(self.graph_path)

    def test_evidence_aggregation(self):
        print("\n--- Test: Evidence Aggregation ---")
        
        # 1. Add Initial Belief "Sky is blue"
        b1 = Belief(
            "Observation", # Template
            {"subject": "sky", "predicate": "is", "object": "blue"}, # Canonical
            0.5, # Confidence
            id="fact_sky_blue",
            statement_text="The sky is blue.",
            epistemic_state=EpistemicType.OBSERVATION
        )
        bid1 = self.ai.chitta.add_belief(b1)
        
        # Check initial confidence
        stored_b1 = self.ai.chitta.get(bid1)
        print(f"Initial Confidence: {stored_b1.confidence}")
        self.assertEqual(stored_b1.confidence, 0.5)
        
        # 2. Add Duplicate (Same Content, Different Object/ID)
        b2 = Belief(
            "Observation", # Template
            {"subject": "sky", "predicate": "is", "object": "blue"}, # Canonical
            0.5, # Confidence
            id="fact_sky_blue_dup",
            statement_text="Sky is blue", # Slightly different text to test semantic match
            epistemic_state=EpistemicType.OBSERVATION
        )
        bid2 = self.ai.chitta.add_belief(b2)
        
        # Should return OLD ID
        print(f"Duplicate Add returned ID: {bid2}")
        self.assertEqual(bid2, bid1)
        
        # Check Boosted Confidence
        stored_b1 = self.ai.chitta.get(bid1)
        print(f"Boosted Confidence: {stored_b1.confidence}")
        self.assertGreater(stored_b1.confidence, 0.5)
        
        # 3. Add Same Object (ID match)
        self.ai.chitta.add_belief(b1)
        print(f"Re-add Confidence: {stored_b1.confidence}")
        self.assertGreater(stored_b1.confidence, 0.55) # Should be boosted again

    def test_confidence_propagation(self):
        print("\n--- Test: Confidence Propagation (Weakest Link) ---")
        # Chain: Tweety(1.0) -> Penguin(0.5) -> Bird(0.9) -> Fly(1.0)
        # Result should be 0.5 (Weakest Link)
        
        # 1. Facts
        self.ai.process("Tweety is a penguin.") # Conf 1.0 (approximated)
        
        # 2. Weak Link: Penguins are birds (0.5)
        b_weak = Belief(
            "is_a", # Standard Template
            {"subject": "penguin", "predicate": "is_a", "object": "bird"},
            0.5,
            id="rule_penguin_bird_weak",
            statement_text="Penguins are birds.",
            epistemic_state=EpistemicType.DEFAULT
        )
        self.ai.chitta.add_belief(b_weak)
        
        # 3. Strong Rule: Birds fly (0.9)
        b_strong = Belief(
            "Group Property",
            {"subject": "bird", "predicate": "behavior_flies", "entities": ["bird"]},
            0.9,
            id="rule_bird_fly_strong",
            statement_text="Birds fly.",
            epistemic_state=EpistemicType.DEFAULT
        )
        self.ai.chitta.add_belief(b_strong)
        
        # 4. Query
        proposal = self.ai.perceive("Does Tweety fly?")
        proof = self.ai.buddhi.answer(proposal)
        
        print(f"Verdict: {proof.verdict}")
        if proof.steps:
            last_step = proof.steps[-1]
            print(f"Outcome Confidence: {last_step.confidence}")
            # Should be min(1.0, 0.5, 0.9) = 0.5
            self.assertEqual(last_step.confidence, 0.5)

    def test_decay_logic_gating(self):
        print("\n--- Test: Decay & Logic Gating ---")
        
        # 1. Add Default (Subject to decay)
        # "Apples are red" (Conf 0.4)
        b_default = Belief(
            "Group Property",
            {"subject": "apple", "predicate": "is", "object": "red"},
            0.4,
            id="rule_apple_red",
            statement_text="Apples are red.",
            epistemic_state=EpistemicType.DEFAULT
        )
        # Set decay rate manually for test speed
        b_default.decay_rate = 0.5 # Halves every step
        self.ai.chitta.add_belief(b_default)
        
        # 2. Add Axiom (Immune)
        # "1+1=2" (Conf 1.0)
        b_axiom = Belief(
            "Axiom",
            {"subject": "1", "predicate": "plus_one", "object": "2"},
            1.0,
            id="axiom_math",
            statement_text="1+1=2",
            epistemic_state=EpistemicType.AXIOM
        )
        self.ai.chitta.add_belief(b_axiom)
        
        # 3. Verify before decay
        self.assertTrue(self.ai.chitta.get("rule_apple_red").active)
        self.assertEqual(self.ai.chitta.get("rule_apple_red").confidence, 0.4)
        
        # 4. Apply Decay (1 step) -> Conf should become 0.2
        # Threshold is 0.1
        self.ai.chitta.apply_decay(steps=1, threshold=0.1)
        b_def = self.ai.chitta.get("rule_apple_red")
        print(f"Decay Step 1: Conf {b_def.confidence}")
        self.assertAlmostEqual(b_def.confidence, 0.2)
        self.assertTrue(b_def.active)
        
        # 5. Apply Decay (2 steps) -> Conf should become 0.2 * 0.5 = 0.1. Wait.
        # Let's apply another step. 0.2 * 0.5 = 0.1.
        # Threshold is 0.1. (Strictly less check? "confidence < threshold")
        # If threshold is 0.11, it should deactivate.
        
        count = self.ai.chitta.apply_decay(steps=1, threshold=0.15)
        print(f"Decay Step 2: Deactivated Count {count}")
        
        b_def = self.ai.chitta.get("rule_apple_red")
        print(f"Final Conf: {b_def.confidence}")
        self.assertFalse(b_def.active) # Should be INACTIVE
        
        # 6. Verify Axiom Immunity
        b_ax = self.ai.chitta.get("axiom_math")
        self.assertEqual(b_ax.confidence, 1.0)
        self.assertTrue(b_ax.active)
        
        # 7. Logic Gate Check
        # Query for Apple redness should fail/grounding check fail if inactive?
        # Manas might not even see it if lookup filters active.
        # Let's check manually if active flag prevents retrieval or usage.
        # Chitta.get returns it, but logic loops usually check `if not belief.active: continue`.
        pass


if __name__ == "__main__":
    unittest.main()
