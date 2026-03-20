
import unittest
import os
import shutil
import json
import sys
sys.path.append(os.getcwd())

from chitta.graph import ChittaGraph
from ahankara.layer import Ahankara
from manas.predicate_normalizer import get_entity_normalizer

class TestLogicHardening(unittest.TestCase):
    OUTPUT_DIR = "./tests_logic_db"
    
    def setUp(self):
        if os.path.exists(self.OUTPUT_DIR):
            shutil.rmtree(self.OUTPUT_DIR)
        os.makedirs(self.OUTPUT_DIR)
        
    def tearDown(self):
        # Leave artifacts for inspection if needed
        pass

    def test_entity_normalization(self):
        print("\n--- Test 1: Entity Normalization ---")
        norm = get_entity_normalizer()
        
        # Test Socrates protection
        self.assertEqual(norm.normalize("Socrates"), "socrates")
        self.assertEqual(norm.normalize("socrates"), "socrates")
        self.assertNotEqual(norm.normalize("Socrates"), "socrate") # Should NOT be singularized
        
        # Test basic normalization
        self.assertEqual(norm.normalize("Birds"), "bird")
        self.assertEqual(norm.normalize("birds"), "bird")
        self.assertEqual(norm.normalize("Birds."), "bird")
        
        print("✓ Entity Normalization verified.")

    def test_honest_loading(self):
        print("\n--- Test 2: Honest Loading ---")
        chitta = ChittaGraph()
        
        # Create a dummy JSON with 1 valid and 1 corrupt belief
        valid_belief = {
            "id": "valid1",
            "template": "is_a",
            "canonical": {"subject": "cat", "object": "animal"},
            "entities": ["cat", "animal"],
            "predicates": ["is_a"],
            "epistemic_state": "DEFAULT",
            "confidence": 0.9,
            "active": True,
            "original_text": "Cats are animals.",
            "statement_text": "Cats are animals.",
            "source": {},
            "polarity_value": 1
        }
        
        corrupt_belief = {
            "id": "corrupt1",
            # Missing template!
            "canonical": {},
        }
        
        data = {
            "beliefs": {
                "valid1": valid_belief,
                "corrupt1": corrupt_belief
            }
        }
        
        json_path = os.path.join(self.OUTPUT_DIR, "graph.json")
        with open(json_path, "w") as f:
            json.dump(data, f)
            
        # Load
        report = chitta.load(json_path)
        print(f"Load Report: {report}")
        
        self.assertEqual(report["loaded"], 1)
        self.assertEqual(report["errors"], 1)
        print("✓ Honest Loading verified.")

    def test_belief_revision(self):
        print("\n--- Test 3: Belief Revision (Supersession) ---")
        engine = Ahankara(persistence_dir=self.OUTPUT_DIR)
        
        # 1. Teach Default: "Birds can fly."
        print("> Teaching: Birds can fly.")
        engine.process("Birds can fly.")
        
        # Verify it's active
        beliefs = list(engine.chitta.beliefs.values())
        print(f"Beliefs count: {len(beliefs)}")
        for b in beliefs:
            print(f" - {b.statement_text} (Preds: {b.predicates}, State: {b.epistemic_state})")
            
        b_default = [b for b in beliefs if "can_fly" in b.predicates][0]
        self.assertTrue(b_default.active)
        self.assertEqual(str(b_default.epistemic_state), "DEFAULT")
        
        # 2. Teach Exception: "Birds cannot fly." (Contradiction)
        print("> Teaching: Birds cannot fly.")
        engine.process("Birds cannot fly.")
        
        # Verify Default is inactive
        self.assertFalse(b_default.active)
        print(f"Old belief active: {b_default.active}")
        
        # Verify Exception is active
        b_except = [b for b in engine.chitta.beliefs.values() if b.active and "can_fly" in b.predicates][0]
        self.assertEqual(str(b_except.epistemic_state), "EXCEPTION")
        self.assertEqual(b_except.polarity_value, -1)
        
        print("✓ Belief Revision verified.")

if __name__ == "__main__":
    unittest.main()
