import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chitta.graph import ChittaGraph
from buddhi.belief import Belief

class TestChitta(unittest.TestCase):
    def setUp(self):
        self.chitta = ChittaGraph()

    def test_add_and_retrieve(self):
        """Test basic storage"""
        # Correct Belief construction
        b = Belief(
            template="relation",
            canonical={"subject": "sky", "predicate": "is", "object": "blue"},
            confidence=1.0,
            statement_text="The sky is blue"
        )
        self.chitta.add_belief(b)
        
        # Use find_by_entity
        results = self.chitta.find_by_entity("sky")
        self.assertEqual(len(results), 1)
        # Note: ID is auto-generated unless manually set, but let's check content
        self.assertEqual(results[0].canonical['subject'], "sky")

if __name__ == '__main__':
    unittest.main()
