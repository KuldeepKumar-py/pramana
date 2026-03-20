import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manas.layer import Manas

class TestManas(unittest.TestCase):
    def setUp(self):
        self.manas = Manas(llm_backend="mock")

    def test_parse_is_a(self):
        text = "Penguins are birds"
        result = self.manas.parse(text)
        self.assertEqual(result['template'], 'is_a')
        # Check canonical structure (stable) rather than list order (unstable due to set conversion)
        self.assertEqual(result['canonical']['subject'], 'penguin')
        self.assertEqual(result['canonical']['object'], 'bird')

    def test_parse_capability_negative(self):
        text = "Penguins cannot fly"
        result = self.manas.parse(text)
        self.assertEqual(result['template'], 'relation')
        # Check if predicate is either 'can_fly' or normalized 'behavior_flies'
        # The key is consistency.
        preds = result['predicates']
        self.assertTrue('can_fly' in preds or 'behavior_flies' in preds, f"Predicates: {preds}")
        self.assertEqual(result['polarity'], -1)

    def test_parse_query(self):
        text = "Do penguins fly?"
        result = self.manas.parse(text)
        self.assertEqual(result['intent'], 'query')

if __name__ == '__main__':
    unittest.main()
