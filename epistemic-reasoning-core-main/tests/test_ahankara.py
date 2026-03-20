import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ahankara.layer import Ahankara
from manas.layer import Manas
from buddhi.layer import Buddhi
from chitta.graph import ChittaGraph

class TestAhankara(unittest.TestCase):
    def setUp(self):
        self.chitta = ChittaGraph()
        self.manas = Manas()
        self.buddhi = Buddhi(self.chitta)
        self.ahankara = Ahankara(self.manas, self.buddhi, self.chitta)

    def test_process_flow(self):
        """Test the teach-process loop"""
        self.ahankara.process("Cats are mammals.")
        # Verify it reached Chitta
        beliefs = self.chitta.find_by_entity("cat")
        self.assertTrue(len(beliefs) > 0)

    def test_ask_flow(self):
        """Test the ask loop"""
        self.ahankara.process("dogs bark")
        answer = self.ahankara.ask("do dogs bark")
        self.assertIn("Yes", answer)

if __name__ == '__main__':
    unittest.main()
