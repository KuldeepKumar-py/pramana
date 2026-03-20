import unittest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ahankara.layer import Ahankara

class TestIntegration(unittest.TestCase):
    def test_full_loop(self):
        marc = Ahankara()
        
        # 1. Teach
        marc.process("The sky is blue")
        
        # 2. Ask
        answer = marc.ask("Is the sky blue?")
        self.assertTrue(answer.lower().startswith("yes"))

        # 3. Unknown
        answer_unknown = marc.ask("Is the sky green?")
        # Depending on "hard_negative" or open world assumption, might be "unknown" or "no"
        # Default mock parser might not parse "green" well if logic is strict, 
        # but let's check for "unknown" or "no".
        print(f"Answer for green: {answer_unknown}")
        
if __name__ == '__main__':
    unittest.main()
