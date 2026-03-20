"""
Scientific Benchmark Generator for Episteme
===========================================

Generates rigorous logic puzzles to stress-test the reasoning engine.
Unlike the list-based benchmarks, this builds STRUCTURAL logic problems.

Scenarios:
1. Deep Taxonomy Chains (Is-A transitivity)
2. Property Inheritance (Downwards propagation)
3. Negation Blocking (Exceptions to rules)
4. Transitivity Chains (A > B > C > D)
5. Modus Ponens/Tollens checks

Output: `tests/data/benchmark_scientific.json`
"""

import json
import random
from dataclasses import dataclass, asdict
from typing import List, Literal, Optional

@dataclass
class ScientificTestItem:
    id: str
    scenario: str              # "taxonomy_depth", "negation_blocking", etc.
    setup_facts: List[str]     # Facts to teach before querying
    query: str                 # The question to ask
    expected_verdict: str      # "positive", "negative", "refused", "unknown"
    difficulty: int            # 1-10 scale
    description: str           # Human-readable explanation of the logic
    tags: List[str]            # Metadata tags

class ScientificBenchmarkGenerator:
    def __init__(self):
        self.items: List[ScientificTestItem] = []
        self._id_counter = 1

    def _next_id(self) -> str:
        tid = f"SCI-{self._id_counter:04d}"
        self._id_counter += 1
        return tid

    def save(self, filepath: str):
        data = {
            "metadata": {
                "version": "1.0",
                "generator": "ScientificBenchmarkGenerator",
                "count": len(self.items)
            },
            "cases": [asdict(item) for item in self.items]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(self.items)} scientific test cases to {filepath}")

    # =========================================================================
    # SCENARIO 1: Deep Taxonomy Chains
    # =========================================================================
    def generate_taxonomy_chains(self):
        """
        Tests if the system can infer Is-A relationships across long chains.
        A -> B -> C -> D -> E. Is A an E?
        """
        depths = [3, 5, 8, 10]
        
        for depth in depths:
            # entities: e.g. ["sub_species", "species", "genus", "family", ...]
            chain = [f"Entity_L{i}" for i in range(depth)]
            
            facts = []
            for i in range(len(chain) - 1):
                # "Entity_L0 is a Entity_L1"
                facts.append(f"{chain[i]} is a {chain[i+1]}.")
            
            # Positive Check: Is L0 an L{depth-1}?
            self.items.append(ScientificTestItem(
                id=self._next_id(),
                scenario="taxonomy_chain",
                setup_facts=facts,
                query=f"Is {chain[0]} a {chain[-1]}?",
                expected_verdict="POSITIVE",
                difficulty=depth,
                description=f"Transitive inference across {depth} levels of taxonomy.",
                tags=["taxonomy", "transitivity", f"depth_{depth}"]
            ))

            # Negative Check: Is L{depth-1} an L0? (Asymmetry)
            self.items.append(ScientificTestItem(
                id=self._next_id(),
                scenario="taxonomy_asymmetry",
                setup_facts=facts,
                query=f"Is {chain[-1]} a {chain[0]}?",
                expected_verdict="UNKNOWN", # Or No, depending on strictness. Usually Unknown unless closed world.
                difficulty=depth,
                description=f"Checking asymmetry of Is-A relationship.",
                tags=["taxonomy", "asymmetry"]
            ))

    # =========================================================================
    # SCENARIO 2: Property Inheritance
    # =========================================================================
    def generate_inheritance(self):
        """
        Tests if properties assigned to a high-level class propagate down.
        Class A has property P. B is A. C is B. Does C have property P?
        """
        # "Specific -> ... -> General"
        chain = ["Chihuahua", "Dog", "Canine", "Mammal", "Animal"]
        
        facts = [
            f"{chain[i]} is a {chain[i+1]}." for i in range(len(chain)-1)
        ]
        
        # Define property at the TOP
        facts.append("Animals breathe oxygen.")
        
        # Test inheritance at bottom
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="property_inheritance",
            setup_facts=facts,
            query="Do chihuahuas breathe oxygen?",
            expected_verdict="POSITIVE",
            difficulty=len(chain),
            description="Inheritance of 'breathe oxygen' from Animal to Chihuahua.",
            tags=["inheritance", "biology"]
        ))
        
        # Define property at MIDDLE
        facts_mid = [f"{chain[i]} is a {chain[i+1]}." for i in range(len(chain)-1)]
        facts_mid.append("Dogs bark.")
        
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="mid_level_inheritance",
            setup_facts=facts_mid,
            query="Does a chihuahua bark?",
            expected_verdict="POSITIVE",
            difficulty=3,
            description="Inheritance from mid-level class.",
            tags=["inheritance"]
        ))
        
        # Test non-upward propagation
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="upward_propagation_fail",
            setup_facts=facts_mid,
            query="Do all animals bark?",
            expected_verdict="UNKNOWN", # Or REFUSED
            difficulty=3,
            description="Properties should not propagate upwards implicitly.",
            tags=["inheritance", "safety"]
        ))

    # =========================================================================
    # SCENARIO 3: Negation Blocking (Exceptions)
    # =========================================================================
    def generate_negation_blocking(self):
        """
        The classic Tweety/Penguin problem.
        Birds fly. Penguins are birds. Penguins do NOT fly. 
        Does a penguin fly?
        """
        
        # 1. Direct Block
        facts_1 = [
            "Birds can fly.",
            "Penguins are birds.",
            "Penguins cannot fly."
        ]
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="negation_blocking",
            setup_facts=facts_1,
            query="Can penguins fly?",
            expected_verdict="NEGATIVE",
            difficulty=5,
            description="Specific negation 'cannot fly' should block inherited 'can fly'.",
            tags=["negation", "conflict_resolution"]
        ))
        
        # 2. Distance Block (Inheritance path blocked)
        # Animal -> moves
        # Sponge is Animal.
        # Sponge cannot move.
        facts_2 = [
            "Animals can move.",
            "Sponges are animals.",
            "Sponges cannot move."
        ]
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="negation_blocking_2",
            setup_facts=facts_2,
            query="Can sponges move?",
            expected_verdict="NEGATIVE",
            difficulty=5,
            description="Specific negation blocks general capability.",
            tags=["negation"]
        ))

    # =========================================================================
    # SCENARIO 4: Transitive Relations (A > B > C)
    # =========================================================================
    def generate_transitivity(self):
        """
        Temporal or Comparative transitivity.
        A is larger than B. B is larger than C. Is A larger than C?
        """
        chain = ["Planet", "Moon", "Asteroid", "Pebble"]
        
        facts = []
        for i in range(len(chain)-1):
            facts.append(f"{chain[i]} is larger than {chain[i+1]}.")
            
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="comparative_transitivity",
            setup_facts=facts,
            query="Is a Planet larger than a Pebble?",
            expected_verdict="POSITIVE",
            difficulty=3,
            description="Transitive inference of 'larger than' relation.",
            tags=["transitivity", "spatial"]
        ))

    # =========================================================================
    # SCENARIO 5: Logical Safety & Hallucination Check
    # =========================================================================
    def generate_safety_checks(self):
        """
        Queries about things that were NEVER taught.
        System must say UNKNOWN, not hallucinate.
        """
        facts = ["Dogs are mammals."]
        
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="hallucination_check",
            setup_facts=facts,
            query="Do ghosts exist?",
            expected_verdict="UNKNOWN", # Or REFUSED
            difficulty=1,
            description="Query about untaught entity.",
            tags=["safety"]
        ))
        
        self.items.append(ScientificTestItem(
            id=self._next_id(),
            scenario="predicate_hallucination",
            setup_facts=facts,
            query="Do dogs play poker?",
            expected_verdict="UNKNOWN",
            difficulty=1,
            description="Query about untaught predicate for known entity.",
            tags=["safety"]
        ))

    def generate_all(self):
        self.generate_taxonomy_chains()
        self.generate_inheritance()
        self.generate_negation_blocking()
        self.generate_transitivity()
        self.generate_safety_checks()


if __name__ == "__main__":
    import os
    
    gen = ScientificBenchmarkGenerator()
    gen.generate_all()
    
    # Save to tests/data/benchmark_scientific.json
    output_path = os.path.join(os.path.dirname(__file__), "data", "benchmark_scientific.json")
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    gen.save(output_path)
