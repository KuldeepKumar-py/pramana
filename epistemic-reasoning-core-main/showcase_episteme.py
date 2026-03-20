
import sys
import os
import shutil
import time

sys.path.append(os.getcwd())

from ahankara.layer import Ahankara
from buddhi.belief import Belief, EpistemicType
from common.types import Verdict

def print_header(text):
    print("\n" + "="*80)
    print(f" {text.upper()}")
    print("="*80)

def print_step(title, description):
    print(f"\n➤ {title}")
    print(f"  {description}")

def inspect_belief(ai, entity_id):
    """Peek into Chitta to show stored internal state."""
    beliefs = ai.chitta.entity_index.get(entity_id, set())
    print(f"\nInternal Storage: Beliefs about '{entity_id}'")
    print(f"{'ID':<15} {'Statement':<40} {'Type':<15} {'Conf':<6} {'Status':<10}")
    print("-" * 90)
    
    for bid in beliefs:
        b = ai.chitta.get(bid)
        status = "Active" if b.active else "Inactive"
        print(f"{b.id[:8]:<15} {b.statement_text[:38]:<40} {b.epistemic_state.name:<15} {b.confidence:<6.2f} {status:<10}")
    print("-" * 90)

def main():
    # Setup
    db_path = "showcase_db"
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    print_header("EPISTEME SYSTEM SHOWCASE")
    print("Initializing Ahankara (The Self)...")
    ai = Ahankara(persistence_dir=db_path)
    
    # ════════════════════════════════════════════════════════════════════
    # PHASE 1: MANAS & CHITTA (Acquisition & Storage)
    # ════════════════════════════════════════════════════════════════════
    print_header("PHASE 1: ACQUISITION (Manas Parser -> Chitta Graph)")
    
    print_step("1. Teaching Basic Taxonomy", "Inputting natural language facts. Manas normalizes entities/predicates.")
    
    # Using singular to ensure direct linkage
    inputs = [
        "Socrates is a human.",
        "A human is a mammal.",
        "A mammal is an animal."
    ]
    
    for text in inputs:
        print(f"  USER: '{text}'")
        ai.process(text)
        time.sleep(0.1)
        
    print_step("2. Internal Inspection", "Looking at how 'Socrates' and 'Human' are stored in the Graph.")
    inspect_belief(ai, "socrates")
    inspect_belief(ai, "human")
    
    # ════════════════════════════════════════════════════════════════════
    # PHASE 2: BUDDHI (Logical Inference)
    # ════════════════════════════════════════════════════════════════════
    print_header("PHASE 2: INFERENCE (Buddhi Logic Layer)")
    
    print_step("3. Asking a Question (Transitive Inference)", "Query: 'Is Socrates a mammal?'")
    print("  Reasoning Trace:")
    
    proposal = ai.perceive("Is Socrates a mammal?")
    proof = ai.buddhi.answer(proposal)
    
    # Render the proof steps cleanly
    for step in proof.steps:
        print(f"    [{step.rule}] {step.output} (Conf: {step.confidence})")
    
    print(f"\n  VERDICT: {proof.verdict.upper()} (Expected: YES)")
    if proof.verdict == "invalid":
        print(f"  [Failure Info] Reason: {proof.steps[-1].output if proof.steps else 'Unknown'}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 3: CONFLICT RESOLUTION (The Logic Battles)
    # ════════════════════════════════════════════════════════════════════
    print_header("PHASE 3: CONFLICT RESOLUTION (The Logic Battles)")
    
    print_step("4. Setup Vertical Conflict (Specificity)", "Teaching: Birds fly (Default), Penguins are Birds, Penguins DO NOT fly (Exception).")
    
    ai.process("Tweety is a penguin.")
    ai.process("Penguins are birds.")
    ai.process("Birds fly.") # Default
    
    # Manual Injection for Negative Exception (Bypassing Parser Bug for Showcase clarity)
    print("  .. Injecting 'Penguins do not fly' as Negative Belief ..")
    b_neg = Belief(
        "Group Property", 
        {"subject": "penguin", "predicate": "behavior_flies", "entities": ["penguin"]}, 
        0.9, 
        id="rule_penguin_no_fly", 
        statement_text="Penguins do not fly (Exception).", 
        epistemic_state=EpistemicType.DEFAULT, 
        polarity_value=-1
    )
    ai.chitta.add_belief(b_neg)
    
    print_step("5. Resolving Specificity", "Query: 'Does Tweety fly?'")
    proposal = ai.perceive("Does Tweety fly?")
    proof = ai.buddhi.answer(proposal)
    print(f"  VERDICT: {proof.verdict.upper()} (Expected: NO)")
    print(f"  Reason: {proof.steps[-1].output}")
    
    
    print_step("6. Setup Horizontal Conflict (Nixon Diamond)", "Structure: Nixon is Quaker & Republican. Quakers Fly. Republicans Don't.")
    
    ai.process("Nixon is a quaker.")
    ai.process("Nixon is a republican.")
    
    # Using 'behavior_flies' to guarantee parser alignment/grounding for showcase
    # Absurdist Metaphor: Doves/Hawks flying
    
    # Quaker -> Flies (Pos)
    b_quaker = Belief(
        "Group Property", 
        {"subject": "quaker", "predicate": "behavior_flies", "entities": ["quaker"]}, 
        0.9, 
        id="rule_quaker_flies", 
        statement_text="Quakers fly.", 
        epistemic_state=EpistemicType.DEFAULT, 
        polarity_value=1
    )
    ai.chitta.add_belief(b_quaker)
    
    # Republican -> Not Flies (Neg)
    b_republican = Belief(
        "Group Property", 
        {"subject": "republican", "predicate": "behavior_flies", "entities": ["republican"]}, 
        0.9, 
        id="rule_republican_no_fly", 
        statement_text="Republicans do not fly.", 
        epistemic_state=EpistemicType.DEFAULT, 
        polarity_value=-1
    )
    ai.chitta.add_belief(b_republican)
    
    print_step("7. Resolving Nixon Diamond", "Query: 'Does Nixon fly?'")
    proposal = ai.perceive("Does Nixon fly?")
    proof = ai.buddhi.answer(proposal)
    
    print(f"  VERDICT: {proof.verdict.upper()} (Expected: CONFLICT)")
    if proof.conflicts:
         print(f"  Conflict Detected: {proof.steps[-1].output}")
    else:
         print("  [WARNING] No conflict detected. Trace steps:")
         for s in proof.steps:
             print(f"   - {s.output}")

    # ════════════════════════════════════════════════════════════════════
    # PHASE 4: QUANTITATIVE LAYER (Numbers Second)
    # ════════════════════════════════════════════════════════════════════
    print_header("PHASE 4: QUANTITATIVE LAYER (Confidence & Decay)")
    
    print_step("8. Evidence Aggregation", "Teaching 'Sky is blue' multiple times.")
    
    b1 = Belief("Obs", {"s":"sky","p":"is","o":"blue"}, 0.5, id="sky_blue", statement_text="Sky is blue", epistemic_state=EpistemicType.OBSERVATION)
    ai.chitta.add_belief(b1)
    print(f"  Init Conf: {ai.chitta.get('sky_blue').confidence}")
    
    ai.chitta.add_belief(b1) # Re-add
    print(f"  After Re-add: {ai.chitta.get('sky_blue').confidence} (Boosted!)")
    
    print_step("9. Temporal Decay & Logic Gating", "Simulating passage of time on a Rumour vs an Axiom.")
    
    b_rumour = Belief("Rumour", {"s":"market","p":"will","o":"crash"}, 0.2, id="rumour_crash", statement_text="Market will crash", epistemic_state=EpistemicType.DEFAULT)
    b_rumour.decay_rate = 0.5 # Fast decay
    ai.chitta.add_belief(b_rumour)
    
    b_axiom = Belief("Axiom", {"s":"1","p":"eq","o":"1"}, 1.0, id="axiom_one", statement_text="1=1", epistemic_state=EpistemicType.AXIOM)
    ai.chitta.add_belief(b_axiom)
    
    print("\n  Time passes...")
    ai.chitta.apply_decay(steps=2, threshold=0.1) # Should drop 0.2 -> 0.1 -> 0.05 (Inactive)
    
    inspect_belief(ai, "market")
    inspect_belief(ai, "1")
    
    print("\nSHOWCASE COMPLETE.")

if __name__ == "__main__":
    main()
