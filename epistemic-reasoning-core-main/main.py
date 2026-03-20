"""
Episteme CLI
============

Interactive command-line interface for the Episteme Reasoning Engine.

Commands:
  > teach <sentence>        : Add a belief to the knowledge base
  > ask <question>          : Query the knowledge base (plain answer)
  > trace <question>        : Query with detailed proof trace
  > hypothesize <question>  : Run a hypothetical query (sandbox)
  > status                  : Show system stats
  > quit                    : Exit
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from ahankara.layer import Ahankara
    from manas.layer import Manas
    from buddhi.layer import Buddhi
    from chitta.graph import ChittaGraph
except ImportError as e:
    print(f"❌ Error importing Episteme modules: {e}")
    sys.exit(1)

def print_header():
    print("\n" + "="*80)
    print("EPISTEME: Epistemic Reasoning Console")
    print("="*80)
    print("Type 'help' for commands.")

def print_help():
    print("\nCommands:")
    print("  teach <text>        : Store a fact (e.g., 'teach Bats are mammals')")
    print("  ask <text>          : Ask a question (e.g., 'ask Do bats fly?')")
    print("  trace <text>        : Ask with full reasoning log")
    print("  hypothesize <text>  : Run a 'what if' query (e.g. 'hypothesize Do X? if Y')")
    print("  status              : Show memory stats")
    print("  save                : Manually save memory to disk")
    print("  system              : Dump full belief graph state")
    print("  quit                : Exit")

def run_interactive():
    # Parse Arguments
    parser = argparse.ArgumentParser(description="Episteme Reasoning Engine")
    parser.add_argument("--db", type=str, help="Path to persistence directory", default=None)
    args = parser.parse_args()

    # Initialize Stack
    persistence_dir = args.db
    
    chitta = ChittaGraph()
    manas = Manas()     # Input Parser
    buddhi = Buddhi(chitta) # Reasoning Core
    # Ahankara Orchestrator
    engine = Ahankara(manas, buddhi, chitta, persistence_dir=persistence_dir)
    
    if persistence_dir:
        print(f"💾 Persistence enabled: {persistence_dir}")

    print_header()

    while True:
        try:
            raw = input("\n> ").strip()
            if not raw:
                continue
            
            parts = raw.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ["quit", "exit", "q"]:
                if engine.persistence_dir:
                    print("Saving state before exit...")
                    engine._persist_state()
                print("Exiting.")
                break
            
            elif cmd == "help":
                print_help()
            
            elif cmd == "status":
                print(f"Beliefs stored: {len(chitta.beliefs)}")
                print(f"Entities known: {len(chitta.entity_index)}")

            elif cmd == "system":
                print("\n🧠 SYSTEM DUMP")
                print("="*60)
                print(f"Total Beliefs: {len(chitta.beliefs)}")
                print("-" * 60)
                for b in chitta.beliefs.values():
                     if not b.active: continue
                     pol_icon = "+" if str(b.polarity) == "POSITIVE" else "-"
                     print(f"[{b.id[:8]}] {pol_icon} {b.statement_text}")
                     print(f"    Confidence: {b.confidence:.2f} | Epistemic: {b.epistemic_state}")
                     print(f"    Predicates: {b.predicates}")
                     print(f"    Entities: {b.entities}")
                     print(f"    Provenance: {[str(p.op) for p in b.provenance]}")
                     print("")
                print("="*60)

            elif cmd == "teach":
                if not arg:
                    print("Usage: teach <sentence>")
                    continue
                result = engine.process(arg)
                print(f"✓ Learned: {arg}")

            elif cmd == "save":
                if not engine.persistence_dir:
                    print("Persistence not enabled. Run with --db <path> to enable.")
                else:
                    engine._persist_state()
                    print(f"💾 Saved state to {engine.persistence_dir}")
            
            elif cmd == "ask":
                if not arg:
                    print("Usage: ask <question>")
                    continue
                # Simple ask
                ans = engine.ask(arg)
                print(f"Answer: {ans}")

            elif cmd == "trace":
                if not arg:
                    print("Usage: trace <question>")
                    continue
                
                # Manual pipeline execution to capture steps
                # 1. Parse
                parsed = manas.parse(arg)
                print(f"[Trace] Parsed: {parsed}")
                
                # 2. Reason
                answer_obj = buddhi.answer(parsed)
                
                # 3. Print Trace
                print("\n🔍 REASONING TRACE")
                print("────────────────" * 4)
                
                for i, step in enumerate(answer_obj.steps, 1):
                    icon = "  "
                    if "check" in step.rule: icon = "✅"
                    if "fail" in step.rule or "no_" in step.rule or "not_" in step.rule or "blocking" in step.output.lower(): icon = "❌"
                    if "focus" in step.rule: icon = "👁️"
                    if "match" in step.rule: icon = "🎯"
                    
                    # Format: Step X: [Rule] Output
                    print(f"Step {i}: {icon} [{step.rule}] {step.output}")
                
                if answer_obj.conflicts:
                    print("\n⚔️ CONFLICTS RESOLVED")
                    for c in answer_obj.conflicts:
                        print(f"  - {c.resolution.upper()}: {c.positive} vs {c.negative} (Delta: {c.delta:.2f})")

                print("────────────────" * 4)
                print(f"🏁 FINAL VERDICT: {str(answer_obj.verdict).upper()}")
                
                if str(answer_obj.verdict).upper() == "UNKNOWN":
                    # Extract the reason from the last negative step
                    failure_reasons = [
                        s.output for s in answer_obj.steps 
                        if s.confidence == 0.0 or "not" in s.rule or "no_" in s.rule
                    ]
                    if failure_reasons:
                        print(f"❓ REASON: {failure_reasons[-1]}")

            elif cmd == "hypothesize":
                if not arg:
                    print("Usage: hypothesize <question> [if <assumption>]")
                    continue
                
                print(f"🔮 Entering Hypothetical Sandbox...")
                if " if " in arg:
                    # simplistic parsing: "do X if Y"
                    parts = arg.split(" if ", 1)
                    q_part = parts[0].strip()
                    a_part = parts[1].strip()
                    print(f"   Assumption: {a_part}")
                    print(f"   Question: {q_part}")
                    result = engine.ask_hypothetically(q_part, assumptions=[a_part])
                else:
                    result = engine.ask_hypothetically(arg)
                
                print(f"\n💡 RESULT: {result}\n")

            else:
                print(f"Unknown command: {cmd}")

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_interactive()
