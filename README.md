# Pramāṇa-Constrained Inference Engine

A Python-based inference engine that enforces epistemic justification based on **Nyāya philosophy** — one of the six classical schools of Indian logic.

## Core Technology Stack

| Concern | Library |
|---|---|
| Input validation | `pydantic` v2 |
| Relational logic / syllogism | `kanren` (miniKanren) |
| Analogical similarity | `numpy` |
| Reasoning DAG | `networkx` |
| Testing | `pytest` |

---

## Package Structure

```
pramana_engine/
├── __init__.py         – public API
├── models.py           – Pydantic data models (Proposition, Evidence)
├── ingestion.py        – input validation & normalisation
├── anumana.py          – kanren relational logic + 5-step Nyāya syllogism
├── upamana.py          – numpy cosine/jaccard similarity matrices
├── arthapatti.py       – abductive reasoning / missing-premise engine
├── inference.py        – pramāṇa checker + modus ponens + engine
├── priority_gate.py    – Pratyakṣa-override conflict resolution
├── verdict.py          – verdict classifier (valid/Asiddha/Satpratipakṣa/Bādhita)
├── trace.py            – networkx DAG reasoning trace
├── engine.py           – top-level PramanaEngine orchestrator
└── cli.py              – CLI entrypoint
tests/
└── test_pramana_engine.py
```

---

## Installation

### Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.10 |
| pip | 22.0 |
| Git | any recent version |

### Step 1 – Clone the repository

```bash
git clone https://github.com/KuldeepKumar-py/pramana.git
cd pramana
```

### Step 2 – Create and activate a virtual environment

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3 – Install the package

**Editable install with development tools** (recommended for contributors and local testing):

```bash
pip install -e ".[dev]"
```

**Runtime-only install** (no test dependencies):

```bash
pip install -e .
```

### Step 4 – Verify the installation

```bash
# The CLI should print help and exit cleanly
pramana-engine --help

# Quick Python smoke-test
python - <<'EOF'
from pramana_engine.engine import PramanaEngine
engine = PramanaEngine()
out = engine.run(
    proposition_data={
        "claim": "test claim",
        "source": "test",
        "pramana_type": "Pratyakṣa",
        "confidence": 0.9,
        "timestamp": "2025-01-01T00:00:00+00:00",
    }
)
print("Verdict:", out.verdict.label.value)
print("Elapsed (ms):", round(out.elapsed_time_ms, 2))
EOF
```

Expected output:

```
Verdict: valid
Elapsed (ms): <some positive number>
```

---

## CLI Usage

```bash
pramana-engine --input input.json
# or pipe stdin
cat input.json | pramana-engine
```

### Example Input (`input.json`)

```json
{
  "proposition": {
    "claim": "hill has fire",
    "source": "observer",
    "pramana_type": "Anumāna",
    "confidence": 0.85,
    "timestamp": "2025-01-01T12:00:00+00:00"
  },
  "evidence": [
    {
      "claim": "smoke observed on hill",
      "source": "sensor-1",
      "pramana_type": "Pratyakṣa",
      "confidence": 0.95,
      "timestamp": "2025-01-01T11:59:00+00:00"
    }
  ],
  "subject": "hill",
  "hetu": "smoke",
  "sadhya": "fire",
  "example_subject": "kitchen"
}
```

### Example Output

```json
{
  "verdict": "valid",
  "normalized": {
    "claim": "hill has fire",
    "pramana_type": "Anumāna",
    "confidence": 0.85,
    "verdict": "valid",
    "justification": "Proposition is epistemically justified and undefeated."
  },
  "reasoning_trace": {
    "nodes": [ ... ],
    "links": [ ... ]
  }
}
```

---

## Pramāṇa Types

| Sanskrit | Meaning |
|---|---|
| Pratyakṣa | Direct perception (highest authority) |
| Anumāna | Inference via vyāpti (invariable concomitance) |
| Upamāna | Analogy / comparison |
| Śabda | Verbal testimony / scripture |
| Arthāpatti | Postulation / abductive reasoning |

---

## Verdict States

| Verdict | Meaning |
|---|---|
| `valid` | Justified, undefeated |
| `unjustified (Asiddha)` | Hetu not established |
| `suspended (Satpratipakṣa)` | Equal counter-evidence exists |
| `rejected/bādhita` | Overridden by Pratyakṣa |

---

## Running Tests

```bash
pytest tests/
```

Covers ≥ 10 test cases including:
- Syllogism success / failure (Asiddha)
- Contradictory evidence (Satpratipakṣa)
- Pratyakṣa override (Bādhita)
- Upamāna cosine/jaccard similarity
- Arthāpatti missing-premise detection
- Full pipeline validation

---

## Python API

```python
from pramana_engine.engine import PramanaEngine

engine = PramanaEngine(confidence_threshold=0.5)
output = engine.run(
    proposition_data={
        "claim": "hill has fire",
        "source": "observer",
        "pramana_type": "Anumāna",
        "confidence": 0.85,
        "timestamp": "2025-01-01T12:00:00+00:00",
    },
    evidence_data=[...],
    hetu="smoke",
    sadhya="fire",
)
print(output.verdict.label)        # VerdictLabel.VALID
print(output.trace_json)           # NetworkX DAG as JSON
```
