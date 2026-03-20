# Episteme Test Suite

Verified using pure Python `unittest`. No external dependencies.

## Structure

```
tests/
├── test_manas.py                 # Unit tests for Input/Parsing layer
├── test_buddhi.py                # Unit tests for Reasoning/Judgment layer
├── test_overall.py               # Integration test for full loop
├── benchmark_strict.py           # Legacy strict benchmark (integration check)
├── run_scientific_benchmark.py   # NEW Research-Grade Benchmark Runner
├── scientific_benchmark_generator.py # Generator for scientific cases
├── data/
│   ├── benchmark_strict.json     # Data for strict benchmark
│   └── benchmark_scientific.json # Generated data for scientific benchmark
└── logs/                         # Execution logs
```

## Running Tests

### 1. Standard Unit Tests
Run the core test suite to verify module functionality.

```bash
python -m unittest discover tests
```

### 2. Scientific Benchmark (Research-Grade)
Run the deep reasoning stress test to generate proof traces.

```bash
python tests/run_scientific_benchmark.py
```
Check `tests/logs/scientific_benchmark_execution.log` for the detailed output.

### 3. Strict Benchmark (Integration)
Run the broader integration verification.

```bash
python tests/benchmark_strict.py
```
