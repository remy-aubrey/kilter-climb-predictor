# Kilter Climb Predictor — Design & Implementation Plan

## Current Status (October 2026)

**Working**: GPT-style transformer with rule-based masking generates realistic climbs.

**What's done:**
- Data pipeline (download, preprocess, encode)
- GPT-style transformer model with causal mask
- Rule-based masking during sampling
- CLI with ASCII visualization
- Training pipeline with checkpointing

**What's next:**
- Train for more epochs (20-50) on full dataset
- Tune hyperparameters (temperature, top_k, max_reach)
- Add evaluation metrics
- Add unit tests

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Kilter DB  │────▶│  Data Layer  │────▶│  GPT Model  │────▶│   CLI    │
│  (SQLite)   │     │  (Pandas)    │     │  (PyTorch)  │     │  (Typer) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
```

## Data Source

- **Library**: [`boardlib`](https://github.com/lemeryfertitta/BoardLib)
- **Command**: `boardlib database kilter data/kilter.db`
- **Contents**: ~344k climbs with hold UUIDs, LED positions, grades, angles
- **Format**: SQLite with tables for climbs, holds, frames, and LED mappings

## ML Approach: GPT-Style Transformer

**Current model**: Decoder-only transformer with causal mask

**Why GPT over LSTM/Markov:**
- LSTM: Mode collapse (predicts "middle" 100% of time)
- Markov: Simple but struggles with long-range dependencies
- GPT: Handles sequential data, long-range dependencies, can be conditioned on grade

**How it works:**
1. Each climb is a sequence of holds: `[GRADE] start start middle ... finish feet... [END]`
2. Each hold is represented as a token (hold_id, hold_type) pair
3. The model learns P(next_hold | sequence_so_far, grade)
4. At generation time, we sample from this distribution with rule-based masking

**Rule-based masking:**
- First holds: 1-2 start holds in bottom third
- Hand moves: must go up or level, within reach limit (140cm)
- Finish: only near top (y > 242), after minimum hand holds (5)
- After finish: only feet or END
- No repeats
- END: only after finish

## Project Structure

```
kilter-climb-predictor/
├── src/
│   ├── data/
│   │   ├── download.py      # Download DB via boardlib
│   │   ├── schema.py        # DB schema exploration
│   │   └── preprocess.py    # Clean and transform raw data
│   ├── model/
│   │   ├── gpt.py           # GPT-style transformer (current)
│   │   ├── train_gpt.py     # Training script for GPT
│   │   ├── markov.py        # Markov chain (deprecated)
│   │   └── transformer.py   # Original transformer (deprecated)
│   ├── cli/
│   │   ├── main.py          # Typer CLI app
│   │   └── visualize.py     # ASCII climb display
│   └── utils/
│       └── constants.py     # Grade mappings, hold types, constraints
├── tests/                   # Unit tests
├── data/                     # Downloaded databases (gitignored)
├── models/                   # Saved model weights (gitignored)
└── outputs/                  # Generated climbs (gitignored)
```

## Implementation Tasks

### Task 1: Project Setup & Data Download ✅
- Project structure, `pyproject.toml`, `.gitignore`
- `src/data/download.py` — wrapper around boardlib
- `src/data/schema.py` — explore and document the SQLite schema

### Task 2: Data Preprocessing ✅
- `src/data/preprocess.py` — parse frames, encode features, create sequences
- Filter to 40° angle, 3-30 holds per climb
- Encode hold types, LED colors, grades
- Stratified split by grade (80/10/10)

### Task 3: Model Definition ✅
- `src/model/gpt.py` — GPT-style transformer with causal mask
- `src/model/train_gpt.py` — training script with checkpointing

### Task 4: Training Pipeline ✅
- Training loop with AdamW optimizer
- Checkpoint saving every epoch
- Progress logging to file
- Early stopping based on validation loss

### Task 5: Generation & Inference ✅
- `src/model/gpt.py` — `generate_climb()` with rule-based masking
- `generate_valid_climb()` — keeps sampling until valid climb
- Temperature and top-k sampling

### Task 6: CLI Interface ✅
- `src/cli/main.py` — Typer app with `generate`, `train`, `download`, `stats`, `visualize`
- `src/cli/visualize.py` — ASCII art board visualization

### Task 7: Testing & Documentation 🔄
- Unit tests for GPT model
- README with installation, usage, examples
- HISTORY.md with learnings

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Data download | `boardlib` | Purpose-built, maintained, handles auth |
| Data processing | `pandas` + `numpy` | Industry standard, beginner-friendly |
| ML framework | `PyTorch` | Most beginner-friendly DL framework |
| Model | GPT-style transformer | Handles sequential data, long-range dependencies |
| CLI | `typer` + `rich` | Modern, type-hinted, pretty output |
| Visualization | ASCII + `rich` colors | No GUI dependencies, works in terminal |
| Testing | `pytest` | Standard Python testing |

## Key Learnings

1. **LSTM mode collapse**: Predicts most common class 100% of time
2. **Markov chain**: Simple but effective with proper constraints
3. **Transformer overfitting**: Near-zero loss = peeking at answer
4. **Rule-based masking**: Most effective way to ensure valid output
5. **Class weights**: Help with imbalanced data but don't solve mode collapse

See [HISTORY.md](HISTORY.md) for detailed history.

## Future Enhancements

- Web UI for visualizing climbs on a board diagram
- Export climbs to Kilter app format
- User feedback loop (rate generated climbs, fine-tune)
- Multi-board support (Tension, Decoy, etc.)
- Hold reachability constraints (max distance between consecutive holds)
- Grade-appropriate hold type distributions
