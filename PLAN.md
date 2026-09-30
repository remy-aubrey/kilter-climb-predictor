# Kilter Climb Predictor — Design & Implementation Plan

## Overview

A Python CLI tool that uses machine learning to generate new climbing routes for the Kilter Board. Users specify a grade (e.g., V5) and the tool outputs a new climb with hold positions and LED colors.

## Success Criteria

V1 is successful when:
1. The tool can generate 5 distinct climbs at a requested grade (e.g., V5)
2. Generated climbs pass reachability validation (see constraints below)
3. Generated climbs don't exactly duplicate any training data
4. Generated climbs use grade-appropriate hold types
5. The CLI is usable by a beginner with clear error messages

## Reachability Constraints

Based on user measurements (height: 157cm, wingspan: 157cm):

| Constraint | Value | Calculation |
|------------|-------|-------------|
| Max hand-to-hand distance | 140cm | 89% of wingspan (157cm) — full stretch, maintainable |
| Comfortable hand-to-hand | 100-120cm | 64-76% of wingspan — controlled pulling |
| Max foot-to-hand distance | 120cm | Beyond this, cannot maintain tension or press |
| Max foot-to-foot distance | 100cm | Feet should stay within reasonable base |

**Validation rules for generated climbs:**
- Consecutive hand holds must be ≤ 140cm apart
- Each foot hold must be ≤ 120cm from the nearest hand hold
- At least one foot hold must be ≤ 100cm from the nearest hand hold (for tension)
- These values are configurable in `src/utils/constants.py` for different body types

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Kilter DB  │────▶│  Data Layer  │────▶│  ML Model   │────▶│   CLI    │
│  (SQLite)   │     │  (Pandas)    │     │  (PyTorch)  │     │  (Typer) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
```

## Data Source

- **Library**: [`boardlib`](https://github.com/lemeryfertitta/BoardLib) (pip installable)
- **Command**: `boardlib database kilter kilter.db --username <user>`
- **Contents**: ~130k climbs with hold UUIDs, LED positions, grades, angles, setter info
- **Format**: SQLite with tables for climbs, holds, frames, and LED mappings

## ML Approach: Grade-Conditioned Sequence Generator

**Recommended model**: LSTM-based sequence generator (beginner-friendly, well-documented, effective for this data size)

**Why LSTM over GAN/Diffusion**:
- Much simpler to implement and debug
- Works well with ~130k training examples
- Naturally handles variable-length sequences
- Easy to condition on grade via embedding
- Large community support and tutorials

**How it works**:
1. Each climb is a sequence of holds: `[hold_1, hold_2, ..., hold_n]`
2. Each hold is represented by: `(hold_id, position_x, position_y, hold_type, led_color)`
3. The model learns P(hold_i | hold_1...hold_{i-1}, grade) — the probability of the next hold given the sequence so far and the target grade
4. At generation time, we sample from this distribution autoregressively

**Alternative simpler approach** (if LSTM proves too difficult):
- Markov chain on hold transitions, conditioned on grade bucket
- Much less code, still produces reasonable results
- Can be upgraded to LSTM later

## Project Structure

```
kilter-climb-predictor/
├── PLAN.md                  # This file
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py      # Download DB via boardlib
│   │   ├── schema.py        # DB schema exploration and constants
│   │   └── preprocess.py    # Clean and transform raw data
│   ├── model/
│   │   ├── __init__.py
│   │   ├── dataset.py       # PyTorch Dataset for climbs
│   │   ├── network.py       # LSTM model definition
│   │   ├── train.py         # Training loop
│   │   └── generate.py      # Inference / generation logic
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py          # Typer CLI app
│   │   └── visualize.py     # ASCII / text-based climb display
│   └── utils/
│       ├── __init__.py
│       └── constants.py     # Grade mappings, hold types, LED colors
├── tests/
│   ├── test_preprocess.py
│   ├── test_model.py
│   └── test_generate.py
├── .gitignore               # Python, data/, models/, outputs/, .venv/
├── LICENSE                  # MIT license
├── data/                    # Downloaded databases (gitignored)
├── models/                  # Saved model weights (gitignored)
└── outputs/                 # Generated climbs (gitignored)
```

## Time Estimates

| Task | Estimated Time |
|------|---------------|
| Task 1: Setup & Data Download | 2-4 hours |
| Task 2: Data Preprocessing | 3-5 hours |
| Task 3: Model Definition | 2-3 hours |
| Task 4: Training | 1-2 days (including tuning) |
| Task 5: Generation & Inference | 2-4 hours |
| Task 6: CLI Interface | 3-5 hours |
| Task 7: Testing & Documentation | 2-4 hours |
| **Total** | **3-5 days** |

**Milestones**:
- **M1** (After Task 2): Data is downloaded, preprocessed, and explored
- **M2** (After Task 4): Model is trained and saves checkpoints
- **M3** (After Task 6): End-to-end CLI works — can generate a climb from command line
- **M4** (After Task 7): Project is documented and tested

## Implementation Tasks

### Task 1: Project Setup & Data Download
**Goal**: Initialize the project, install dependencies, download the Kilter database.

**Steps**:
1. Create virtual environment and `pyproject.toml` with dependencies: `boardlib`, `pandas`, `numpy`, `torch`, `typer`, `rich`, `matplotlib`
2. Create project directory structure
3. Create `.gitignore` (Python boilerplate, `data/`, `models/`, `outputs/`, `.venv/`)
4. Write `src/data/download.py` — wrapper around `boardlib database kilter` command
5. Write `src/data/schema.py` — explore and document the SQLite schema (tables, columns, relationships)
6. Test: download the DB and print table names + row counts

**Verification**: `python -m src.data.download` produces `data/kilter.db` with expected tables.

---

### Task 2: Data Preprocessing
**Goal**: Transform raw SQLite data into a clean, model-ready format.

**Steps**:
1. Write `src/data/preprocess.py`:
   - Load climbs table, join with holds and LED mappings
   - Filter to 40-degree angle (most common angle with the most data; multi-angle support is future work)
   - Filter out climbs with fewer than 3 holds or more than 30 holds
   - Map V-grades to integer indices (V0=0, V1=1, ..., V17=17)
   - Encode each hold as a feature vector: `(hold_id_index, x, y, hold_type, led_color)`
   - LED colors are categorical: `red`, `green`, `blue`, `yellow`, `purple`, `off` — use embedding, not RGB
   - Pad sequences to fixed length for batching
   - **Split strategy**: Stratified by grade to ensure all grades represented in train/val/test (80/10/10)
   - **Class imbalance**: Analyze grade distribution. For grades with < 100 climbs, either oversample or use weighted loss. Document expected performance per grade bucket.
2. Write `src/utils/constants.py` with grade mappings, hold type enums, LED color definitions
3. Test: print dataset statistics (num climbs per grade, avg sequence length, etc.)

**Verification**: Preprocessed data loads correctly, distributions look reasonable, all grades represented in all splits.

---

### Task 3: Model Definition & Dataset
**Goal**: Define the LSTM model and PyTorch Dataset.

**Steps**:
1. Write `src/model/dataset.py`:
   - `ClimbDataset` class returning `(sequence, grade, length)` tuples
   - Collate function for padding batches
2. Write `src/model/network.py`:
   - `ClimbGenerator(nn.Module)`:
     - Embedding layer for hold IDs
     - Grade embedding layer
     - 2-layer LSTM (hidden size 256)
     - Linear output layer predicting next hold
   - Forward pass: concatenate hold embedding + grade embedding → LSTM → linear → logits
3. Test: forward pass on a batch produces correct output shape

**Verification**: Model summary prints, forward pass works, loss decreases on a tiny batch.

---

### Task 4: Training Pipeline
**Goal**: Train the model on the preprocessed data.

**Steps**:
1. Write `src/model/train.py`:
   - **Reproducibility**: Set `torch.manual_seed(42)`, `numpy.random.seed(42)`, `random.seed(42)` at start
   - Training loop with Adam optimizer, cross-entropy loss
   - Grade-conditioned: grade embedding concatenated at each step
   - Early stopping based on validation loss (patience=5)
   - Save best model checkpoint to `models/climb_generator.pt`
   - Print training progress (loss, perplexity) using `rich`
   - Support for CPU and GPU (if available)
   - **Hyperparameter sweep**: Try learning rate ∈ {1e-3, 1e-4}, hidden size ∈ {128, 256}, dropout ∈ {0.0, 0.2}. Use a simple grid search on a subset of data, then train final model with best config.
   - **Experiment tracking**: Log each run's hyperparameters, seed, and final validation loss to `outputs/experiments.csv`
2. Test: train for 2 epochs on a small subset, verify loss decreases

**Verification**: Training completes, validation loss improves, checkpoint saved, experiment log written.

---

### Task 5: Generation & Inference
**Goal**: Generate new climbs from the trained model.

**Steps**:
1. Write `src/model/generate.py`:
   - Load trained model
   - Function `generate_climb(grade, temperature=1.0, max_holds=20)`:
     - Start with a random start hold (or user-specified)
     - Autoregressively sample next holds using temperature-scaled softmax
     - Stop at max_holds or when model predicts an "end" token
   - Function `generate_multiple(grade, n=5)` for variety
2. **Post-processing validation** (in `generate.py` or separate `validate.py`):
   - Consecutive hand holds must be ≤ 140cm apart (89% of 157cm wingspan)
   - Each foot hold must be ≤ 120cm from nearest hand hold
   - At least one foot hold ≤ 100cm from nearest hand hold (for tension)
   - Start holds should be in the lower third of the board
   - Finish holds should be in the upper third
   - At least 2 foot holds for climbs > 5 holds
   - Regenerate any climb that fails validation (max 10 retries)
3. **Evaluation metrics** (in `evaluate.py`):
   - **Diversity**: unique holds used across N generated climbs
   - **Validity**: % of generated holds that exist on the board
   - **Novelty**: % of generated climbs that don't exactly match training data
   - **Grade appropriateness**: compare hold difficulty distribution of generated vs. real climbs at that grade
4. Test: generate 5 V5 climbs, print hold sequences, run evaluation metrics

**Verification**: Generated climbs pass validation, evaluation metrics are reasonable, no exact duplicates of training data.

---

### Task 6: CLI Interface
**Goal**: Build the user-facing CLI tool.

**Steps**:
1. Write `src/cli/main.py` using Typer:
   - `kilter-gen generate --grade V5 --count 3` — generate climbs
   - `kilter-gen train --epochs 50` — train the model
   - `kilter-gen download` — download the database
   - `kilter-gen stats` — show dataset statistics
   - `kilter-gen visualize --grade V5` — show ASCII visualization
   - **Input validation**: Validate grade is in V0-V17, count is positive integer. Provide clear error messages.
2. Write `src/cli/visualize.py`:
   - ASCII art representation of the board with holds marked
   - Color-coded output using `rich` (green=start, yellow=foot, blue=hand, purple=finish)
3. Test: `python -m src.cli.main generate --grade V5 --count 1` produces output

**Verification**: All CLI commands work, output is readable and informative, invalid input produces clear errors.

---

### Task 7: Testing & Documentation
**Goal**: Ensure quality and usability.

**Steps**:
1. Write unit tests for preprocessing, model, and generation
2. Write `README.md` with:
   - Installation instructions
   - Usage examples
   - How the model works (simplified explanation)
   - Troubleshooting
3. Add `setup.py` for pip installation
4. Test: `pytest` passes, README instructions work from scratch

**Verification**: Clean install on fresh venv, all tests pass, CLI works.

---

## Dependency Graph

```
Task 1 (Setup) ──▶ Task 2 (Preprocess) ──┬──▶ Task 3 (Model) ──▶ Task 4 (Train) ──▶ Task 5 (Generate) ──▶ Task 6 (CLI) ──▶ Task 7 (Test/Docs)
                                        │
                                        └──▶ (data format agreed upon, Task 3 can start while Task 2 finishes)
```

Task 3 (model definition) can start once the data format from Task 2 is agreed upon, even before preprocessing is fully complete.

## Technology Choices Summary

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Data download | `boardlib` | Purpose-built, maintained, handles auth |
| Data processing | `pandas` + `numpy` | Industry standard, beginner-friendly |
| ML framework | `PyTorch` | Most beginner-friendly DL framework |
| Model | LSTM sequence generator | Simple, effective, well-documented |
| CLI | `typer` + `rich` | Modern, type-hinted, pretty output |
| Visualization | ASCII + `rich` colors | No GUI dependencies, works in terminal |
| Testing | `pytest` | Standard Python testing |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| BoardLib auth issues | Provide clear error messages; allow manual DB path |
| Model quality too poor | Start with Markov chain baseline; iterate to LSTM |
| Training too slow on CPU | Subset data for development; document GPU setup |
| Generated climbs are unrealistic | Add post-processing validation (hold reachability, grade-appropriate hold types) |
| Sequence too short/long | Add length conditioning or length tokens |
| Model too large for CPU inference | Use smaller hidden size (128); quantize model; document expected inference time (~1-5s per climb on CPU) |
| Memory constraints during training | Use batch size 32-64; if RAM is tight, use a streaming dataset that loads batches on-demand |

## Future Enhancements (Out of Scope for V1)

- Web UI for visualizing climbs on a board diagram
- Export climbs to Kilter app format
- Diffusion model for higher quality generation
- User feedback loop (rate generated climbs, fine-tune)
- Multi-board support (Tension, Decoy, etc.)
- Hold reachability constraints (max distance between consecutive holds)
