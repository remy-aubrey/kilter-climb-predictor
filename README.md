# kilter-climb-predictor

A Python CLI tool that uses machine learning to generate new climbing routes for the Kilter Board. Specify a grade (e.g., V5) and the tool outputs a new climb with hold positions and LED colors.

## Current Status

**Working**: GPT-style transformer with rule-based masking generates climbs that start at the bottom, progress upward, and finish on a finish hold. The climbs are too short though and the holds are clustered closely together. Some climbs are missing start holds or have transitions that are too reachy or don't make much sense.

**Trained**: 2 epochs on 50% subset (40k samples). Validation loss: 4.07.

**Next**: Train for more epochs (20-50) on full dataset for better quality.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Download data (no username needed)
boardlib database kilter data/kilter.db

# Preprocess
python -m src.data.preprocess --db data/kilter.db --output data/processed/

# Train (GPT-style model)
python -m src.model.train_gpt --epochs 20 --batch-size 32

# Generate climbs
kilter-gen generate --grade V5 --count 3
```

## How It Works

1. **Data**: Downloads the Kilter Board climb database (100k+ climbs)
2. **Preprocessing**: Parses climb sequences, encodes hold types/positions/grades
3. **Model**: GPT-style transformer with causal mask
4. **Generation**: Autoregressive sampling with rule-based masking
5. **CLI**: Typer-based CLI with ASCII visualization

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Kilter DB  │────▶│  Data Layer  │────▶│  GPT Model  │────▶│   CLI    │
│  (SQLite)   │     │  (Pandas)    │     │  (PyTorch)  │     │  (Typer) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────┘
```

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

## CLI Commands

| Command | Description |
|---------|-------------|
| `kilter-gen generate --grade V5 --count 3` | Generate new climbs |
| `kilter-gen train --epochs 20` | Train the model |
| `kilter-gen download` | Download the database |
| `kilter-gen stats` | Show dataset statistics |
| `kilter-gen visualize --grade V5` | ASCII visualization |

## Technology

- **Data**: `boardlib` for download, `pandas` for processing
- **ML**: `PyTorch` with GPT-style transformer
- **CLI**: `typer` + `rich` for pretty terminal output
- **Testing**: `pytest`

## History & Learnings

See [HISTORY.md](HISTORY.md) for detailed history of what worked/didn't work and key learnings.

## License

MIT
