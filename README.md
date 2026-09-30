# kilter-climb-predictor

A Python CLI tool that uses machine learning to generate new climbing routes for the Kilter Board. Specify a grade (e.g., V5) and the tool outputs a new climb with hold positions and LED colors.

## How It Works

1. **Data**: Downloads the Kilter Board climb database (100k+ climbs) via `boardlib`
2. **Preprocessing**: Parses climb sequences, encodes hold types/positions/grades, splits into train/val/test
3. **Model**: LSTM-based sequence generator conditioned on grade — learns P(next_hold | sequence_so_far, grade)
4. **Generation**: Autoregressively samples new climbs, validates reachability constraints
5. **CLI**: Typer-based CLI with ASCII visualization

## Installation

```bash
# Clone and enter the project
git clone <repo-url>
cd kilter-climb-predictor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"
```

## Setup

1. Copy `.env.example` to `.env` and fill in your Kilter Board credentials:
   ```bash
   cp .env.example .env
   ```

2. Download the database (no username needed — uses bundled data from the app):
   ```bash
   boardlib database kilter data/kilter.db
   ```

## Usage

### Preprocess the data
```bash
python -m src.data.preprocess --db data/kilter.db --output data/processed/
```

### Train the model
```bash
# Quick smoke test (2 epochs)
python -m src.model.train --epochs 2 --batch-size 64

# Full training run
python -m src.model.train --epochs 50 --batch-size 32 --patience 5
```

### Generate climbs
```bash
# Using the CLI
kilter-gen generate --grade V5 --count 3

# Or directly
python -m src.model.generate --grade V5 --count 3 --temperature 1.0
```

### Visualize a climb
```bash
kilter-gen visualize --grade V5
```

### View dataset statistics
```bash
kilter-gen stats
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `kilter-gen generate --grade V5 --count 3` | Generate new climbs |
| `kilter-gen train --epochs 50` | Train the model |
| `kilter-gen download` | Download the database |
| `kilter-gen stats` | Show dataset statistics |
| `kilter-gen visualize --grade V5` | ASCII visualization |

## Project Structure

```
kilter-climb-predictor/
├── src/
│   ├── data/
│   │   ├── download.py      # Download DB via boardlib
│   │   ├── schema.py        # DB schema exploration
│   │   └── preprocess.py    # Clean and transform raw data
│   ├── model/
│   │   ├── dataset.py       # PyTorch Dataset for climbs
│   │   ├── network.py       # LSTM model definition
│   │   ├── train.py         # Training loop
│   │   └── generate.py      # Inference / generation logic
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

## Reachability Constraints

Generated climbs are validated against user measurements (height: 157cm, wingspan: 157cm):

| Constraint | Value |
|------------|-------|
| Max hand-to-hand distance | 140cm |
| Max foot-to-hand distance | 120cm |
| Max foot-to-foot distance | 100cm |

These are configurable in `src/utils/constants.py`.

## Technology

- **Data**: `boardlib` for download, `pandas` for processing
- **ML**: `PyTorch` with LSTM sequence generator
- **CLI**: `typer` + `rich` for pretty terminal output
- **Testing**: `pytest`

## License

MIT
