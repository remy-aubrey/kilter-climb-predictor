# Project History & Learnings

## What We Built

### Data Pipeline
- Downloaded Kilter Board database via `boardlib` (198MB, 344k climbs)
- Preprocessed to 100,714 climbs at 40° angle
- 771 unique holds, 80/10/10 train/val/test split
- Hold types: start, middle, finish, foot
- LED colors: green, cyan, magenta, orange

### Models Tried

#### 1. LSTM (Original Approach)
- **Status**: Failed
- **Problem**: Mode collapse — predicted "middle" holds 100% of the time
- **Why**: 45.7% of holds are "middle", so the model learned to always predict the most common type
- **Fix attempted**: Class weights (middle=0.38, finish=1.75) — didn't help enough
- **Result**: Abandoned in favor of Markov chain

#### 2. Markov Chain
- **Status**: Partially working, but not good enough
- **What worked**:
  - Position-aware transitions (filter by distance)
  - Smooth transitions (borrow from nearby holds)
  - Template finish fallback (ensure climb reaches top)
- **What didn't work**:
  - Holds not sorted by y (climbing order) — fixed
  - Feet mixed with hands — fixed
  - Stopping rule based on chance, not height — fixed
  - Too little data per hold — partially fixed with smoothing
  - Constraints applied by giving up — fixed with resampling
- **Result**: Climbs were too short, didn't reach top, not realistic

#### 3. Transformer (Original)
- **Status**: Failed
- **Problem**: Generated horizontal lines of holds — not realistic climbs
- **Why**: Model learned to predict the same position regardless of input
  - Position regressor collapsed to constant output
  - Hold type classifier predicted "middle" 100% of time (mode collapse)
  - Model wasn't learning sequential structure
- **Result**: Climbs were horizontal lines with no start/finish holds, no progression
- **Key insight**: Near-zero validation loss (0.0243) didn't mean memorization — it meant the model found a trivial solution (predict constant) that happened to minimize loss on training data

#### 4. GPT-Style Transformer (Current)
- **Status**: Working! ish! Produces almost realistic climbs
- **Key fixes**:
  - One token per (hold, role) pair with one output head
  - Sequence format: [GRADE] start start middle ... finish feet... [END]
  - Decoder-only model with causal mask
  - Input = seq[:-1], target = seq[1:], padding ignored
  - No class weights
  - Rule-based masking during sampling
- **Rules enforced during sampling**:
  - First holds: 1-2 start holds in bottom third
  - Hand moves: must go up or level, within reach limit
  - Finish: only near top, after minimum hand holds
  - After finish: only feet or END
  - No repeats
  - END: only after finish
- **Result**: Climbs start at bottom, progress upward, finish near top, 7-9 holds. Climbs are generally still too short though. They sometimes are missing start holds, and tend to cluster middle holds together. The climbs are not very natural or good. There are improvements to be made. Next step will be to train with more data.

## Key Learnings

### Data
- Kilter Board database has 344k climbs at 31 angles
- 40° angle has 100k+ climbs (most popular)
- Hold types: start (13.5%), middle (45.7%), finish (10%), foot (30.8%)
- Coordinates are in cm, range: x ∈ [-56, 204], y ∈ [-12, 291]
- Frames string format: `p{hole_id}r{role_id}` pairs

### Model Architecture
- **LSTM**: Suffers from mode collapse on imbalanced data
- **Markov Chain**: Simple but effective, needs careful constraint handling
- **Transformer**: Powerful but prone to overfitting, needs regularization
- **GPT-style**: Best results — decoder-only with causal mask + rule-based masking

### Training
- **Overfitting**: Validation loss near zero = model is peeking at answer
- **Class weights**: Help with imbalanced data but don't solve mode collapse
- **Regularization**: Dropout, weight decay, label smoothing help but aren't enough
- **Rule-based masking**: Most effective way to ensure valid output

### Generation
- **Greedy decoding**: Always picks most likely hold — leads to mode collapse
- **Temperature sampling**: Adds randomness, helps diversity
- **Top-k sampling**: Limits to top-k most likely — balances quality/diversity
- **Rule-based constraints**: Essential for valid climbs

## What's Working Now

- GPT-style transformer with rule-based masking
- Trained for 2 epochs on 50% subset (40k samples)
- Validation loss: 4.07 (healthy range: 2-4)
- Generates climbs with:
  - Start holds at bottom (most of the time)
  - Upward progression
  - Finish holds near top (but still quite low)
  - 7-9 holds per climb (clustered together)
  - Foot holds after placed after the hand holds

## What's Next

1. **Train for more epochs** (20-50) to improve quality
2. **Train on full dataset** for better generalization
3. **Tune hyperparameters** (temperature, top_k, max_reach)
4. **Add evaluation metrics** (diversity, novelty, grade appropriateness)
5. **Add unit tests** for the GPT model
6. **Create better UI** to show a more readable grid

## Files

- `src/model/gpt.py` — GPT-style transformer model
- `src/model/train_gpt.py` — Training script for GPT model
- `src/model/markov.py` — Markov chain model (deprecated)
- `src/model/transformer.py` — Original transformer (deprecated)
- `src/cli/main.py` — CLI interface
- `src/cli/visualize.py` — ASCII visualization
- `src/data/preprocess.py` — Data preprocessing
- `src/data/download.py` — Database download
