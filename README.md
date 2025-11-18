## Data Preparation Overview

This repo includes a `preprocess_datasets.py` helper that normalizes both DailyDialog and EmpatheticDialogues into the same JSONL schema so everyone on the team can immediately plug the data into training/evaluation scripts.

### Why JSONL?
- **Unified structure** – both datasets end up with the same keys (`dataset`, `split`, `dialog_id`, `turn_id`, `speaker`, `context`, `response`, `emotion`). Any downstream script can treat them identically.
- **Streaming-friendly** – JSON Lines can be loaded incrementally (line-by-line) which keeps memory usage low on Colab or local machines.
- **Metadata preserved** – dialog/turn ids, speaker roles, and emotion/tags are stored explicitly, which is useful for analysis and plotting.
- **Fair evaluation** – LLM side and small-model side both read the exact same `context`/`response` pairs, ensuring prompts stay aligned.

### What the script does
1. **DailyDialog**
   - Downloads `train/validation/test.zip` from Hugging Face (`roskoN/dailydialog`).
   - Reads `dialogues_{split}.txt`, splits turns using `__eou__`, normalizes punctuation/whitespace.
   - Reads `dialogues_emotion_{split}.txt` to keep the emotion ID per turn.
   - For each turn, concatenates the previous *n* utterances (default `--history 3`) with `<eot>` separators to build the `context`, and saves the current utterance as `response`.
   - Writes all samples to `data/processed/dailydialog/{split}.jsonl`.

2. **EmpatheticDialogues**
   - Downloads the official tarball from Facebook (`https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz`) into `data/cache/`.
   - Reads `empatheticdialogues/{train,valid,test}.csv` directly inside the tar without needing extra txt files.
   - Uses the provided `context` column (with `__eou__` replaced by `<eot>`) and `utterance` column as the response.
   - Keeps `conv_id`, `speaker_idx`, and `tags` as metadata, then writes JSONL to `data/processed/empathetic_dialogues/{split}.jsonl`.

### Running the script
```bash
python preprocess_datasets.py \
  --output-dir data/processed \
  --history 3 \
  --cache-dir data/cache
```

After running, you should see:
```
data/
  cache/
    empatheticdialogues.tar.gz
  processed/
    dailydialog/{train,validation,test}.jsonl
    empathetic_dialogues/{train,validation,test}.jsonl
```

These files are all the team needs for tomorrow's modeling work. No extra txt files are required for EmpatheticDialogues; keeping the tarball plus the JSONL outputs is enough.

## Shared Configuration for Fair Evaluation

To ensure fair comparison between small transformer and LLM models, we use shared configuration files that both team members must use.

### Configuration Files

1. **`config.json`** - Main configuration file containing:
   - Test sample settings (using all test samples)
   - Prompt template format
   - Context settings (history turns, separators)
   - Evaluation metrics configuration

2. **`prompt_templates.py`** - Python module with standardized prompt formatting functions:
   - `format_prompt()` - Format context for model input
   - `format_training_example()` - Format training examples
   - Standard separators and prefixes

3. **`data/test_sample_ids.json`** - List of all test sample IDs from both datasets:
   - DailyDialog: 6,740 samples
   - EmpatheticDialogues: 10,943 samples
   - Total: 17,683 samples

### Usage

**For Small Transformer (DialoGPT) side:**
```python
from prompt_templates import format_prompt, format_training_example
import json

# Load config
with open('config.json') as f:
    config = json.load(f)

# Load test sample IDs
with open('data/test_sample_ids.json') as f:
    test_samples = json.load(f)

# Format prompts using shared template
context = "Hello <eot> How are you?"
prompt = format_prompt(context)  # "[User]: Hello <eot> How are you? <eos> [System]:"
```

**For LLM side:**
```python
from prompt_templates import format_prompt
import json

# Load config and test samples (same as above)
# Use format_prompt() to ensure identical prompt format
```

### Important Notes

- **Both models must use the same prompt format** - Use `format_prompt()` from `prompt_templates.py`
- **Both models must use the same test samples** - Load IDs from `data/test_sample_ids.json`
- **Both models must use the same context settings** - Follow `config.json` settings
- **Evaluation metrics must be calculated identically** - Use the same BERTScore model, BLEU settings, etc.

### Generating Test Sample List

If you need to regenerate the test sample list:
```bash
python generate_test_samples.py
```

This will create `data/test_sample_ids.json` with all test sample IDs from both datasets.

