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

These files are all the team needs for tomorrow’s modeling work. No extra txt files are required for EmpatheticDialogues; keeping the tarball plus the JSONL outputs is enough.

