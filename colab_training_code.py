"""
Complete training code for DialoGPT-small with LoRA fine-tuning.
Copy each section into separate cells in Colab notebook.
"""

# ============================================================================
# CELL 1: Load configuration and prompt templates
# ============================================================================
import json
import sys
from pathlib import Path

# Add prompt_templates to path (if uploaded to Drive)
drive_base = "/content/drive/MyDrive/6521_AI_Final_Project"
sys.path.insert(0, drive_base)

# Load configuration
with open(f"{drive_base}/config.json", 'r') as f:
    config = json.load(f)

# Load prompt templates (copy prompt_templates.py content or import from Drive)
from prompt_templates import format_training_example, format_prompt

# Load test sample IDs
with open(f"{drive_base}/data/test_sample_ids.json", 'r') as f:
    test_samples = json.load(f)

print("Configuration loaded successfully!")
print(f"Test samples: {test_samples['total_count']} total")

# ============================================================================
# CELL 2: Clear GPU cache and load DialoGPT-small model and tokenizer
# ============================================================================
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gc

# Clear GPU cache before loading model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU cache cleared")

model_name = "microsoft/DialoGPT-small"

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set pad_token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

print(f"Model loaded on {device}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test tokenizer
test_text = "Hello, how are you?"
tokens = tokenizer.encode(test_text, return_tensors="pt").to(device)
decoded = tokenizer.decode(tokens[0])
print(f"Tokenizer test - Encoded: {tokens.shape}, Decoded: {decoded}")

# ============================================================================
# CELL 3: Create PyTorch Dataset
# ============================================================================
from torch.utils.data import Dataset, DataLoader
import json
from pathlib import Path

class DialogueDataset(Dataset):
    """Dataset for loading dialogue data from JSONL files."""
    
    def __init__(self, jsonl_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))
        
        print(f"Loaded {len(self.examples)} examples from {jsonl_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        context = example['context']
        response = example['response']
        
        # Format training example using shared template
        training_text = format_training_example(context, response)
        
        # Tokenize - don't create tensors here, return lists
        encoding = self.tokenizer(
            training_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors=None  # Return lists, not tensors
        )
        
        # For causal LM, labels are the same as input_ids
        labels = encoding['input_ids'].copy()
        
        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': labels
        }

# Create datasets
base_dir = f"{drive_base}/data/processed_sample"

# Reduce max_length to save memory (T4 GPU has limited memory)
max_length = 64  # Further reduced to 64 to save memory

# Option 1: Use full dataset (if memory allows)
# Option 2: Use subset for testing (uncomment to use)
USE_SUBSET = False  # Set to True to use only first 1000 samples for testing
SUBSET_SIZE = 1000

train_dataset = DialogueDataset(
    f"{base_dir}/dailydialog/train.jsonl",
    tokenizer,
    max_length=max_length
)

# Optionally limit dataset size for testing
if USE_SUBSET:
    train_dataset.examples = train_dataset.examples[:SUBSET_SIZE]
    print(f"Using subset of {SUBSET_SIZE} samples for testing")

val_dataset = DialogueDataset(
    f"{base_dir}/dailydialog/validation.jsonl",
    tokenizer,
    max_length=max_length
)

if USE_SUBSET:
    val_dataset.examples = val_dataset.examples[:SUBSET_SIZE // 10]  # Smaller validation set

# Optionally combine with empathetic_dialogues
# You can create separate datasets and combine them with ConcatDataset

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ============================================================================
# CELL 4: Create DataLoader
# ============================================================================
# Reduce batch size to save memory - use batch_size=1 for maximum memory savings
batch_size = 1  # Minimum batch size to save memory
gradient_accumulation_steps = 16  # Increased to maintain effective batch size = 1 * 16 = 16

# Note: Trainer will create its own DataLoader, so these are optional
# But we can still create them for manual inspection if needed
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     num_workers=0,
#     pin_memory=False  # Disable pin_memory to save memory
# )

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ============================================================================
# CELL 5: Configure LoRA with PEFT
# ============================================================================
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# Configure quantization - try 4-bit for even less memory usage
# If 4-bit doesn't work, fall back to 8-bit
use_4bit = True  # Set to False to use 8-bit instead

if use_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    print("Using 4-bit quantization for maximum memory savings")
else:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    print("Using 8-bit quantization")

# Reload model with quantization
print("Loading model with 8-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# Enable gradient checkpointing to save memory
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    print("Gradient checkpointing enabled")

# Configure LoRA - reduce rank to save memory
lora_rank = 4  # Reduced from 8 to save memory (can increase if memory allows)
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=lora_rank * 2,  # Typically alpha = 2 * rank
    target_modules=["c_attn"],  # Target attention modules for DialoGPT
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
print(f"Total parameters: {total_params:,}")

# ============================================================================
# CELL 6: Configure training parameters
# ============================================================================
from transformers import TrainingArguments, Trainer
import os

# Save checkpoints to Drive so they persist after Colab restart
checkpoint_dir = f"{drive_base}/models/dialogpt_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Also keep a local copy for faster access during training
local_checkpoint_dir = "/content/dialogpt_checkpoints"
os.makedirs(local_checkpoint_dir, exist_ok=True)

output_dir = checkpoint_dir  # Save to Drive

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=2e-4,
    fp16=True,  # Mixed precision training
    gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
    logging_steps=100,  # Less frequent logging to save memory
    eval_steps=1000,  # Less frequent evaluation to save memory
    save_steps=1000,  # Less frequent saving to save memory
    eval_strategy="steps",  # Changed from evaluation_strategy in newer transformers versions
    save_strategy="steps",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=100,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    report_to="none",  # Disable wandb/tensorboard in Colab
    save_total_limit=3,  # Keep only last 3 checkpoints
    dataloader_pin_memory=False,  # Disable pin_memory to save memory
    remove_unused_columns=False,  # Keep all columns
    dataloader_num_workers=0,  # No multiprocessing to save memory
    max_grad_norm=1.0,  # Gradient clipping
)

print("Training arguments configured!")

# ============================================================================
# CELL 7: Define evaluation metrics and Trainer
# ============================================================================
import numpy as np
from transformers import Trainer

def compute_metrics(eval_pred):
    """Compute perplexity for evaluation."""
    predictions, labels = eval_pred
    # Shift so that tokens < n predict n
    shift_logits = predictions[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tokens
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Calculate perplexity
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(shift_logits, shift_labels)
    perplexity = torch.exp(loss)
    
    return {"perplexity": perplexity.item()}

# Custom data collator that converts lists to tensors
# Trainer will automatically move tensors to the correct device
def data_collator(features):
    """Convert list features to tensors, handling padding."""
    batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': []
    }
    
    for feature in features:
        batch['input_ids'].append(feature['input_ids'])
        batch['attention_mask'].append(feature['attention_mask'])
        batch['labels'].append(feature['labels'])
    
    # Convert to tensors (Trainer will move to device automatically)
    batch = {
        'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
        'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
        'labels': torch.tensor(batch['labels'], dtype=torch.long)
    }
    
    return batch

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator
)

print("Trainer created!")

# ============================================================================
# CELL 8: Clear memory before training
# ============================================================================
import gc
import glob

# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("GPU cache cleared before training")

# ============================================================================
# CELL 9: Check for existing checkpoint and resume training
# ============================================================================

# Check if there are existing checkpoints
checkpoint_files = glob.glob(f"{checkpoint_dir}/checkpoint-*")
if checkpoint_files:
    # Get the latest checkpoint
    checkpoint_numbers = [int(f.split("-")[-1]) for f in checkpoint_files]
    latest_checkpoint = max(checkpoint_numbers)
    resume_from_checkpoint = f"{checkpoint_dir}/checkpoint-{latest_checkpoint}"
    print(f"Found existing checkpoint: {resume_from_checkpoint}")
    print("Resuming training from checkpoint...")
else:
    resume_from_checkpoint = None
    print("No existing checkpoint found. Starting fresh training...")

# ============================================================================
# CELL 10: Start training (with resume capability)
# ============================================================================
print("Starting training...")

# Additional memory optimization: set environment variable
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Clear cache one more time
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

try:
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
except RuntimeError as e:
    if "out of memory" in str(e):
        print("\n" + "="*50)
        print("CUDA OUT OF MEMORY ERROR!")
        print("="*50)
        print("\nTry these solutions:")
        print("1. Set USE_SUBSET = True in CELL 3 to use smaller dataset")
        print("2. Reduce max_length further (e.g., to 32)")
        print("3. Reduce batch_size to 1 (already done)")
        print("4. Restart Colab runtime and try again")
        print("5. Use Colab Pro for more GPU memory")
        print("\nClearing GPU cache...")
        torch.cuda.empty_cache()
        gc.collect()
        raise
    else:
        raise

print("Training completed!")
print(f"Best checkpoint saved at: {trainer.state.best_model_checkpoint}")

# Copy final checkpoint to Drive if using local directory
if output_dir != checkpoint_dir:
    import shutil
    if os.path.exists(local_checkpoint_dir):
        for item in os.listdir(local_checkpoint_dir):
            src = os.path.join(local_checkpoint_dir, item)
            dst = os.path.join(checkpoint_dir, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print(f"Checkpoints copied to Drive: {checkpoint_dir}")

# ============================================================================
# CELL 11: Save final model to Drive
# ============================================================================
# Save final model to Drive (persistent storage)
drive_model_path = f"{drive_base}/models/dialogpt_final"
os.makedirs(drive_model_path, exist_ok=True)

# Save the best model (loaded automatically if load_best_model_at_end=True)
model.save_pretrained(drive_model_path)
tokenizer.save_pretrained(drive_model_path)
print(f"Final model saved to Drive: {drive_model_path}")

# Also save to local for quick access
local_model_path = "/content/dialogpt_final_model"
model.save_pretrained(local_model_path)
tokenizer.save_pretrained(local_model_path)
print(f"Model also saved locally: {local_model_path}")

# ============================================================================
# CELL 11: Load saved model (for resuming or inference)
# ============================================================================
# If you want to load a previously saved model instead of training:
# Uncomment the following code:

# from peft import PeftModel
# 
# # Load base model
# base_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quantization_config,
#     device_map="auto"
# )
# 
# # Load LoRA weights
# model = PeftModel.from_pretrained(
#     base_model,
#     f"{drive_base}/models/dialogpt_final"
# )
# 
# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(f"{drive_base}/models/dialogpt_final")
# 
# print("Model loaded from Drive!")

# ============================================================================
# CELL 12: Test generation (optional - quick test)
# ============================================================================
model.eval()
test_context = "Hello, how are you?"

# Format prompt using shared template
prompt = format_prompt(test_context)
print(f"Prompt: {prompt}")

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generated: {generated_text}")

