from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Fine-tune a small language model")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps for gradient accumulation")
args = parser.parse_args()

# Load dataset
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

# Rename to 'text' only if needed
dataset = dataset.map(lambda x: {"text": x["text"].strip()}).remove_columns([col for col in dataset.column_names if col != "text"])

# Tokenizer + model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Enable gradient checkpointing if requested (saves memory but slightly slower)
if args.gradient_checkpointing:
    print("Enabling gradient checkpointing for memory efficiency")
    model.gradient_checkpointing_enable()

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
tokenized = dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    save_strategy="no",
)

print(f"Using gradient accumulation steps: {args.gradient_accumulation_steps}")
print(f"Effective batch size: {1 * args.gradient_accumulation_steps}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
