from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Load dataset
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

# Rename to 'text' only if needed
dataset = dataset.map(lambda x: {"text": x["text"].strip()}).remove_columns([col for col in dataset.column_names if col != "text"])

# Tokenizer + model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
tokenized = dataset.map(tokenize, batched=True)

# Train
training_args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
    save_strategy="no",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
