# Step 1: Install libraries
!pip install --upgrade wandb transformers datasets

# Step 2: Restart runtime manually after installing, then continue from here

# Step 3: Import and log in to wandb 
import wandb
wandb.login(key="b83de0abf9e84d191165f0da2d94f4ab94c338ef")  

# Step 4: Upload  dataset 
from google.colab import files
uploaded = files.upload()  

# Step 5: Read and clean the text
with open("quotes.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

from datasets import Dataset
dataset = Dataset.from_dict({"text": lines})

# Step 6: Load GPT-2 and tokenizer
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Avoid padding token errors

model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 7: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 8: Define training arguments with W&B logging enabled
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=5,
    report_to="wandb",
    run_name="gpt2-quotes-finetune"
)

# Step 9: Set up Trainer
from transformers import DataCollatorForLanguageModeling, Trainer

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Number of examples:", len(tokenized_dataset))
print("Sample tokenized text:", tokenized_dataset[0])

# Step 10: Train the model
trainer.train()

# Step 11: Save the model and tokenizer
model.save_pretrained("finetuned-gpt2-quotes")
tokenizer.save_pretrained("finetuned-gpt2-quotes")
