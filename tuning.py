model_path = "model/gemma-3-ib-it"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

datasets= load_dataset("json", data_files = "datasets.json") 
datasets = datasets["train"].train_test_split(test_size = 0.25)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation='eager')
if torch.cuda.is_available():
    model = model.to("cuda")

from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

model_name = model_path.split("/")[-1]

lora_config = LoraConfig(
    r = 8,
    lora_alpha = 32,
    target_modules
     = ["q_proj", "v_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

def preprocess_function(examples):
    texts = []
    for conv in examples["messages"]:
        text = "\n".join([msg["role"] + ": " + msg["content"] for msg in conv])
        texts.append(text)
    model_inputs = tokenizer(texts, max_length=512, truncation=True, padding = "max_length")
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    return model_inputs

tokenized_datasets = datasets.map(preprocess_function, batched=True, remove_columns=datasets["train"].column_names)


training_args = TrainingArguments(
    "trained_model",
    eval_strategy = "epoch",
    learning_rate=2e-4,
    save_strategy="no",
    weight_decay=0.01,
    push_to_hub=False,
    num_train_epochs=5,
    report_to="none",
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 2
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets["train"],
    eval_dataset = tokenized_datasets["test"],
)

torch.cuda.empty_cache()
trainer.train()

# trainer.push_to_hub()
trainer.save_model("model/trained_model")
tokenizer.save_pretrained("model/trained_model")