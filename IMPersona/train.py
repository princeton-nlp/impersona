import argparse
from dotenv import load_dotenv
load_dotenv()
import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Set up argument parser
parser = argparse.ArgumentParser(description='Train an IMPersona Locally')
parser.add_argument('--dataset_path', type=str, default=f"./data/impersona_imessage_0buffer_B50.json",
                    help='path to dataset')
parser.add_argument('--model_name', type=str, default="/scratch/gpfs/qbshi/.cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
                    help='base model to fine-tune')
parser.add_argument('--output_dir', type=str, default="./output",
                    help='directory to save the model')
parser.add_argument('--num_epochs', type=int, default=3,
                    help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4,
                    help='learning rate for training')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch size for training')
parser.add_argument('--use_lora', action='store_true', default=True,
                    help='use LoRA for parameter-efficient fine-tuning')
parser.add_argument('--lora_r', type=int, default=8,
                    help='rank of LoRA matrices')
parser.add_argument('--lora_alpha', type=int, default=32,
                    help='scaling factor for LoRA')
parser.add_argument('--lora_dropout', type=float, default=0.05,
                    help='dropout probability for LoRA layers')
parser.add_argument('--max_seq_length', type=int, default=2048,
                    help='maximum sequence length for training')
parser.add_argument('--format_template', type=str, default='default',
                    choices=['default', 'llama', 'qwen', 'custom'],
                    help='template format for instruction tuning')
parser.add_argument('--custom_template', type=str, default=None,
                    help='custom template string if format_template is set to "custom"')
parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                    help='number of steps to accumulate gradients before performing an optimization step')

args = parser.parse_args()

# Define formatting templates for different models
FORMAT_TEMPLATES = {
    'default': "{instruction}{input}{output}",
    'llama': "<|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>",
    'qwen': "<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>",
}

# Get the template to use
template = args.custom_template if args.format_template == 'custom' else FORMAT_TEMPLATES[args.format_template]
print(f"Using template format: {args.format_template}")

# Load dataset
dataset_path = args.dataset_path
with open(dataset_path, 'r') as f:
    dataset = json.load(f)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not defined

# After loading the tokenizer
if "llama" in args.model_name.lower() and args.format_template != "llama":
    print(f"Warning: Using {args.format_template} template with a Llama model. Consider using --format_template llama")
elif "qwen" in args.model_name.lower() and args.format_template != "qwen":
    print(f"Warning: Using {args.format_template} template with a Qwen model. Consider using --format_template qwen")
elif "mistral" in args.model_name.lower() and args.format_template != "mistral":
    print(f"Warning: Using {args.format_template} template with a Mistral model. Consider using --format_template mistral")

# Process dataset with proper token masking
def process_dataset(dataset):
    processed_data = []
    for item in dataset:
        # Format the prompt using the selected template
        formatted_text = template.format(
            instruction=item['instruction'],
            input=item['input'],
            output=item['output']
        )
        # This is needed to properly mask the loss for non-output tokens
        output_placeholder = template.format(
            instruction=item['instruction'],
            input=item['input'],
            output=""
        )
        
        # Tokenize the text
        encoded = tokenizer(formatted_text, truncation=True, max_length=args.max_seq_length)
        placeholder_len = len(tokenizer(output_placeholder, truncation=True, max_length=args.max_seq_length)["input_ids"])
        
        processed_data.append({
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "prompt_length": placeholder_len
        })
    return processed_data

processed_dataset = process_dataset(dataset)
train_dataset = Dataset.from_list(processed_dataset)

# Print a sample training example to verify formatting
print("\n===== SAMPLE TRAINING EXAMPLE =====")
sample_idx = 0  # First example in the dataset
sample = processed_dataset[sample_idx]
print(f"Input IDs: {sample['input_ids'][:10]}... (truncated)")
print(f"Attention mask: {sample['attention_mask'][:10]}... (truncated)")
print(f"Prompt length: {sample['prompt_length']}")
print(f"Total tokens: {len(sample['input_ids'])}")
print("===== END SAMPLE =====\n")

# Custom data collator that masks prompt tokens
class MaskingDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = "pt"
        
        # Extract input_ids and attention_mask
        input_ids = [feature["input_ids"] for feature in features]
        attention_mask = [feature["attention_mask"] for feature in features]
        prompt_lengths = [feature["prompt_length"] for feature in features]
        
        # Pad the sequences
        batch = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors=return_tensors
        )
        
        # Create labels tensor and mask prompt tokens with -100
        labels = batch["input_ids"].clone()
        for i, prompt_length in enumerate(prompt_lengths):
            labels[i, :prompt_length] = -100
            
        batch["labels"] = labels
        return batch

# Load model
if args.use_lora:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    # Load model in 8-bit precision for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adjust based on model architecture
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

# Configure training arguments
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    weight_decay=0.01,
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to=None,
)

# Create the custom data collator
data_collator = MaskingDataCollator(
    tokenizer=tokenizer,
    mlm=False
)

# Initialize SFTTrainer with our custom data collator
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
)

# Start training
print(f"Starting training on {args.model_name}...")
training_stats = trainer.train()

# Output training statistics
print("\n===== TRAINING STATISTICS =====")
print(f"Total training time: {training_stats.metrics['train_runtime']:.2f} seconds")
print(f"Training samples per second: {training_stats.metrics['train_samples_per_second']:.2f}")
print(f"Final training loss: {training_stats.metrics['train_loss']:.4f}")
print(f"Total steps: {training_stats.metrics['step']}")
if 'epoch' in training_stats.metrics:
    print(f"Epochs completed: {training_stats.metrics['epoch']:.2f}")
print("===== END TRAINING STATISTICS =====\n")

# Save the model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"Model saved to {args.output_dir}")



