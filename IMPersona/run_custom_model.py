from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict
import os
from peft import PeftModel

def load_model(model_path, adapter_path=None):
    """
    Load a local model and tokenizer
    
    Args:
        model_path: Path to the base model
        adapter_path: Optional path to a LoRA adapter
    """
    print(f"Loading model from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if adapter_path:
        print(f"Loading adapter from {adapter_path}...")
        # First load the base model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Then load and apply the LoRA adapter
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        # Load the model without adapter
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Warning: No adapter path provided, using model as is.")
    
    return model, tokenizer

def run_custom_model(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, history: List[Dict], max_new_tokens=512, temperature=0.7):
    """
    Run inference on a local model with the given chat history
    
    Args:
        model: The loaded model object
        tokenizer: The loaded tokenizer object
        history: List of message dictionaries in the format [{"role": "user", "content": "..."}, ...]
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness in generation (higher = more random)
        
    Returns:
        Generated text response
    """
    # Format the conversation using the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        # Convert history to the format expected by apply_chat_template
        messages = [{"role": msg["role"], "content": msg["content"]} for msg in history]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    else:
        raise ValueError("Tokenizer does not support chat template")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    if response.startswith(prompt):
        response = response[len(prompt):]
    
    return response.strip()