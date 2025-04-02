import json
import os
import sys
from dotenv import load_dotenv
from memory_utils import extract_personal_attributes_parallel, combine_and_infer_attributes_parallel, generate_top_order_memories_sync

load_dotenv()

# Configuration
DATA_DIR = 'data'
TEMP_DIR = 'temp'
MEMORY_BANK_PATH = os.path.join(DATA_DIR, 'memory_bank.json')
CONVERSATION_STORE_PATH = os.path.join(DATA_DIR, 'conversation_store_memory.json')
GATHERED_ATTRIBUTES_PATH = os.path.join(TEMP_DIR, 'gathered_attributes.json')
COMBINED_ATTRIBUTES_PATH = os.path.join(TEMP_DIR, 'combined_attributes.json')
TOP_MEMORIES_PATH = os.path.join(TEMP_DIR, 'top_memories.json')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Model configuration
model_name_extract = "gpt-4o-mini"
model_name_combine = "gpt-4o"
batch_size_extract = 20
batch_size_combine = 25
model_name_top = "gpt-4o"
target_count_top = 25
max_concurrent = 100

person_name = input("Enter the name of the person you want to create a memory bank for (usually your name): ")

# Check if memory bank already exists
if os.path.exists(MEMORY_BANK_PATH):
    print("Memory bank already exists. Exiting...")
    sys.exit()

# Check if we can skip all processing
if os.path.exists(COMBINED_ATTRIBUTES_PATH):
    print("Found combined attributes, skipping extraction and combination steps...")
    with open(COMBINED_ATTRIBUTES_PATH, 'r') as f:
        combined_attributes = json.load(f)
    
    # Check if top memories already exist
    if os.path.exists(TOP_MEMORIES_PATH):
        print("Found top memories, skipping generation step...")
        with open(TOP_MEMORIES_PATH, 'r') as f:
            top_memories = json.load(f)
    else:
        # Generate top-order memories
        print("Generating top-order memories...")
        top_memories = generate_top_order_memories_sync(
            combined_attributes,
            model_name_top,
            target_count_top,
            person_name,
            max_concurrent
        )
        with open(TOP_MEMORIES_PATH, 'w') as f:
            json.dump(top_memories, f)
    
    combined_attributes["top_memories"] = top_memories
else:
    # Check if conversation store exists
    if not os.path.exists(CONVERSATION_STORE_PATH):
        print(f"Error: Conversation store not found at {CONVERSATION_STORE_PATH}")
        sys.exit(1)
        
    # Load conversation store
    with open(CONVERSATION_STORE_PATH, 'r') as f:
        conversation_store = json.load(f)
    
    # Extract attributes if needed
    if os.path.exists(GATHERED_ATTRIBUTES_PATH):
        print("Found gathered attributes, skipping extraction step...")
        with open(GATHERED_ATTRIBUTES_PATH, 'r') as f:
            attributes = json.load(f)
    else:
        print("Extracting personal attributes...")
        attributes = extract_personal_attributes_parallel(
            conversation_store,
            model_name_extract,
            batch_size_extract, 
            person_name,
            max_concurrent
        )
        with open(GATHERED_ATTRIBUTES_PATH, 'w') as f:
            json.dump(attributes, f)
    
    # Combine attributes
    print("Combining and inferring attributes...")
    combined_attributes = combine_and_infer_attributes_parallel(
        attributes,
        model_name_combine,
        batch_size_combine,
        person_name,
        max_concurrent
    )
    with open(COMBINED_ATTRIBUTES_PATH, 'w') as f:
        json.dump(combined_attributes, f)

    # Generate top-order memories
    print("Generating top-order memories...")
    top_memories = generate_top_order_memories_sync(
        combined_attributes,
        model_name_top,
        target_count_top,
        person_name,
        max_concurrent
    )
    with open(TOP_MEMORIES_PATH, 'w') as f:
        json.dump(top_memories, f)
    
    combined_attributes["top_memories"] = top_memories

# Save the final memory bank
print(f"Saving memory bank for {person_name}...")
with open(MEMORY_BANK_PATH, 'w') as f:
    json.dump(combined_attributes, f)

print("Memory bank created successfully!")