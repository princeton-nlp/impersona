from IMPersona.agents import BasicAgent
from IMPersona.example_module import ExampleModule
from IMPersona.memory_module import MemoryModule, HierarchicalMemoryModule
import json
import argparse
import os
from dotenv import load_dotenv

load_dotenv()

MEMORY_BANK_PATH = "data/memory_bank.json"
CONVERSATION_STORE_PATH = "data/conversation_store.json"
TOP_K = 10
SYSTEM_MESSAGE = ""

# Set up argument parser
parser = argparse.ArgumentParser(description='Run IMPersona with different agent types and models')
parser.add_argument('--icl', action='store_true', help='Whether to use sample chats')
parser.add_argument('--memory', action='store_true', help='Whether to use memory module')
parser.add_argument('--hierarchical_memory', action='store_true', help='Whether to use hierarchical memory module')
parser.add_argument('--custom_model_path', type=str, default="",
                    help='Path to custom model, or model itself (required for custom models)')
parser.add_argument('--adapter_path', type=str, default="",
                    help='Path to LoRA adapter (optional)')
parser.add_argument('--impersonation_name', type=str, required=True,
                    help='Name of the persona to impersonate')
parser.add_argument('--model_name', type=str, required=True,
                    help='Model name to use')

args = parser.parse_args()

user_name = input("Enter your name: ")

# Set variables from parsed arguments
example_module = None
memory_module = None
if args.icl:
    example_module = ExampleModule(conversation_store_path=CONVERSATION_STORE_PATH)
    icl_example = example_module.search_conversation_store(user_name)

if args.memory and args.hierarchical_memory:
    print("Warning: Both --memory and --hierarchical_memory flags are set. Using hierarchical memory.")
    memory_module = HierarchicalMemoryModule(attribute_path=MEMORY_BANK_PATH, model=args.model_name)
elif args.hierarchical_memory: # Warning: Hierarchical memory is quite slow
    memory_module = HierarchicalMemoryModule(attribute_path=MEMORY_BANK_PATH, model=args.model_name)
elif args.memory:
    memory_module = MemoryModule(attribute_path=MEMORY_BANK_PATH)

agent = BasicAgent(args.model_name, args.impersonation_name, user_name, example_module=example_module, memory_module=memory_module, custom_model_path=args.custom_model_path, adapter_path=args.adapter_path)

print(f"Welcome to IMPersona! You are talking to {args.impersonation_name}. Type 'exit' to end the conversation.")
while True:
    user_input = input("Enter your message: ")
    if user_input == "exit":
        break
    response = agent.generate_response(user_input)
    print(f"{args.impersonation_name}: {response}")