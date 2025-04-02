import os
import json
from typing import List, Dict
import random

class ExampleModule():
    def __init__(self, conversation_store_path: str):
        assert conversation_store_path is not None, "conversation_store_path must be provided"
        assert os.path.exists(conversation_store_path), "conversation_store_path must exist"
        self.conversation_store_path = conversation_store_path
        conversation_store = []
        with open(conversation_store_path, "r") as f:
            conversation_store = json.load(f)
        
        self.conversation_store = conversation_store
        self.conversation_store_dict = self._process_conversation_store(conversation_store)

    def _process_conversation_store(self, conversation_store: List[List[Dict]]) -> dict:
        processed_store = {}
        for conversation in conversation_store:
            processed_conversation = self._process_conversation(conversation)
            if processed_conversation:
                name, messages = processed_conversation
                if not name:
                    continue
                name = name.lower().strip()
                if name not in processed_store:
                    processed_store[name] = []
                processed_store[name].append(messages)
        
        # Print the number of unique names processed
        # print(f"Processed {len(processed_store)} unique conversation partners")
        return processed_store

    def _process_conversation(self, conversation: List[Dict]) -> tuple:
        # Check if this is a one-on-one conversation (only "Me" and one other person)
        participants = set()
        for message in conversation:
            if "speaker" in message:
                participants.add(message["speaker"])
        
        # If there are exactly 2 participants and one is "Me"
        if len(participants) == 2 and "Me" in participants:
            # Get the other person's name
            other_person = next(name for name in participants if name != "Me")
            return (other_person, conversation)
        
        return None, None  # Not a one-on-one conversation

    def search_conversation_store(self, query_name: str) -> str:
        query_name = query_name.lower().strip()
        selected_conversation = None
        
        # Search for matching conversations
        for key in self.conversation_store_dict:
            if query_name in key:
                selected_conversation = random.choice(self.conversation_store_dict[key])
                print(f"Found conversation example for {query_name}")
                break
        
        # If no match found, select a random conversation
        if selected_conversation is None:
            random_name = random.choice(list(self.conversation_store_dict.keys()))
            selected_conversation = random.choice(self.conversation_store_dict[random_name])
            # print(f"No conversation example found for {query_name}, using random conversation with {random_name}")
        
        final_conversation = ""
        for message in selected_conversation:
            final_conversation += f"{message['timestamp']} {message['speaker']}: {message['content']}\n"
        return final_conversation
