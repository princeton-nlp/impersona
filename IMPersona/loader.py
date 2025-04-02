from typing import List, Dict
import os
import json
from IMPersona.formatter import ModelFormatter
from datasets import Dataset

STANDARD_SYSTEM_PROMPT = "You are a human being named {}. You are not an AI. Respond as yourself. Now you may begin the conversation."

class MessageLoader:
    """
    Loads parsed message conversations and converts them into a format suitable 
    for fine-tuning with various models.
    """
    
    def __init__(self, conversations_dir: str, formatter: ModelFormatter, persona_name: str, metadata_file: str = None, system_prompt: str = STANDARD_SYSTEM_PROMPT):
        """
        Initialize the MessageLoader.
        
        Args:
            conversations_dir: Path to the directory containing JSON conversation files
            formatter: ModelFormatter instance to use for formatting conversations
        """
        self.conversations_dir = conversations_dir
        self.metadata_file = metadata_file
        self.formatter = formatter
        self.conversations_by_group = self._load_conversations()
        self.metadata = self._load_metadata() # Dictionary that maps from user to user metadata
        self.system_prompt = system_prompt.format(persona_name)
    
    def _load_metadata(self) -> Dict:
        """
        Load system prompt metadata from a file.

        Returns:
            Dictionary mapping usernames to their metadata descriptions
        """
        if not self.metadata_file:
            return {}
            
        with open(self.metadata_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content by separator and filter out empty entries
        entries = [entry.strip() for entry in content.split('---') if entry.strip()]
        
        metadata_dict = {}
        for entry in entries:
            # Split each entry into name and description
            if ':' in entry:
                name, description = entry.split(':', 1)
                name = name.strip()
                description = description.strip()
                if name and description:  # Only add if both name and description exist
                    metadata_dict[name] = description
        
        return metadata_dict
    
    def _load_conversations(self) -> Dict[str, List[Dict]]:
        """
        Load conversations from all JSON files in the specified directory.
        
        Returns:
            Dictionary mapping group names (from filenames) to lists of conversations
        """
        conversations_by_group = {}
        for filename in os.listdir(self.conversations_dir):
            if filename.endswith('.json'):
                group_name = filename[:-5]  # Remove .json extension
                file_path = os.path.join(self.conversations_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    conversations = json.load(f)
                    conversations_by_group[group_name] = conversations
        return conversations_by_group

    def _generate_metadata_string(self, group_chat: List[List[Dict]]) -> str:
        """
        Generate a metadata string containing information about unique participants in the group chat.
        
        Args:
            group_chat: List of conversations, where each conversation is a list of message dictionaries
            
        Returns:
            A formatted string containing relevant participant metadata
        """
        # Get unique participants from all conversations in the group chat
        participants = set()
        for conversation in group_chat:
            for message in conversation:
                if 'speaker' in message:
                    participants.add(message['speaker'])
        
        # Build metadata string for participants that have metadata
        metadata_entries = []
        for participant in participants:
            if participant in self.metadata:
                metadata_entries.append(f"{participant}: {self.metadata[participant]}")
        
        if not metadata_entries:
            return ""
        
        # Construct the final string
        header = "Here is the information about the people (note that not everyone in the conversation is necessarily included, only the people that are most relevant. This is my reflections in Febuary 2024."
        metadata_text = "\n".join(metadata_entries)
        
        return f"{header}\n{metadata_text}"
    
    def generate_set(self, group_chat: List[List[Dict]], conversation_buffer) -> Dataset:
        """
        Generate a training set from a group chat.

        Args:
            group_chat: List of conversation messages
            conversation_buffer: Number of conversations as prior context in each training example.
                               If 0, no prior conversations will be included.
            
        Returns:
            Dataset containing formatted training examples
        """
        training_examples = []
        metadata_str = self._generate_metadata_string(group_chat)
        
        # Process each conversation in the group chat
        for conv_idx, conversation in enumerate(group_chat):
            
            # Get prior conversations as context (empty list if conversation_buffer is 0)
            prior_conversations = []
            if conversation_buffer > 0:
                start_idx = max(0, conv_idx - conversation_buffer)
                prior_conversations = group_chat[start_idx:conv_idx]
            
            # Process each message where "Me" is the speaker and is not a reply
            for i, message in enumerate(conversation):
                # Skip empty messages, messages that only contain the delimiter, or messages that are replies
                if (message['speaker'] == 'Me' and 
                    message['content'].strip() and 
                    message['content'].strip() != "<|msg|>" and
                    not message.get('is_reply', False)):  # Skip if is_reply is True
                    
                    # Get conversation history up to this message
                    history = conversation[:i]
                    
                    # Format the example using the formatter
                    example = self.formatter.format_conversation(
                        system_prompt=self.system_prompt,
                        metadata=metadata_str,
                        prior_conversations=prior_conversations,
                        current_conversation=history,
                        target_response=message['content'],
                        timestamp=message['timestamp']
                    )
                    
                    training_examples.append(example)
        
        # Convert to HuggingFace Dataset
        return training_examples
        
    def generate_full_dataset(self, conversation_buffer: int = 6) -> Dataset:
        """
        Generate a complete training dataset from all group chats.

        Args:
            conversation_buffer: Number of conversations to use as prior context
                               in each training example

        Returns:
            Dataset containing formatted training examples from all group chats
        """
        all_examples = []
        
        # Process each group chat
        for group_name, conversations in self.conversations_by_group.items():
            # Generate training examples for this group chat
            group_dataset = self.generate_set(conversations, conversation_buffer)
            all_examples.extend(group_dataset)
            
        # Combine all examples into a single dataset
        return Dataset.from_list(all_examples)
        