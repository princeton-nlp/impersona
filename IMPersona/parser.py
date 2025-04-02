from abc import ABC, abstractmethod
from typing import List, Dict
import os
import json
import glob
from datetime import datetime, timedelta

class MessageParser(ABC):
    """Base class for parsing different types of message exports into conversations."""
    
    def __init__(self, max_gap_hours: float = 6.0):
        """
        Initialize the parser.
        
        Args:
            max_gap_hours: Maximum time gap between messages to be considered same conversation
        """
        self.max_gap_hours = max_gap_hours

    @abstractmethod
    def parse(self, input_path: str) -> List[List[Dict]]:
        """Parse messages into conversations."""
        pass

    def save_conversations(self, conversations: List[List[Dict]], output_file: str) -> None:
        """Save parsed conversations to JSON file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, indent=2, default=str)

class DiscordParser(MessageParser):
    """Parser for Discord message exports.
    
    Accepts:
        - Folder containing Discord JSON files (named *_page_*.json)
        - Each JSON file should contain an array of message objects
    """

    def _load_all_pages(self, folder_path: str) -> List[Dict]:
        all_messages = []
        json_pattern = os.path.join(folder_path, "*_page_*.json")
        json_files = glob.glob(json_pattern)
        
        for file_path in json_files:
            with open(file_path, 'r') as f:
                messages = json.load(f)
                all_messages.extend(messages)
        
        return all_messages

    def parse(self, folder_path: str) -> List[List[Dict]]:
        messages = self._load_all_pages(folder_path)
        
        # Filter valid messages
        messages = [
            msg for msg in messages 
            if msg.get('content') and msg['content'].strip() and msg.get('type') == 0
        ]
        
        # Sort messages by timestamp
        messages.sort(key=lambda x: x['timestamp'])
        
        conversations: List[List[Dict]] = []
        current_conversation: List[Dict] = []
        
        for msg in messages:
            timestamp = datetime.fromisoformat(msg['timestamp'].replace('Z', '+00:00'))
            
            cleaned_msg = {
                'content': msg['content'].strip(),
                'speaker': msg['author'].get('global_name') or msg['author'].get('username'),
                'timestamp': timestamp
            }
            
            if not current_conversation:
                current_conversation.append(cleaned_msg)
                continue
                
            last_msg_time = current_conversation[-1]['timestamp']
            
            if timestamp - last_msg_time > timedelta(hours=self.max_gap_hours):
                if current_conversation:
                    conversations.append(current_conversation)
                current_conversation = [cleaned_msg]
            else:
                current_conversation.append(cleaned_msg)
        
        if current_conversation:
            conversations.append(current_conversation)
        
        return conversations

class IMessageParser(MessageParser):
    """Parser for iMessage exports.
    
    Accepts:
        - Text file containing iMessage export
        - Format should be:
          [Timestamp]
          [Sender]
          [Content]
          (repeated)
    """

    def parse(self, file_path: str) -> List[List[Dict]]:
        timestamp_pattern = r"([A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M)"
        
        conversations: List[List[Dict]] = []
        current_conversation: List[Dict] = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            timestamp_match = re.match(timestamp_pattern, line)
            if timestamp_match:
                timestamp_str = timestamp_match.group(1)
                timestamp = datetime.strptime(timestamp_str, "%b %d, %Y %I:%M:%S %p")
                
                i += 1
                if i >= len(lines):
                    break
                sender = lines[i].strip()
                
                i += 1
                if i >= len(lines):
                    break
                content = lines[i].strip()
                
                # Skip system messages and attachments
                if (content.startswith("This message") or 
                    content.startswith("Tapbacks:") or 
                    content.startswith("/Users/") or
                    content.startswith("Edited to")):
                    i += 1
                    continue
                    
                cleaned_msg = {
                    'content': content,
                    'speaker': 'Me' if sender == 'Me' else sender,
                    'timestamp': timestamp
                }
                
                if not current_conversation:
                    current_conversation.append(cleaned_msg)
                else:
                    last_msg_time = current_conversation[-1]['timestamp']
                    if timestamp - last_msg_time > timedelta(hours=self.max_gap_hours):
                        if current_conversation:
                            conversations.append(current_conversation)
                        current_conversation = [cleaned_msg]
                    else:
                        current_conversation.append(cleaned_msg)
                        
            i += 1
        
        if current_conversation:
            conversations.append(current_conversation)
        
        return conversations