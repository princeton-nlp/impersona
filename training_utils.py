from datetime import datetime, timedelta
from typing import List, Dict, Protocol
from abc import ABC, abstractmethod
import re
import json
import glob
import os

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

    def __init__(self, max_gap_hours: float = 6.0, combine_consecutive: bool = True, delimiter: str = '<|msg|>'):
        """
        Initialize the parser.
        
        Args:
            max_gap_hours: Maximum time gap between messages to be considered same conversation
            combine_consecutive: Whether to combine consecutive messages from the same sender
            delimiter: Delimiter to use when combining consecutive messages
        """
        super().__init__(max_gap_hours)
        self.combine_consecutive = combine_consecutive
        self.delimiter = delimiter

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
                
                # Check for reply information in the next line
                is_reply = False
                reply_info = None
                if i + 1 < len(lines) and lines[i + 1].strip() == "This message responded to an earlier message.":
                    is_reply = True
                    reply_info = "This message responded to an earlier message."
                    i += 1  # Skip the reply info line
                
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
                    'timestamp': timestamp,
                    'is_reply': is_reply,
                    'reply_info': reply_info
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
        
        # Combine consecutive messages if enabled
        if self.combine_consecutive:
            conversations = [self._combine_consecutive_messages(conv) for conv in conversations]
        
        return conversations
    
    def _combine_consecutive_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Combines consecutive messages from the same sender into a single message.
        Uses a delimiter that's unlikely to appear in natural conversation.
        """
        if not messages:
            return messages

        combined = []
        current_group = messages[0].copy()

        for msg in messages[1:]:
            if msg['speaker'] == current_group['speaker']:
                # Combine with delimiter
                current_group['content'] += self.delimiter + msg['content']
                # Update timestamp to the latest message's timestamp
                current_group['timestamp'] = msg['timestamp']
                # If any message in the group is a reply, mark the group as a reply
                if msg.get('is_reply', False):
                    current_group['is_reply'] = True
                    current_group['reply_info'] = msg.get('reply_info')
            else:
                # Add the current group to combined list and start a new group
                combined.append(current_group)
                current_group = msg.copy()
        
        # Don't forget to add the last group
        combined.append(current_group)
        
        return combined

def normalize_phone(phone):
    """Normalize phone numbers to a standard format by removing non-digits."""
    return ''.join(c for c in phone if c.isdigit())

def parse_vcard_file(file_path):
    """Parse vCard file and return a dictionary mapping phone numbers to names."""
    phone_to_name = {}
    current_name = None
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Get the display name
            if line.startswith('FN:'):
                current_name = line[3:].strip()
            
            # Get phone numbers
            elif line.startswith('TEL') or line.startswith('item1.TEL'):
                # Extract phone number from the line
                phone = line.split(':')[-1].strip()
                
                # Normalize the phone number
                normalized_phone = normalize_phone(phone)
                
                # Only add if we have both a name and a valid phone number
                if current_name and normalized_phone:
                    # Remove any "mobile" label or other metadata
                    if len(normalized_phone) >= 10:  # Only store valid phone numbers
                        phone_to_name[normalized_phone] = current_name
            
            # Reset current name at the end of a vCard entry
            elif line == 'END:VCARD':
                current_name = None

    return phone_to_name

def normalize_phone_advanced(phone: str) -> str:
    """Normalize phone number by removing +1 or 1 prefix and any non-digit characters."""
    # Remove +1 or 1 prefix and any non-digit characters
    phone = re.sub(r'[^\d]', '', phone)
    if phone.startswith('1') and len(phone) == 11:
        phone = phone[1:]
    return phone

def load_contacts_dict(file_path: str) -> dict:
    """Load and normalize the contacts dictionary."""
    with open(file_path, 'r') as f:
        contacts = json.load(f)
    
    # Create normalized dictionary with all possible formats mapping to names
    normalized = {}
    for phone, name in contacts.items():
        norm_phone = normalize_phone_advanced(phone)
        normalized[norm_phone] = name
    
    return normalized

def replace_phone_numbers(text: str, contacts: dict) -> str:
    """Replace phone numbers with contact names in the text."""
    def replace_match(match):
        # Get the full phone number match
        phone = match.group(0)
        # Normalize the phone number
        norm_phone = normalize_phone_advanced(phone)
        # Return the contact name if found, otherwise keep original phone
        return contacts.get(norm_phone, phone)
    
    # Pattern matches +1 followed by 10 digits, 1 followed by 10 digits, or just 10 digits
    pattern = r'(?:\+1|1)?[2-9]\d{9}'
    return re.sub(pattern, replace_match, text)

def process_imessage_files(directory: str, contacts_dict_path: str):
    """Process all iMessage text files in the directory."""
    # Load and normalize contacts dictionary
    contacts = load_contacts_dict(contacts_dict_path)
    
    # Process each .txt file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Replace phone numbers with contact names
            updated_content = replace_phone_numbers(content, contacts)
            
            # Write updated content back to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            
            print(f"Processed: {filename}")

def load_and_sort_conversations(
    directory: str, 
    num_conversations: int = None, 
    min_messages: int = None,    
    max_messages: int = None      
) -> List[List[Dict]]:
    # List to store all conversations
    all_conversations = []
    
    # Define words to filter out (case-insensitive)
    filtered_words = ['sex', 'sexy', 'gamepigeon']
    
    # Load all JSON files from the directory
    for file_path in glob.glob(os.path.join(directory, '*.json')):
        with open(file_path, 'r') as f:
            conversations = json.load(f)
            
            # Filter conversations based on message count and content
            filtered_convos = []
            for conv in conversations:
                # Check message count constraints if specified
                if ((min_messages is None or len(conv) >= min_messages) and 
                    (max_messages is None or len(conv) <= max_messages)):
                    # Check if any filtered words appear in any message
                    contains_filtered_words = False
                    for message in conv:
                        content = message.get('content', '').lower()
                        if any(word in content for word in filtered_words):
                            contains_filtered_words = True
                            break
                    
                    if not contains_filtered_words:
                        filtered_convos.append(conv)
                        
            all_conversations.extend(filtered_convos)
    
    # Helper function to get the timestamp of the last message in a conversation
    def get_last_message_time(conversation: List[Dict]) -> datetime:
        if not conversation:
            return datetime.min
        # Get the timestamp from the last message
        last_message = conversation[-1]
        timestamp_str = last_message.get('timestamp', '')
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            return datetime.min

    # Sort conversations by the timestamp of their last message
    sorted_conversations = sorted(
        all_conversations,
        key=get_last_message_time,
        reverse=True  # Most recent first
    )
    
    # Take only the most recent conversations if num_conversations is specified
    if num_conversations is not None:
        recent_conversations = sorted_conversations[:num_conversations]
    else:
        recent_conversations = sorted_conversations
    
    # Print summary information
    total_loaded = len(all_conversations)
    message_range = ""
    if min_messages is not None and max_messages is not None:
        message_range = f" with {min_messages}-{max_messages} messages"
    elif min_messages is not None:
        message_range = f" with at least {min_messages} messages"
    elif max_messages is not None:
        message_range = f" with at most {max_messages} messages"
    
    print(f"Loaded {total_loaded} conversations{message_range}")
    
    if num_conversations is not None:
        print(f"Kept {len(recent_conversations)} most recent conversations")
    else:
        print(f"Returning all {len(recent_conversations)} conversations")
    
    return recent_conversations

def create_time_spaced_subset(dataset_list, size, min_time_gap=24*60*60):  # Default 1 day in seconds
    """
    Create a subset of specified size prioritizing recent messages but ensuring 
    samples are at least min_time_gap seconds apart from each other.
    
    Args:
        dataset_list: List of conversation examples with timestamps
        size: Size of the subset to create
        min_time_gap: Minimum time gap between samples in seconds
    
    Returns:
        A list containing the subset of data
    """
    # Convert string timestamps to datetime objects and then to timestamps
    from datetime import datetime
    
    def get_timestamp(item):
        """Convert item timestamp to numeric timestamp for comparison"""
        ts = item.get('timestamp', '')
        if isinstance(ts, (int, float)):
            return ts
        elif isinstance(ts, str):
            try:
                # Try to parse the timestamp string - adjust format as needed
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return dt.timestamp()
            except (ValueError, TypeError):
                try:
                    # Try another common format
                    dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                    return dt.timestamp()
                except (ValueError, TypeError):
                    return 0
        return 0
    
    # Sort by timestamp in descending order (most recent first)
    sorted_dataset = sorted(dataset_list, key=get_timestamp, reverse=True)
    
    subset = []
    last_numeric_timestamp = None
    
    # Iterate through the sorted dataset
    for item in sorted_dataset:
        current_numeric_timestamp = get_timestamp(item)
        
        # If this is the first item or the time gap is sufficient
        if last_numeric_timestamp is None or (current_numeric_timestamp > 0 and 
                                             (last_numeric_timestamp - current_numeric_timestamp) >= min_time_gap):
            subset.append(item)
            last_numeric_timestamp = current_numeric_timestamp
        
        # Stop once we have enough samples
        if len(subset) >= size:
            break
    
    # If we couldn't get enough samples with the time gap, fill with remaining most recent
    if len(subset) < size and len(subset) < len(sorted_dataset):
        # Get IDs of items already in the subset
        subset_ids = {id(item) for item in subset}
        
        # Add remaining items until we reach the desired size
        for item in sorted_dataset:
            if id(item) not in subset_ids and len(subset) < size:
                subset.append(item)
                
            if len(subset) >= size:
                break
    
    return subset

def analyze_all_messages(directory: str):
    """
    Analyze all message files and return total counts of speakers, phone numbers, and names.
    """
    # Pattern for phone numbers (matches +1XXXXXXXXXX, XXXXXXXXXX, or +XXXXXXXXXXX format)
    phone_pattern = r'(?:\+1)?[2-9]\d{9}|\+\d{11}'
    
    # Sets to store unique speakers across all files
    all_speakers = set()
    all_phones = set()
    all_names = set()
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Split by timestamp pattern
                timestamp_pattern = r'[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M'
                messages = re.split(timestamp_pattern, content)
                
                for message in messages:
                    # Look for lines that could be speakers (first line after timestamp)
                    lines = message.strip().split('\n')
                    if lines and lines[0]:  # If there are lines and first line isn't empty
                        speaker = lines[0].strip()
                        if speaker and speaker != "Tapbacks:" and not speaker.startswith('Find My:'):
                            all_speakers.add(speaker)
    
    # Categorize all speakers
    all_phones = set(speaker for speaker in all_speakers if re.match(phone_pattern, speaker))
    all_names = all_speakers - all_phones - {'Me'}  # Exclude 'Me' from named speakers
    
    print(f"\nTotal Analysis:")
    print(f"Total unique speakers: {len(all_speakers)}")
    print(f"Unmatched phone numbers: {len(all_phones)}")
    print(f"Named speakers: {len(all_names)}")
    
    # Optionally print the actual phone numbers that need to be replaced
    if all_phones:
        print("\nPhone numbers that still need to be replaced:")
        for phone in sorted(all_phones):
            print(phone)
