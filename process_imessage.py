import argparse
import os
import re
import glob
import sys
import json
import logging
from pathlib import Path
from training_utils import parse_vcard_file, analyze_all_messages, IMessageParser, process_imessage_files, load_and_sort_conversations, create_time_spaced_subset
from IMPersona.loader import MessageLoader
from IMPersona.formatter import LlamaFormatter
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
import tiktoken
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime

# Ask for the user's name at the beginning
persona_name = input("Please enter your name (this will be used as your persona name): ")
print(f"Setting up iMessage processing for persona: {persona_name}")

# File paths and constants
DATA_DIR = 'data'
IMESSAGE_EXPORT_DIR = os.path.join(DATA_DIR, 'imessage_export')
IMESSAGE_PARSED_DIR = os.path.join(DATA_DIR, 'imessage_export_parsed')
CONTACTS_VCF_PATH = os.path.join(DATA_DIR, 'contacts.vcf')
CONTACTS_DICT_PATH = os.path.join(DATA_DIR, 'contacts_dict.json')
CONVERSATION_STORE_PATH = os.path.join(DATA_DIR, 'conversation_store.json')
CONVERSATION_STORE_MEMORY_PATH = os.path.join(DATA_DIR, 'conversation_store_memory.json')
FILTERED_DATASET_PATH = os.path.join(DATA_DIR, 'filtered_imessage_dataset.json')
FULL_DATASET_PATH = os.path.join(DATA_DIR, f'{persona_name}_impersona_imessage_train_full.json')
TOKEN_DISTRIBUTION_PLOT_PATH = os.path.join(DATA_DIR, 'token_distribution.png')
LOG_FILE_PATH = 'imessage_processing.log'

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH)
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Starting iMessage processing for persona: {persona_name}")

parser = argparse.ArgumentParser(description="Process iMessage data for IMPersona training")

parser.add_argument(
    "--message_path", 
    type=str, 
    default=IMESSAGE_EXPORT_DIR,
    help="Path to the imessage_export directory"
)
parser.add_argument(
    "--contact_path", 
    type=str, 
    default=CONTACTS_VCF_PATH,
    help="Path to the contacts.vcf file (optional but recommended)"
)

#########################
# STEP 1: Parse contacts
#########################
logger.info("Step 1: Parsing contacts...")
args = parser.parse_args()

if args.contact_path:
    logger.info(f"Reading contacts from: {args.contact_path}")
    try:
        contacts_dict = parse_vcard_file(args.contact_path)
        logger.info(f"Successfully parsed {len(contacts_dict)} contacts")
        
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(CONTACTS_DICT_PATH, 'w') as f:
            json.dump(contacts_dict, f, indent=2)
        logger.info(f"Saved {len(contacts_dict)} contacts to {CONTACTS_DICT_PATH}")
        
        # Log a sample of contacts
        sample_size = min(5, len(contacts_dict))
        logger.debug(f"Sample of {sample_size} contacts:")
        for i, (phone, name) in enumerate(list(contacts_dict.items())[:sample_size]):
            logger.debug(f"  {i+1}. Phone: {phone} -> Name: {name}")
    except Exception as e:
        logger.error(f"Error parsing contacts: {str(e)}")
        logger.error("Exiting due to critical error in Step 1")
        sys.exit(1)
else:
    logger.warning("No contacts file provided. Proceeding without contact information (Not Recommended).")
    contacts_dict = {}

########################################################
# STEP 2: Filter out low quality spam conversations
########################################################
def count_messages(content):
    # Split by timestamp pattern to count messages
    # This pattern matches timestamps like "Aug 25, 2021  5:17:24 PM"
    messages = re.split(r'\w{3}\s+\d{1,2},\s+\d{4}\s+\d{1,2}:\d{2}:\d{2}\s+[AP]M', content)
    # Filter out empty strings from the split
    return len([m for m in messages if m.strip()])

def contains_me_messages(content):
    # Look for lines starting with "Me"
    return bool(re.search(r'^Me\n', content, re.MULTILINE))

def filter_messages(directory):
    logger.info("Step 2: Filtering low-quality conversation files...")
    
    # Get all .txt files in the directory
    try:
        files = [f for f in os.listdir(directory) if f.endswith('.txt')]
        logger.info(f"Found {len(files)} text files to process")
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return
    except Exception as e:
        logger.error(f"Error accessing directory {directory}: {str(e)}")
        return
    
    kept_count = 0
    deleted_count = 0
    
    for filename in files:
        filepath = os.path.join(directory, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check our filtering conditions
            message_count = count_messages(content)
            has_me = contains_me_messages(content)
            
            # If file should be deleted (20 or fewer messages OR no "Me" messages)
            if message_count <= 20 or not has_me:
                logger.info(f"Deleting {filename}: {message_count} messages, contains 'Me' messages: {has_me}")
                os.remove(filepath)
                deleted_count += 1
            else:
                logger.debug(f"Keeping {filename}: {message_count} messages, contains 'Me' messages: {has_me}")
                kept_count += 1
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
    
    logger.info(f"Filtering complete: kept {kept_count} files, deleted {deleted_count} files")

logger.info(f"Processing iMessage data from: {args.message_path}")
try:
    filter_messages(args.message_path)
except Exception as e:
    logger.error(f"Error during message filtering: {str(e)}")
    logger.error("Exiting due to critical error in Step 2")
    sys.exit(1)

########################################################
# STEP 3: Matching Contacts to Conversation Files
########################################################
logger.info("Step 3: Matching contacts to conversation files...")
try:
    logger.info(f"Processing iMessage files with contact information...")
    process_imessage_files(args.message_path, CONTACTS_DICT_PATH)
    logger.info("Successfully completed contact matching and conversation processing")
except Exception as e:
    logger.error(f"Error during contact matching and conversation processing: {str(e)}")
    logger.error("Exiting due to critical error in Step 3")
    sys.exit(1)

def get_unmatched_numbers():
    # Pattern for timestamp lines
    timestamp_pattern = r"[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2} [AP]M"
    # Pattern for phone numbers (matches +1XXXXXXXXXX or XXXXXXXXXX format)
    phone_pattern = r'(?:\+\d{1,2})?[2-9]\d{9}'
    
    # Dictionary to count occurrences of each phone number
    phone_counts = {}
    
    # Get all message files
    message_files = glob.glob(os.path.join(args.message_path, '*.txt'))
    
    for file_path in message_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            # If we find a timestamp, check the next line for a phone number
            if re.match(timestamp_pattern, line.strip()):
                if i + 1 < len(lines):  # Make sure we have a next line
                    next_line = lines[i + 1].strip()
                    if re.match(phone_pattern, next_line):
                        # Increment count for this phone number
                        phone_counts[next_line] = phone_counts.get(next_line, 0) + 1
    
    # Filter to only include phone numbers that appear at least 15 times
    MIN_MESSAGE_THRESHOLD = 50
    frequent_numbers = {phone: count for phone, count in phone_counts.items() 
                        if count >= MIN_MESSAGE_THRESHOLD}
    
    # Print results
    logger.info(f"Found {len(phone_counts)} total unmatched phone numbers")
    logger.info(f"Found {len(frequent_numbers)} frequent unmatched phone numbers (≥{MIN_MESSAGE_THRESHOLD} messages)")
    
    # Sort by frequency (most frequent first)
    sorted_numbers = sorted(frequent_numbers.items(), key=lambda x: x[1], reverse=True)
    
    # If we have frequent unmatched numbers, offer to add contact names
    if sorted_numbers:
        # Log the most frequent numbers
        logger.info("Most frequent unmatched numbers:")
        for phone, count in sorted_numbers[:10]:  # Show top 10
            logger.info(f"  {phone}: {count} messages")
            
        # Extract just the phone numbers for the add_contact_names function
        unmatched_numbers = [phone for phone, _ in sorted_numbers]
        add_contact_names(unmatched_numbers)
    else:
        logger.info("No frequent unmatched numbers found.")

def add_contact_names(unmatched_numbers):
    """Allow user to manually add contact names for unmatched phone numbers."""
    logger.info("You can now add contact names for frequently occurring phone numbers.")
    logger.info("For each number, enter a name or press Enter to skip.")
    logger.info("Enter 'q' at any time to quit this process and continue with the data processing.")
    
    # Load existing contacts dictionary if it exists
    if os.path.exists(CONTACTS_DICT_PATH):
        with open(CONTACTS_DICT_PATH, 'r') as f:
            contacts_dict = json.load(f)
    else:
        contacts_dict = {}
    
    # Track if any new contacts were added
    added_contacts = False
    
    # Process each unmatched number
    for number in unmatched_numbers:
        # Skip if already in contacts
        if number in contacts_dict:
            logger.info(f"Number {number} already mapped to {contacts_dict[number]}")
            continue
            
        # Ask for contact name
        name_input = input(f"Enter name for {number} (or press Enter to skip): ")
        
        # Check if user wants to quit
        if name_input.lower() == 'q':
            logger.info("Quitting contact name entry.")
            break
            
        # If user entered a name, add it to the dictionary
        if name_input.strip():
            contacts_dict[number] = name_input.strip()
            logger.info(f"Added: {number} -> {name_input.strip()}")
            added_contacts = True
    
    # Save updated contacts dictionary if changes were made
    if added_contacts:
        with open(CONTACTS_DICT_PATH, 'w') as f:
            json.dump(contacts_dict, f, indent=2)
        logger.info(f"Updated contacts saved to {CONTACTS_DICT_PATH}")
        
        # Ask if user wants to reprocess files with new contacts
        reprocess = input("Do you want to reprocess message files with the updated contacts? (y/n): ")
        if reprocess.lower() == 'y':
            try:
                logger.info("Reprocessing message files with updated contacts...")
                process_imessage_files(args.message_path, CONTACTS_DICT_PATH)
                logger.info("Successfully reprocessed files with updated contacts")
            except Exception as e:
                logger.error(f"Error during reprocessing: {str(e)}")
    else:
        logger.info("No new contacts were added.")

# Call the function at the end
get_unmatched_numbers()

logger.info("Printing analysis of contact status..")
analyze_all_messages(args.message_path)

########################################################
# STEP 4: Parsing the messages
########################################################
logger.info("Step 4: Parsing all messages...")
parser = IMessageParser(combine_consecutive=True, delimiter='<|msg|>')

# Create output directory if it doesn't exist
os.makedirs(IMESSAGE_PARSED_DIR, exist_ok=True)
logger.info(f"Created output directory: {IMESSAGE_PARSED_DIR}")

# Get all .txt files in the export directory
message_files = glob.glob(f'{args.message_path}/*.txt')
logger.info(f"Found {len(message_files)} message files to parse")

# Process each file
successful_files = 0
failed_files = 0

for file_path in message_files:
    # Get base filename without extension for output file naming
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_path = f'{IMESSAGE_PARSED_DIR}/{base_name}.json'
    
    try:
        # Parse conversations from the file
        logger.debug(f"Parsing file: {file_path}")
        conversations = parser.parse(file_path)
        
        # Save the parsed conversations
        parser.save_conversations(conversations, output_path)
        
        # Log progress
        logger.info(f"Parsed {len(conversations)} conversations from {base_name}")
        successful_files += 1
    except Exception as e:
        logger.error(f"Error parsing file {file_path}: {str(e)}")
        failed_files += 1

logger.info(f"Parsing complete: Successfully processed {successful_files} files, Failed: {failed_files} files")
logger.info(f"All conversations have been parsed and saved to {IMESSAGE_PARSED_DIR}/")

########################################################
# STEP 5: Creating a Conversation Store for Retrieval
########################################################
logger.info("Step 5: Creating a conversation store for retrieval...")

try:
    logger.info(f"Loading and sorting conversations from {IMESSAGE_PARSED_DIR}...")
    recent_convos = load_and_sort_conversations(
        IMESSAGE_PARSED_DIR, 
        num_conversations=1000, 
        min_messages=15,
        max_messages=75,
    )
    logger.info(f"Successfully loaded {len(recent_convos)} conversations")
    
    with open(CONVERSATION_STORE_PATH, "w") as f:
        json.dump(recent_convos, f, indent=2)
    logger.info(f"Saved conversation store to {CONVERSATION_STORE_PATH}")
except Exception as e:
    logger.error(f"Error creating conversation store: {str(e)}")
    logger.error("Exiting due to critical error in Step 5")
    sys.exit(1)

logger.info("Step 5.5: Creating a Conversation Store for Memory Module...")
try:
    logger.info(f"Loading and sorting conversations from {IMESSAGE_PARSED_DIR}...")
    recent_convos = load_and_sort_conversations(
        IMESSAGE_PARSED_DIR, 
        num_conversations=8000,
        min_messages=5,
        max_messages=500
    )
    logger.info(f"Successfully loaded {len(recent_convos)} conversations")
    
    with open(CONVERSATION_STORE_MEMORY_PATH, "w") as f:
        json.dump(recent_convos, f, indent=2)
    logger.info(f"Saved conversation store for memory to {CONVERSATION_STORE_MEMORY_PATH}")
except Exception as e:
    logger.error(f"Error creating conversation store: {str(e)}")
    logger.error("Exiting due to critical error in Step 5")
    sys.exit(1)

########################################################
# STEP 6: Creating the training set
########################################################    
logger.info("Step 6: Creating the training set...")

try:
    # Load and prepare dataset
    logger.info(f"Loading messages from {IMESSAGE_PARSED_DIR}...")
    loader = MessageLoader(conversations_dir=IMESSAGE_PARSED_DIR, formatter=LlamaFormatter(), persona_name=persona_name)
    full_dataset_imessage = loader.generate_full_dataset(conversation_buffer=0)
    logger.info(f"Loaded {len(full_dataset_imessage)} raw conversation examples")

    # Sort by timestamp
    logger.info("Sorting dataset by timestamp...")
    full_dataset_imessage = full_dataset_imessage.sort("timestamp")
    logger.info("Dataset sorted successfully")

    # Download stopwords if not already downloaded
    try:
        logger.info("Checking for NLTK stopwords...")
        nltk.data.find('corpora/stopwords')
        logger.info("NLTK stopwords already downloaded")
    except LookupError:
        logger.info("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        logger.info("NLTK stopwords downloaded successfully")

    STOP_WORDS = set(stopwords.words('english'))
    # Add additional common words you don't want to count
    STOP_WORDS.update(['yeah', 'ok', 'okay', 'lol', 'like', 'um', 'uh', 'haha', 'im', 'i', 'u'])
    logger.info(f"Using {len(STOP_WORDS)} stop words for filtering")

    def contains_url(text):
        # Common URL patterns
        url_patterns = [
            'http://', 'https://', 'www.', '.com', '.org', '.edu', '.gov',
            '.net', 'maps.google', 'discord.com', 'tickets.princeton'
        ]
        return any(pattern in text.lower() for pattern in url_patterns)

    def has_excessive_repetition(text, max_repeats=15):
        # Convert to lowercase and split into words
        words = text.lower().split()
        # Count occurrences of non-stop words
        word_counts = Counter(word for word in words if word not in STOP_WORDS)
        # Check if any word appears more than max_repeats times
        return any(count > max_repeats for count in word_counts.values())

    def get_unique_speakers(text):
        """Extract unique speakers from conversation text."""
        speakers = set()
        # Look for patterns like "[timestamp] Name:" in the text
        lines = text.split('\n')
        for line in lines:
            if '] ' in line and ': ' in line:
                # Extract the speaker name between "] " and ": "
                speaker = line.split('] ')[1].split(': ')[0]
                if speaker != 'Me':  # Don't count "Me" as a separate speaker
                    speakers.add(speaker)
        return speakers

    def has_long_messages(text, max_words=50):
        """Check if any individual message in the text exceeds max_words."""
        # Split by timestamp pattern to get individual messages
        timestamp_pattern = r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]'
        messages = re.split(timestamp_pattern, text)
        
        # For each message, check if any part (split by <|msg|>) is too long
        for message in messages:
            if not message.strip():  # Skip empty messages
                continue
            # Split into individual messages if there are consecutive messages
            sub_messages = message.split('<|msg|>')
            for sub_message in sub_messages:
                # Remove speaker prefix (everything before first ':')
                if ':' in sub_message:
                    content = sub_message.split(':', 1)[1]
                else:
                    content = sub_message
                
                # Count words in the message content
                word_count = len(content.split())
                if word_count > max_words:
                    return True
        return False

    def is_emoji_only(text):
        """Check if a text contains only emojis and whitespace."""
        # Unicode ranges for emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251" 
            "]+|"
            "[\U0001F1E6-\U0001F1FF]{2}|"  # country flags
            "[\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\u3030\ufe0f]"
            "+"
        )
        
        # Remove emojis and whitespace
        text_without_emoji = emoji_pattern.sub('', text.strip())
        # If nothing remains, it was only emojis (and possibly whitespace)
        return len(text_without_emoji) == 0

    def count_total_messages(text):
        """Count total number of messages in a conversation by counting timestamps."""
        # Look for timestamp pattern [YYYY-MM-DD HH:MM:SS]
        timestamp_pattern = r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\]'
        return len(re.findall(timestamp_pattern, text))

    def count_consecutive_messages(text):
        # Count the maximum number of consecutive <|msg|> delimiters in any message
        parts = text.split('\n')
        max_consecutive = 0
        for part in parts:
            delimiter_count = part.count('<|msg|>')
            max_consecutive = max(max_consecutive, delimiter_count)
        return max_consecutive

    # Define strings to filter out (case-insensitive)
    filtered_strings = [
        'GamePigeon message:',
        'Library/Messages/', 
    ]
    
    logger.info(f"Using {len(filtered_strings)} filtered strings for content filtering")

    # Track filtering statistics
    filter_stats = {
        "filtered_strings": 0,
        "too_short": 0,
        "too_many_consecutive_msgs_input": 0,
        "too_many_consecutive_msgs_output": 0,
        "contains_url": 0,
        "excessive_repetition": 0,
        "too_many_speakers": 0,
        "emoji_only": 0,
        "too_many_messages": 0,
        "long_messages_input": 0,
        "long_messages_output": 0,
        "too_little_messages_input": 0
    }

    logger.info("Applying filters to dataset...")
    # Filter the dataset
    def filter_function(example):
        # Check each filter condition and update stats
        if any(bad_word.lower() in example['input'].lower() or 
               bad_word.lower() in example['output'].lower() 
               for bad_word in filtered_strings):
            filter_stats["filtered_strings"] += 1
            return False
            
        if len(example['output'].split()) <= 2:
            filter_stats["too_short"] += 1
            return False
            
        if count_consecutive_messages(example['input']) > 20:
            filter_stats["too_many_consecutive_msgs_input"] += 1
            return False
            
        if count_consecutive_messages(example['output']) > 20:
            filter_stats["too_many_consecutive_msgs_output"] += 1
            return False
            
        if contains_url(example['output']):
            filter_stats["contains_url"] += 1
            return False
            
        if has_excessive_repetition(example['output']):
            filter_stats["excessive_repetition"] += 1
            return False
            
        if len(get_unique_speakers(example['input'])) > 1:
            filter_stats["too_many_speakers"] += 1
            return False
            
        if is_emoji_only(example['output']):
            filter_stats["emoji_only"] += 1
            return False
            
        if count_total_messages(example['input']) > 35:
            filter_stats["too_many_messages"] += 1
            return False
            
        if has_long_messages(example['input']):
            filter_stats["long_messages_input"] += 1
            return False
            
        if has_long_messages(example['output']):
            filter_stats["long_messages_output"] += 1
            return False
        
        if count_total_messages(example['input']) <= 3:
            filter_stats["too_little_messages_input"] += 1
            return False
            
        return True

    filtered_dataset = full_dataset_imessage.filter(filter_function)

    # Log filtering results
    logger.info(f"Original dataset size: {len(full_dataset_imessage)}")
    logger.info(f"Filtered dataset size: {len(filtered_dataset)}")
    logger.info(f"Removed {len(full_dataset_imessage) - len(filtered_dataset)} examples")
    
    # Log detailed filter statistics
    logger.info("Filter statistics:")
    for filter_name, count in filter_stats.items():
        logger.info(f"  - {filter_name}: {count} examples removed")
    
    # Save filtered dataset
    filtered_dataset.to_json(FILTERED_DATASET_PATH)
    logger.info(f"Filtered dataset saved to {FILTERED_DATASET_PATH}")
    
    full_dataset_imessage = filtered_dataset
    # Convert dataset to list of dictionaries
    logger.info("Converting dataset to list format...")
    dataset_list = [
        {
            'instruction': item['instruction'],
            'input': item['input'],
            'output': item['output'],
            'timestamp': item['timestamp']
        }
        for item in full_dataset_imessage
    ]
    logger.info(f"Converted {len(dataset_list)} examples to list format")
    # Save to JSON file
    with open(FULL_DATASET_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset_list, f, indent=2, ensure_ascii=False)
    
    # Create and save Together API compatible format for the full dataset
    together_full_filename = f'{persona_name}_impersona_imessage_train_full_together_format.jsonl'
    together_full_filepath = os.path.join(DATA_DIR, together_full_filename)
    
    logger.info(f"Creating Together API compatible format for full dataset with {len(dataset_list)} examples...")
    with open(together_full_filepath, 'w', encoding='utf-8') as f:
        for item in dataset_list:
            together_format = {
                "prompt": item['input'],
                "completion": item['output']
            }
            f.write(json.dumps(together_format) + '\n')
    
    logger.info(f"Saved Together API compatible format for full dataset to {together_full_filepath}")
    
    # Creating the subsets in order from smallest to largest:
    logger.info("Creating time-spaced subsets of different sizes (smallest to largest)...")
    total_examples = len(dataset_list)

    # Dictionary to store created subsets
    subsets = {}

    # Function to create subset if enough data is available
    def create_subset_if_possible(name, size, time_gap_hours=None):
        if total_examples < size:
            logger.warning(f"Cannot create {name} subset: requires {size} examples but only {total_examples} are available")
            return None
        
        if time_gap_hours:
            logger.info(f"Creating {size}-example subset ({time_gap_hours} hour minimum gap)...")
            subset = create_time_spaced_subset(dataset_list, size, min_time_gap=time_gap_hours*60*60)
            logger.info(f"Created {name} subset with {len(subset)} examples ({time_gap_hours} hour gap)")
        else:
            logger.info(f"Creating {size}-example subset (most recent examples)...")
            subset = dataset_list[-size:]
            logger.info(f"Created {name} subset with {len(subset)} examples")
        
        return subset

    # Create subsets from smallest to largest
    subsets['B25'] = create_subset_if_possible('B25', 25, time_gap_hours=24)
    subsets['B50'] = create_subset_if_possible('B50', 50, time_gap_hours=12)
    subsets['B100'] = create_subset_if_possible('B100', 100, time_gap_hours=12)
    subsets['B250'] = create_subset_if_possible('B250', 250, time_gap_hours=4)
    subsets['B500'] = create_subset_if_possible('B500', 500, time_gap_hours=2)
    subsets['B1k'] = create_subset_if_possible('B1k', 1000, time_gap_hours=1)
    subsets['B2k'] = create_subset_if_possible('B2k', 2000)
    subsets['B4k'] = create_subset_if_possible('B4k', 4000)
    subsets['B8k'] = create_subset_if_possible('B8k', 8000)
    
    # Create validation set of 200 examples that aren't in any training subset
    logger.info("Creating validation set of 200 examples...")
    
    # First, identify all examples used in training subsets
    training_examples_ids = set()
    for size_key in subsets:
        if subsets[size_key] is not None:
            # Use timestamps as unique identifiers for examples
            for example in subsets[size_key]:
                training_examples_ids.add(example['timestamp'])
    
    logger.info(f"Identified {len(training_examples_ids)} unique examples used in training subsets")
    
    # Find examples that aren't in any training subset
    validation_candidates = []
    for example in dataset_list:
        if example['timestamp'] not in training_examples_ids:
            validation_candidates.append(example)
    
    logger.info(f"Found {len(validation_candidates)} candidate examples for validation set")
    
    # Create validation set with time spacing if possible
    if len(validation_candidates) >= 200:
        validation_set = create_time_spaced_subset(validation_candidates, 200, min_time_gap=3600)  # 1 hour gap
        logger.info(f"Created validation set with {len(validation_set)} examples")
        
        # Save validation set
        validation_filename = f'impersona_imessage_validation.json'
        validation_filepath = os.path.join(DATA_DIR, validation_filename)
        with open(validation_filepath, 'w', encoding='utf-8') as f:
            json.dump(validation_set, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved validation set to {validation_filepath}")
        
        # Create and save Together API compatible format for validation set
        together_val_filename = f'{persona_name}_impersona_imessage_validation_together_format.jsonl'
        together_val_filepath = os.path.join(DATA_DIR, together_val_filename)
        
        with open(together_val_filepath, 'w', encoding='utf-8') as f:
            for item in validation_set:
                together_format = {
                    "prompt": item['input'],
                    "completion": item['output']
                }
                f.write(json.dumps(together_format) + '\n')
        
        logger.info(f"Saved Together API compatible format for validation set to {together_val_filepath}")
    else:
        logger.warning(f"Not enough unique examples for validation set. Need 200, but only found {len(validation_candidates)}")
        if validation_candidates:
            logger.info("Saving all available validation candidates instead")
            validation_set = validation_candidates
            
            validation_filename = f'impersona_imessage_validation_partial.json'
            validation_filepath = os.path.join(DATA_DIR, validation_filename)
            with open(validation_filepath, 'w', encoding='utf-8') as f:
                json.dump(validation_set, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved partial validation set with {len(validation_set)} examples to {validation_filepath}")
            
            # Create and save Together API compatible format for partial validation set
            together_val_filename = f'{persona_name}_impersona_imessage_validation_partial_together_format.jsonl'
            together_val_filepath = os.path.join(DATA_DIR, together_val_filename)
            
            with open(together_val_filepath, 'w', encoding='utf-8') as f:
                for item in validation_set:
                    together_format = {
                        "prompt": item['input'],
                        "completion": item['output']
                    }
                    f.write(json.dumps(together_format) + '\n')
            
            logger.info(f"Saved Together API compatible format for partial validation set to {together_val_filepath}")
    
    # Reverse all subsets to have earliest messages first
    logger.info("Reversing all subsets to have earliest messages first...")
    for size_key in subsets:
        if subsets[size_key] is not None:
            subsets[size_key].reverse()
            logger.info(f"Reversed {size_key} subset to chronological order (earliest first)")

    # Save all created subsets
    logger.info("Saving dataset subsets to data directory...")
    os.makedirs(DATA_DIR, exist_ok=True)

    # Define duplication factors for each subset size
    duplication_factors = {
        'B25': 4,    # Duplicate small datasets more times
        'B50': 2,
        'B100': 2,
        'B250': 1,
        'B500': 1,
        'B1k': 1,
        'B2k': 1,
        'B4k': 1,
        'B8k': 1     # Duplicate large datasets fewer times
    }
    logger.info(f"Using custom duplication factors: {duplication_factors}")

    # Count how many subsets were successfully created
    successful_subsets = 0

    # Save subsets in order from smallest to largest
    for size_key in ['B25', 'B50', 'B100', 'B250', 'B500', 'B1k', 'B2k', 'B4k', 'B8k']:
        if subsets[size_key] is not None:
            # Save original format
            filename = f'impersona_imessage_0buffer_{size_key}.json'
            filepath = os.path.join(DATA_DIR, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(subsets[size_key], f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(subsets[size_key])} examples to {filepath}")
            
            # Create and save Together API compatible format (JSONL)
            together_filename = f'{persona_name}_impersona_imessage_0buffer_{size_key}_together_format.jsonl'
            together_filepath = os.path.join(DATA_DIR, together_filename)
            
            # Get duplication factor for this subset size
            duplication_factor = duplication_factors.get(size_key, 1)  # Default to 4 if not specified
            logger.info(f"Duplicating {size_key} examples {duplication_factor} times")
            
            with open(together_filepath, 'w', encoding='utf-8') as f:
                for item in subsets[size_key]:
                    together_format = {
                        "prompt": item['input'],
                        "completion": item['output']
                    }
                    # Write the same example duplication_factor times
                    for _ in range(duplication_factor):
                        f.write(json.dumps(together_format) + '\n')
            
            logger.info(f"Saved Together API compatible format with {len(subsets[size_key]) * duplication_factor} examples (original: {len(subsets[size_key])}, duplicated {duplication_factor}x) to {together_filepath}")
            successful_subsets += 1

    logger.info(f"Successfully created and saved {successful_subsets} dataset subsets (in both formats)")
    logger.info(f"Saved full dataset with {len(dataset_list)} conversations to {FULL_DATASET_PATH}")
    logger.info("Step 6 completed successfully")

except Exception as e:
    logger.error(f"Error in Step 6 (Creating training set): {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    logger.error("Exiting due to critical error in Step 6")
    sys.exit(1)


########################################################
# STEP 7: Visualizing the dataset:
########################################################
logger.info("Step 7: Visualizing the dataset and analyzing token distribution...")

def analyze_token_distribution(dataset_path, encoding_name="cl100k_base"):
    logger.info(f"Analyzing token distribution for dataset: {dataset_path}")
    logger.info(f"Using encoding: {encoding_name}")
    
    # Initialize tokenizer
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        logger.info(f"Successfully initialized {encoding_name} tokenizer")
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {str(e)}")
        return
    
    # Load the dataset
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logger.info(f"Successfully loaded dataset with {len(dataset)} examples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        return
    
    # Lists to store token counts
    input_tokens = []
    output_tokens = []
    total_tokens = []
    
    logger.info("Tokenizing examples...")
    for i, example in enumerate(dataset):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(dataset)} examples")
            
        input_len = len(encoding.encode(example['input']))
        output_len = len(encoding.encode(example['output']))
        
        input_tokens.append(input_len)
        output_tokens.append(output_len)
        total_tokens.append(input_len + output_len)
    
    # Calculate statistics
    logger.info("Calculating token statistics...")
    stats = {
        "dataset_name": os.path.basename(dataset_path),
        "num_examples": len(dataset),
        "total_tokens": sum(total_tokens),
        "input_tokens": {
            "total": sum(input_tokens),
            "mean": np.mean(input_tokens),
            "median": np.median(input_tokens),
            "max": max(input_tokens),
            "min": min(input_tokens),
            "95percentile": np.percentile(input_tokens, 95),
            "std_dev": np.std(input_tokens)
        },
        "output_tokens": {
            "total": sum(output_tokens),
            "mean": np.mean(output_tokens),
            "median": np.median(output_tokens),
            "max": max(output_tokens),
            "min": min(output_tokens),
            "95percentile": np.percentile(output_tokens, 95),
            "std_dev": np.std(output_tokens)
        },
        "combined_tokens": {
            "total": sum(total_tokens),
            "mean": np.mean(total_tokens),
            "median": np.median(total_tokens),
            "max": max(total_tokens),
            "min": min(total_tokens),
            "95percentile": np.percentile(total_tokens, 95),
            "std_dev": np.std(total_tokens)
        }
    }
    
    # Log statistics
    logger.info(f"Token Length Statistics:")
    logger.info(f"Dataset: {stats['dataset_name']}")
    logger.info(f"Number of examples: {stats['num_examples']}")
    logger.info(f"Total tokens: {stats['total_tokens']}")
    
    logger.info(f"\nInput Tokens:")
    logger.info(f"Total: {stats['input_tokens']['total']}")
    logger.info(f"Mean: {stats['input_tokens']['mean']:.1f}")
    logger.info(f"Median: {stats['input_tokens']['median']:.1f}")
    logger.info(f"Max: {stats['input_tokens']['max']}")
    logger.info(f"Min: {stats['input_tokens']['min']}")
    logger.info(f"95th percentile: {stats['input_tokens']['95percentile']:.1f}")
    logger.info(f"Standard deviation: {stats['input_tokens']['std_dev']:.1f}")
    
    logger.info(f"\nOutput Tokens:")
    logger.info(f"Total: {stats['output_tokens']['total']}")
    logger.info(f"Mean: {stats['output_tokens']['mean']:.1f}")
    logger.info(f"Median: {stats['output_tokens']['median']:.1f}")
    logger.info(f"Max: {stats['output_tokens']['max']}")
    logger.info(f"Min: {stats['output_tokens']['min']}")
    logger.info(f"95th percentile: {stats['output_tokens']['95percentile']:.1f}")
    logger.info(f"Standard deviation: {stats['output_tokens']['std_dev']:.1f}")
    
    # Print for console output as well
    print(f"Token Length Statistics:")
    print(f"Dataset: {stats['dataset_name']}")
    print(f"Number of examples: {stats['num_examples']}")
    print(f"Total tokens: {stats['total_tokens']}")
    
    print(f"\nInput Tokens:")
    print(f"Total: {stats['input_tokens']['total']}")
    print(f"Mean: {stats['input_tokens']['mean']:.1f}")
    print(f"Median: {stats['input_tokens']['median']:.1f}")
    print(f"Max: {stats['input_tokens']['max']}")
    print(f"Min: {stats['input_tokens']['min']}")
    print(f"95th percentile: {stats['input_tokens']['95percentile']:.1f}")
    print(f"Standard deviation: {stats['input_tokens']['std_dev']:.1f}")
    
    print(f"\nOutput Tokens:")
    print(f"Total: {stats['output_tokens']['total']}")
    print(f"Mean: {stats['output_tokens']['mean']:.1f}")
    print(f"Median: {stats['output_tokens']['median']:.1f}")
    print(f"Max: {stats['output_tokens']['max']}")
    print(f"Min: {stats['output_tokens']['min']}")
    print(f"95th percentile: {stats['output_tokens']['95percentile']:.1f}")
    print(f"Standard deviation: {stats['output_tokens']['std_dev']:.1f}")
    
    # Create histogram
    logger.info("Generating token distribution histograms...")
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot histograms
        plt.subplot(1, 2, 1)
        plt.hist(input_tokens, bins=50, alpha=0.7)
        plt.title('Distribution of Input Tokens')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        
        plt.subplot(1, 2, 2)
        plt.hist(output_tokens, bins=50, alpha=0.7)
        plt.title('Distribution of Output Tokens')
        plt.xlabel('Token Count')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(TOKEN_DISTRIBUTION_PLOT_PATH)
        logger.info(f"Saved token distribution plot to {TOKEN_DISTRIBUTION_PLOT_PATH}")
        
        # Show the plot if in interactive environment
        plt.show()
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
    
    # Save statistics to a JSON file
    stats_file_path = os.path.join(DATA_DIR, f"{os.path.splitext(os.path.basename(dataset_path))[0]}_stats.json")
    try:
        with open(stats_file_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved dataset statistics to {stats_file_path}")
    except Exception as e:
        logger.error(f"Error saving statistics to file: {str(e)}")
    
    # Also save a human-readable version
    readable_stats_path = os.path.join(DATA_DIR, f"{os.path.splitext(os.path.basename(dataset_path))[0]}_stats.txt")
    try:
        with open(readable_stats_path, 'w') as f:
            f.write(f"Dataset Statistics for {stats['dataset_name']}\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"General Statistics:\n")
            f.write(f"  Number of examples: {stats['num_examples']}\n")
            f.write(f"  Total tokens: {stats['total_tokens']}\n\n")
            
            f.write(f"Input Tokens:\n")
            f.write(f"  Total: {stats['input_tokens']['total']}\n")
            f.write(f"  Mean: {stats['input_tokens']['mean']:.1f}\n")
            f.write(f"  Median: {stats['input_tokens']['median']:.1f}\n")
            f.write(f"  Max: {stats['input_tokens']['max']}\n")
            f.write(f"  Min: {stats['input_tokens']['min']}\n")
            f.write(f"  95th percentile: {stats['input_tokens']['95percentile']:.1f}\n")
            f.write(f"  Standard deviation: {stats['input_tokens']['std_dev']:.1f}\n\n")
            
            f.write(f"Output Tokens:\n")
            f.write(f"  Total: {stats['output_tokens']['total']}\n")
            f.write(f"  Mean: {stats['output_tokens']['mean']:.1f}\n")
            f.write(f"  Median: {stats['output_tokens']['median']:.1f}\n")
            f.write(f"  Max: {stats['output_tokens']['max']}\n")
            f.write(f"  Min: {stats['output_tokens']['min']}\n")
            f.write(f"  95th percentile: {stats['output_tokens']['95percentile']:.1f}\n")
            f.write(f"  Standard deviation: {stats['output_tokens']['std_dev']:.1f}\n\n")
            
            f.write(f"Combined Tokens (Input + Output):\n")
            f.write(f"  Total: {stats['combined_tokens']['total']}\n")
            f.write(f"  Mean: {stats['combined_tokens']['mean']:.1f}\n")
            f.write(f"  Median: {stats['combined_tokens']['median']:.1f}\n")
            f.write(f"  Max: {stats['combined_tokens']['max']}\n")
            f.write(f"  Min: {stats['combined_tokens']['min']}\n")
            f.write(f"  95th percentile: {stats['combined_tokens']['95percentile']:.1f}\n")
            f.write(f"  Standard deviation: {stats['combined_tokens']['std_dev']:.1f}\n")
            
        logger.info(f"Saved human-readable statistics to {readable_stats_path}")
    except Exception as e:
        logger.error(f"Error saving human-readable statistics: {str(e)}")
    
    return stats

def analyze_all_datasets():
    """Analyze all dataset files in the data directory and create a summary file."""
    logger.info("Analyzing all dataset files...")
    
    # Find all dataset files
    dataset_files = []
    for file in os.listdir(DATA_DIR):
        if file.endswith('.json') and not file.endswith('_stats.json') and not file in ['contacts_dict.json', 'conversation_store.json', 'conversation_store_memory.json']:
            dataset_files.append(os.path.join(DATA_DIR, file))
    
    logger.info(f"Found {len(dataset_files)} dataset files to analyze")
    
    # Analyze each dataset
    all_stats = []
    for dataset_path in dataset_files:
        try:
            logger.info(f"Analyzing dataset: {dataset_path}")
            stats = analyze_token_distribution(dataset_path)
            if stats:
                all_stats.append(stats)
        except Exception as e:
            logger.error(f"Error analyzing dataset {dataset_path}: {str(e)}")
    
    # Create a summary file
    if all_stats:
        summary_path = os.path.join(DATA_DIR, "dataset_stats_summary.txt")
        try:
            with open(summary_path, 'w') as f:
                f.write(f"IMPersona Dataset Statistics Summary\n")
                f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Table header
                f.write(f"{'Dataset':<40} {'Examples':<10} {'Total Tokens':<15} {'Avg Input':<12} {'Avg Output':<12}\n")
                f.write(f"{'-'*40} {'-'*10} {'-'*15} {'-'*12} {'-'*12}\n")
                
                # Sort by number of examples
                all_stats.sort(key=lambda x: x['num_examples'])
                
                # Table rows
                for stats in all_stats:
                    f.write(f"{stats['dataset_name']:<40} {stats['num_examples']:<10} {stats['total_tokens']:,<15} {stats['input_tokens']['mean']:<12.1f} {stats['output_tokens']['mean']:<12.1f}\n")
                
                # Add grand totals
                total_examples = sum(s['num_examples'] for s in all_stats)
                total_tokens = sum(s['total_tokens'] for s in all_stats)
                f.write(f"{'-'*40} {'-'*10} {'-'*15} {'-'*12} {'-'*12}\n")
                f.write(f"{'TOTAL':<40} {total_examples:<10} {total_tokens:,<15}\n")
                
            logger.info(f"Saved dataset statistics summary to {summary_path}")
        except Exception as e:
            logger.error(f"Error creating summary file: {str(e)}")

try:
    # Analyze the full dataset
    logger.info("Starting token distribution analysis...")
    analyze_token_distribution(FULL_DATASET_PATH)
    
    # Analyze all datasets and create summary
    analyze_all_datasets()
    
    logger.info("Token distribution analysis completed successfully")
except Exception as e:
    logger.error(f"Error during token distribution analysis: {str(e)}")
    import traceback
    logger.error(traceback.format_exc())
    logger.error("Exiting due to critical error in Step 7")
    sys.exit(1)

logger.info("iMessage processing pipeline completed successfully!")















