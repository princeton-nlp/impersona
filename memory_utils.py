import os
import json
from openai import AsyncOpenAI
import asyncio
import re


async def extract_personal_attributes_async(conversations, model_name="gpt-4o-mini", person_name="Ben", batch_size=20):
    """
    Process conversations in batches to extract personal attributes/facts about a specific person using async API calls.
    
    Args:
        conversations: List of conversation dictionaries
        batch_size: Number of conversations to process in each batch
        person_name: Name of the person to extract attributes about
        
    Returns:
        List of dictionaries containing attributes, their citations, and timestamps
    """
    
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Format conversations for the prompt with clear conversation numbering
    formatted_convos = []
    for idx, convo in enumerate(conversations):
        formatted_convo = f"CONVERSATION {idx+1} START:\n"
        for message in convo:
            speaker = message.get('speaker', 'unknown')
            content = message.get('content', '')
            timestamp = message.get('timestamp', '')
            formatted_convo += f"{timestamp} {speaker}: {content}\n"
        formatted_convo += f"CONVERSATION {idx+1} END\n"
        formatted_convos.append(formatted_convo)
    
    prompt = f"""
    You are an expert at analyzing conversations and extracting personal attributes and facts about people.
    
    I'll provide you with {len(conversations)} conversations involving {person_name}. Your task is to:
    
    1. Extract specific attributes and facts about {person_name} from these conversations. Do not extract traits, opinions on personality, etc.
    2. Focus on persistent attributes (like hobbies, background, relationships) rather than one-time events. 
    
    For example: if Ben is currently in a research project in AI, you should extract that "Ben does research in AI.", not "Ben is currently in a research project in AI".

    3. For each attribute, cite a specific message from the conversations where this information appears.
    4. Include the timestamp of when this information was mentioned.
    5. Format each attribute as a clear, concise statement about {person_name}.
    6. IMPORTANT: Always include the exact conversation number in your citation (e.g., "Conversation 2").
    
    Examples of good attributes:
    - "{person_name} went to high school in Morgantown, West Virginia at Morgantown High School." (Citation: Conversation 2, {person_name}: "I grew up in Morgantown and went to high school there at Morgantown High School.", Timestamp: "2023-05-15 14:30")
    - "{person_name} has a dog named Ellie." (Citation: Conversation 1, {person_name}: "I need to take Ellie, my dog, for a walk later.", Timestamp: "2023-06-02 09:15")
    - "{person_name} likes to create music." (Citation: Conversation 3, Friend: "How's your music production going? I need to hear some banger songs again.", Timestamp: "2023-04-28 18:45")
    - "{person_name} played tennis on his high school team." (Citation: Conversation 1, {person_name}: "I miss playing tennis competitively like I did in high school.", Timestamp: "2023-05-10 11:20")
    
    Please format your response EXACTLY as follows, with each attribute having these three lines:
    
    - Attribute: {person_name} went to high school in Morgantown, West Virginia.
    - Citation: Conversation 2, {person_name}: "I grew up in Morgantown and went to high school there."
    - Timestamp: 2023-05-15 14:30
    
    - Attribute: {person_name} has a dog named Ellie.
    - Citation: Conversation 1, {person_name}: "I need to take Ellie, my dog, for a walk later."
    - Timestamp: 2023-06-02 09:15
    
    Here are the conversations:
    
    {'''
'''.join(formatted_convos)}
    """
    
    # Call the model
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You extract personal attributes from conversations with citations and timestamps."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the response using a custom parser instead of JSON
    return parse_attributes_from_text(response.choices[0].message.content)

def parse_attributes_from_text(text):
    """
    Parse attributes from the model's text response.
    
    Args:
        text: The text response from the model
    
    Returns:
        List of dictionaries containing attributes, their citations, and timestamps
    """
    attributes = []
    current_attribute = {}
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Remove leading dash or bullet point if present
        if line.startswith('- '):
            line = line[2:]
        elif line.startswith('• '):
            line = line[2:]
            
        # Check for attribute, citation, or timestamp
        if line.lower().startswith('attribute:'):
            # If we already have an attribute in progress, save it
            if current_attribute and 'attribute' in current_attribute:
                attributes.append(current_attribute)
                current_attribute = {}
            
            current_attribute['attribute'] = line[len('attribute:'):].strip()
        
        elif line.lower().startswith('citation:'):
            if 'attribute' in current_attribute:  # Only add if we have an attribute
                citation_text = line[len('citation:'):].strip()
                current_attribute['citation'] = citation_text
                
                # Extract conversation number from citation
                conversation_match = re.search(r'conversation\s+(\d+)', citation_text.lower())
                if conversation_match:
                    current_attribute['conversation_number'] = int(conversation_match.group(1))
                else:
                    current_attribute['conversation_number'] = None
        
        elif line.lower().startswith('timestamp:'):
            if 'attribute' in current_attribute:  # Only add if we have an attribute
                current_attribute['timestamp'] = line[len('timestamp:'):].strip()
                
                # If we have all three fields, add to attributes and reset
                if all(k in current_attribute for k in ['attribute', 'citation', 'timestamp']):
                    attributes.append(current_attribute)
                    current_attribute = {}
    
    # Add the last attribute if it wasn't added
    if current_attribute and 'attribute' in current_attribute:
        # Ensure all fields exist
        if 'citation' not in current_attribute:
            current_attribute['citation'] = ''
            current_attribute['conversation_number'] = None
        elif 'conversation_number' not in current_attribute:
            # Try to extract conversation number if not already done
            conversation_match = re.search(r'conversation\s+(\d+)', current_attribute['citation'].lower())
            if conversation_match:
                current_attribute['conversation_number'] = int(conversation_match.group(1))
            else:
                current_attribute['conversation_number'] = None
                
        if 'timestamp' not in current_attribute:
            current_attribute['timestamp'] = ''
        attributes.append(current_attribute)
    
    return attributes

async def process_conversations_in_parallel(conversations, model_name="gpt-4o-mini", batch_size=20, person_name="Ben", max_concurrent=5):
    """
    Process multiple batches of conversations concurrently to extract personal attributes.
    
    Args:
        conversations: List of conversation dictionaries
        batch_size: Number of conversations to process in each batch
        person_name: Name of the person to extract attributes about
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        List of dictionaries containing attributes, their citations, and timestamps
    """

    all_attributes = []
    
    # Create batches of conversations
    batches = [conversations[i:i+batch_size] for i in range(0, len(conversations), batch_size)]
    total_batches = len(batches)
    
    # Process batches in chunks to control concurrency
    for i in range(0, len(batches), max_concurrent):
        current_chunk = batches[i:i+max_concurrent]
        current_tasks = []
        
        # Print progress manually instead of using tqdm
        print(f"Processing batches {i+1}-{min(i+max_concurrent, total_batches)} of {total_batches} ({(i+1)/total_batches*100:.1f}% - {min(i+max_concurrent, total_batches)/total_batches*100:.1f}%)")
        
        # Create tasks for all batches in the current chunk
        for batch_idx, batch in enumerate(current_chunk):
            task = extract_personal_attributes_async(batch, model_name, person_name, batch_size)
            current_tasks.append(task)
        
        # Execute all tasks concurrently and wait for them to complete
        batch_results = await asyncio.gather(*current_tasks)
        
        # Process the results
        for batch_idx, result in enumerate(batch_results):
            # Get the corresponding batch
            batch = current_chunk[batch_idx]
            
            # Add the actual conversation to each attribute based on conversation_number
            for attr in result:
                if 'conversation_number' in attr and attr['conversation_number'] is not None:
                    # Adjust the conversation number to be 1-indexed within the batch
                    conv_idx = attr['conversation_number'] - 1
                    if 0 <= conv_idx < len(batch):
                        attr['conversation_content'] = batch[conv_idx]
                    else:
                        attr['conversation_content'] = None
                else:
                    attr['conversation_content'] = None
            
            # Add results to all_attributes
            all_attributes.extend(result)
        
        # Optional: add a small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    print(f"Processing complete! Extracted {len(all_attributes)} attributes.")
    return all_attributes

# Function to run the async code from a synchronous context
def extract_personal_attributes_parallel(conversations, model_name="gpt-4o-mini", batch_size=20, person_name="Ben", max_concurrent=5):
    """
    Wrapper function to run the async extraction in a synchronous context.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_conversations_in_parallel(
            conversations, model_name, batch_size, person_name, max_concurrent
        ))
    finally:
        loop.close()

########################################################
# Combine and infer attributes
########################################################

async def combine_and_infer_attributes_async(attributes_batch, person_name="Ben", model_name="gpt-4o"):
    """
    Asynchronously combines related attributes and infers new attributes based on existing ones.
    
    Args:
        attributes_batch: A batch of attribute dictionaries
        person_name: Name of the person the attributes are about
        
    Returns:
        Dictionary containing combined and inferred attributes for this batch
    """
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    # Format attributes for the prompt
    formatted_attributes = []
    for idx, attr in enumerate(attributes_batch):
        counter = 0
        timestamp = attr.get('timestamp', '')
        if not timestamp:
            timestamp = attr.get('Timestamp', '')
        while not timestamp and idx+counter < len(attributes_batch) and idx-counter >= 0:
            timestamp = attributes_batch[idx+counter].get('timestamp', '')
            if not timestamp:
                timestamp = attributes_batch[idx+counter].get('Timestamp', '')
            if not timestamp:
                timestamp = attributes_batch[idx-counter].get('timestamp', '')
            if not timestamp:
                timestamp = attributes_batch[idx-counter].get('Timestamp', '')
            counter += 1
        formatted_attr = f"Attribute {idx+1}: {attr['attribute']} (Citation: {attr['citation']}, Timestamp: {timestamp})"
        formatted_attributes.append(formatted_attr)
    
    prompt = f"""
    You are an expert at analyzing personal information and making connections between related facts.
    
    I'll provide you with a list of attributes about {person_name} that were extracted from conversations.
    Your task is to:
    
    1. Identify and combine attributes that are talking about the same thing but might be phrased differently or contain complementary information.
    2. Infer new attributes that can be reasonably deduced by combining existing attributes.
    
    For example:
    - If one attribute says "{person_name} studies at Stanford" and another says "{person_name} is majoring in Computer Science", 
      you might combine them as "{person_name} studies Computer Science at Stanford".
    - If attributes mention "{person_name} grew up in Seattle" and "{person_name} moved to Boston for college", 
      you might infer "{person_name} relocated from the West Coast to the East Coast for higher education".
    
    Please format your response EXACTLY as follows, with two clearly labeled sections:

    COMBINED ATTRIBUTES:
    - Attribute: {person_name} studies Computer Science at Stanford.
    - Based on: Attribute 3, Attribute 7
    
    - Attribute: {person_name} has been playing piano for over 10 years and performs in local venues.
    - Based on: Attribute 12, Attribute 15
    
    INFERRED ATTRIBUTES:
    - Attribute: {person_name} relocated from the West Coast to the East Coast for higher education.
    - Based on: Attribute 2, Attribute 9
    
    - Attribute: {person_name} likely has an interest in both technology and music.
    - Based on: Attribute 3, Attribute 12
    
    Here are the attributes about {person_name}:
    
    {'''
'''.join(formatted_attributes)}
    """
    
    # Call the model
    response = await client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You analyze personal attributes, combine related ones, and infer new information."},
            {"role": "user", "content": prompt}
        ]
    )
    
    # Parse the response using a custom parser
    result = parse_combined_attributes_from_text(response.choices[0].message.content)
    
    # Add the actual attribute objects to the combined and inferred attributes
    for combined_attr in result["combined_attributes"]:
        source_attributes = []
        if "based_on" in combined_attr:
            # Extract attribute numbers from the "based_on" field
            attr_nums = re.findall(r'Attribute\s+(\d+)', combined_attr["based_on"])
            for num in attr_nums:
                try:
                    idx = int(num) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(attributes_batch):
                        source_attributes.append(attributes_batch[idx])
                except ValueError:
                    continue
        combined_attr["source_attributes"] = source_attributes
    
    for inferred_attr in result["inferred_attributes"]:
        source_attributes = []
        if "based_on" in inferred_attr:
            # Extract attribute numbers from the "based_on" field
            attr_nums = re.findall(r'Attribute\s+(\d+)', inferred_attr["based_on"])
            for num in attr_nums:
                try:
                    idx = int(num) - 1  # Convert to 0-indexed
                    if 0 <= idx < len(attributes_batch):
                        source_attributes.append(attributes_batch[idx])
                except ValueError:
                    continue
        inferred_attr["source_attributes"] = source_attributes
    
    return result

def parse_combined_attributes_from_text(text):
    """
    Parse combined and inferred attributes from the model's text response.
    
    Args:
        text: The text response from the model
    
    Returns:
        Dictionary containing combined and inferred attributes
    """
    combined_attributes = []
    inferred_attributes = []
    
    # Determine which section we're in
    current_section = None
    current_attribute = {}
    
    # Split the text into lines
    lines = text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Check for section headers
        if "COMBINED ATTRIBUTES" in line.upper():
            current_section = "combined"
            continue
        elif "INFERRED ATTRIBUTES" in line.upper():
            current_section = "inferred"
            continue
        
        # Remove leading dash or bullet point if present
        if line.startswith('- '):
            line = line[2:]
        elif line.startswith('• '):
            line = line[2:]
            
        # Check for attribute or based_on
        if line.lower().startswith('attribute:'):
            # If we already have an attribute in progress, save it
            if current_attribute and 'attribute' in current_attribute:
                if current_section == "combined":
                    combined_attributes.append(current_attribute)
                elif current_section == "inferred":
                    inferred_attributes.append(current_attribute)
                current_attribute = {}
            
            current_attribute['attribute'] = line[len('attribute:'):].strip()
        
        elif line.lower().startswith('based on:'):
            if 'attribute' in current_attribute:  # Only add if we have an attribute
                current_attribute['based_on'] = line[len('based on:'):].strip()
                
                # If we have both fields, add to appropriate list and reset
                if all(k in current_attribute for k in ['attribute', 'based_on']):
                    if current_section == "combined":
                        combined_attributes.append(current_attribute)
                    elif current_section == "inferred":
                        inferred_attributes.append(current_attribute)
                    current_attribute = {}
    
    # Add the last attribute if it wasn't added
    if current_attribute and 'attribute' in current_attribute:
        # Ensure all fields exist
        if 'based_on' not in current_attribute:
            current_attribute['based_on'] = ''
            
        if current_section == "combined":
            combined_attributes.append(current_attribute)
        elif current_section == "inferred":
            inferred_attributes.append(current_attribute)
    
    return {
        "combined_attributes": combined_attributes,
        "inferred_attributes": inferred_attributes
    }

async def process_attributes_in_parallel(attributes, model_name="gpt-4o", batch_size=50, person_name="Ben", max_concurrent=5):
    """
    Process multiple batches of attributes concurrently to combine and infer new attributes.
    
    Args:
        attributes: List of attribute dictionaries
        batch_size: Number of attributes to process in each batch
        person_name: Name of the person the attributes are about
        max_concurrent: Maximum number of concurrent API calls
        
    Returns:
        Dictionary containing original, combined, and inferred attributes
    """
    import asyncio
    
    # Create batches of attributes
    batches = [attributes[i:i+batch_size] for i in range(0, len(attributes), batch_size)]
    total_batches = len(batches)
    
    all_combined_attributes = []
    all_inferred_attributes = []
    
    # Process batches in chunks to control concurrency
    for i in range(0, len(batches), max_concurrent):
        current_chunk = batches[i:i+max_concurrent]
        current_tasks = []
        
        # Print progress
        print(f"Processing batches {i+1}-{min(i+max_concurrent, total_batches)} of {total_batches} ({(i+1)/total_batches*100:.1f}% - {min(i+max_concurrent, total_batches)/total_batches*100:.1f}%)")
        
        for batch in current_chunk:
            task = combine_and_infer_attributes_async(batch, person_name, model_name)
            current_tasks.append(task)
        
        # Wait for all tasks in the current chunk to complete
        results = await asyncio.gather(*current_tasks)
        
        # Collect results
        for result in results:
            all_combined_attributes.extend(result["combined_attributes"])
            all_inferred_attributes.extend(result["inferred_attributes"])
        
        # Optional: add a small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    # Return the final results
    return {
        "original_attributes": attributes,
        "combined_attributes": all_combined_attributes,
        "inferred_attributes": all_inferred_attributes
    }

# Function to run the async code from a synchronous context
def combine_and_infer_attributes_parallel(attributes, model_name="gpt-4o", batch_size=50, person_name="Ben", max_concurrent=5):
    """
    Wrapper function to run the async attribute processing in a synchronous context.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_attributes_in_parallel(
            attributes, model_name, batch_size, person_name, max_concurrent
        ))
    finally:
        loop.close()

async def generate_top_order_memories(attributes_data, model_name="gpt-4o-mini", target_count=20, person_name="Ben", max_concurrent=5):
    """
    Generate top-order memories focused on specific events and life periods rather than general attributes.
    Handles large attribute sets by processing in chunks concurrently.
    """
    # Extract all attributes from the data
    all_attributes = []
    
    # Add original attributes
    if "original_attributes" in attributes_data:
        for attr in attributes_data["original_attributes"]:
            all_attributes.append({
                "type": "Original Attribute",
                "id": attr.get("id", len(all_attributes) + 1),
                "content": attr["attribute"],
                "citation": attr.get("citation", "")
            })
    
    # Add combined attributes
    if "combined_attributes" in attributes_data:
        for attr in attributes_data["combined_attributes"]:
            all_attributes.append({
                "type": "Combined Attribute",
                "id": attr.get("id", len(all_attributes) + 1),
                "content": attr["attribute"],
                "citation": attr.get("based_on", "")
            })
    
    # Add inferred attributes
    if "inferred_attributes" in attributes_data:
        for attr in attributes_data["inferred_attributes"]:
            all_attributes.append({
                "type": "Inferred Attribute",
                "id": attr.get("id", len(all_attributes) + 1),
                "content": attr["attribute"],
                "citation": attr.get("based_on", "")
            })
    
    # Handle large attribute sets by chunking
    chunk_size = 100  # Process 100 attributes at a time
    all_memories = []
    
    # Create chunks of attributes
    chunks = [all_attributes[i:i+chunk_size] for i in range(0, len(all_attributes), chunk_size)]
    total_chunks = len(chunks)
    
    client = AsyncOpenAI()
    
    # Process chunks in batches to control concurrency
    for i in range(0, len(chunks), max_concurrent):
        current_batch = chunks[i:i+max_concurrent]
        tasks = []
        
        # Print progress
        print(f"Processing memory chunks {i+1}-{min(i+max_concurrent, total_chunks)} of {total_chunks} ({(i+1)/total_chunks*100:.1f}% - {min(i+max_concurrent, total_chunks)/total_chunks*100:.1f}%)")
        
        # Create tasks for each chunk in the current batch
        for chunk_idx, chunk in enumerate(current_batch):
            # Create a prompt for generating event-based memories from this chunk
            prompt = f"""
            You are tasked with creating event-based memories about {person_name} based on the attributes provided.
            
            IMPORTANT: Focus on specific events, experiences, and time periods in {person_name}'s life, NOT general traits or characteristics.
            
            Good examples (These are made up, do not use them as examples):
            - "{person_name} attended Morgantown High School from 2016 to 2020, where he was part of the quiz bowl team and developed a strong interest in chemistry thanks to his favorite teacher, Mr. Johnson."
            - "During summer 2023, {person_name} traveled to Las Vegas with friends Alex and Jordan, where they attended a music festival and {person_name} won $200 at blackjack."
            - "In Fall 2022, {person_name} began working on his first major AI research paper with Professor Zhang, focusing on personalized AI models, which was later published in March 2023."
            
            Bad examples (too general, not event-based):
            - "{person_name} enjoys playing video games like Stardew Valley."
            - "{person_name} is interested in AI research and collaborates with friends."
            - "{person_name} values financial management and budgeting."
            
            For each memory:
            1. Focus on specific events, time periods, or experiences
            2. Include relevant details like when it happened, who was involved, and what specifically occurred
            3. Organize memories chronologically when possible
            4. Cite which attributes this memory is based on
            
            Here are the attributes to work with (chunk {i+chunk_idx+1} of {total_chunks}):
            
            {json.dumps(chunk, indent=2)}
            
            Return your response as a JSON array of objects, each with:
            - "attribute": The detailed event-based memory
            - "citation": References to the attributes this is based on (e.g., "Original Attribute 5, Combined Attribute 2")
            - "timestamp": Approximate time period if known (e.g., "Summer 2023", "2016-2020", "March 2023")
            
            Create up to 5 high-quality, event-based memories from this chunk of attributes.
            """
            
            # Create a task for this chunk
            task = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates meaningful, event-based memories from attribute data."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            tasks.append(task)
        
        # Process all tasks in the current batch concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process the responses
        for response in responses:
            if isinstance(response, Exception):
                print(f"Error processing chunk: {response}")
                continue
                
            try:
                content = response.choices[0].message.content
                # Extract JSON from the response
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    chunk_memories = json.loads(json_str)
                    all_memories.extend(chunk_memories)
                else:
                    # Fallback if JSON parsing fails
                    print(f"Warning: Could not parse JSON from a chunk. Skipping.")
                    print(f"Response content: {content}")
            except Exception as e:
                print(f"Error processing response: {e}")
        
        # Optional: add a small delay to avoid rate limiting
        await asyncio.sleep(0.1)
    
    # Sort memories by time_period if available
    all_memories.sort(key=lambda x: x.get("time_period", "Unknown"))
    
    # Return the top memories up to the target count
    return all_memories[:target_count]

def generate_top_order_memories_sync(attributes_data, model_name="gpt-4o-mini", target_count=20, person_name="Ben", max_concurrent=5):
    """
    Wrapper function to run the async memory generation in a synchronous context.
    """
    import asyncio
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(generate_top_order_memories(
            attributes_data, model_name, target_count, person_name, max_concurrent
        ))
    finally:
        loop.close()