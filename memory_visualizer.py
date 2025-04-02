import json
from flask import Flask, render_template
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def memory_visualizer():
    # Load memory data from JSON file
    try:
        with open('data/memory_bank.json', 'r') as f:
            memory_data = json.load(f)
    except Exception as e:
        return f"Error loading memory bank: {str(e)}"
    
    # Extract all types of attributes
    original_attributes = memory_data.get('original_attributes', [])
    combined_attributes = memory_data.get('combined_attributes', [])
    inferred_attributes = memory_data.get('inferred_attributes', [])

    assert original_attributes, "No original attributes found"
    assert combined_attributes, "No combined attributes found"
    assert inferred_attributes, "No inferred attributes found"
    
    # Process all attributes
    all_attributes = []
    
    # Process original attributes
    for attr in original_attributes:
        attr_copy = attr.copy()
        attr_copy['type'] = 'original'  # Add type label
        # Preserve conversation content for display
        if 'conversation_content' in attr_copy:
            attr_copy['has_conversation'] = True
        else:
            attr_copy['has_conversation'] = False
            
        if 'timestamp' in attr_copy:
            try:
                attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M:%S')
                all_attributes.append(attr_copy)
            except ValueError:
                try:
                    # Try alternative format
                    attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M')
                    all_attributes.append(attr_copy)
                except ValueError:
                    # If timestamp format is different, use a default date
                    attr_copy['datetime_obj'] = datetime.min
                    all_attributes.append(attr_copy)
    
    # Process combined attributes
    for attr in combined_attributes:
        attr_copy = attr.copy()
        attr_copy['type'] = 'combined'  # Add type label
        
        # Check if source attributes have conversation content
        attr_copy['has_conversation'] = False
        if 'source_attributes' in attr_copy:
            for source_attr in attr_copy['source_attributes']:
                if 'conversation_content' in source_attr:
                    attr_copy['has_conversation'] = True
                    break
        
        # For combined attributes, try to get timestamp from source_attributes
        if 'timestamp' not in attr_copy and 'source_attributes' in attr_copy:
            # Find the earliest timestamp from source attributes
            earliest_timestamp = None
            for source_attr in attr_copy['source_attributes']:
                if 'timestamp' in source_attr:
                    try:
                        timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M:%S')
                        if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                            earliest_timestamp = timestamp_obj
                            attr_copy['timestamp'] = source_attr['timestamp']
                    except ValueError:
                        try:
                            timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M')
                            if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                                earliest_timestamp = timestamp_obj
                                attr_copy['timestamp'] = source_attr['timestamp']
                        except ValueError:
                            continue
        
        if 'timestamp' in attr_copy:
            try:
                attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M:%S')
                all_attributes.append(attr_copy)
            except ValueError:
                try:
                    # Try alternative format
                    attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M')
                    all_attributes.append(attr_copy)
                except ValueError:
                    # If timestamp format is different, use a default date
                    attr_copy['datetime_obj'] = datetime.min
                    all_attributes.append(attr_copy)
        else:
            # If no timestamp found, use default date
            print(f"No timestamp found for attribute: {attr_copy['attribute']}")
    
    # Process inferred attributes
    for attr in inferred_attributes:
        attr_copy = attr.copy()
        attr_copy['type'] = 'inferred'  # Add type label
        
        # Check if source attributes have conversation content
        attr_copy['has_conversation'] = False
        if 'source_attributes' in attr_copy:
            for source_attr in attr_copy['source_attributes']:
                if 'conversation_content' in source_attr:
                    attr_copy['has_conversation'] = True
                    break
        
        # For inferred attributes, try to get timestamp from source_attributes (same as combined)
        if 'timestamp' not in attr_copy and 'source_attributes' in attr_copy:
            # Find the earliest timestamp from source attributes
            earliest_timestamp = None
            for source_attr in attr_copy['source_attributes']:
                if 'timestamp' in source_attr:
                    try:
                        timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M:%S')
                        if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                            earliest_timestamp = timestamp_obj
                            attr_copy['timestamp'] = source_attr['timestamp']
                    except ValueError:
                        try:
                            timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M')
                            if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                                earliest_timestamp = timestamp_obj
                                attr_copy['timestamp'] = source_attr['timestamp']
                        except ValueError:
                            continue
        
        if 'timestamp' in attr_copy:
            try:
                attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M:%S')
                all_attributes.append(attr_copy)
            except ValueError:
                try:
                    # Try alternative format
                    attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M')
                    all_attributes.append(attr_copy)
                except ValueError:
                    # If timestamp format is different, use a default date
                    attr_copy['datetime_obj'] = datetime.min
                    all_attributes.append(attr_copy)
        else:
            # If no timestamp found, use default date
            print(f"No timestamp found for attribute: {attr_copy['attribute']}")
    
    # Sort all attributes by timestamp (newest first)
    sorted_attributes = sorted(all_attributes, key=lambda x: x.get('datetime_obj', datetime.min), reverse=True)
    
    # Add global index to each attribute for conversation lookup
    for i, attr in enumerate(sorted_attributes):
        attr['global_index'] = i
    
    # Group attributes by date for better visualization
    grouped_memories = {}
    for attr in sorted_attributes:
        date_str = attr['datetime_obj'].strftime('%Y-%m-%d')
        if date_str not in grouped_memories:
            grouped_memories[date_str] = []
        grouped_memories[date_str].append(attr)
    
    # Store the sorted attributes in the session for conversation view
    app.config['sorted_attributes'] = sorted_attributes
    
    return render_template('memory_visualizer.html', grouped_memories=grouped_memories)

@app.route('/conversation/<int:attr_index>')
def view_conversation(attr_index):
    # Get the sorted attributes from the app config
    sorted_attributes = app.config.get('sorted_attributes', [])
    
    if not sorted_attributes:
        # If sorted_attributes is not available, recreate it
        try:
            with open('data/combined_attributes.json', 'r') as f:
                memory_data = json.load(f)
        except Exception as e:
            return f"Error loading memory bank: {str(e)}"
        
        # Extract all types of attributes
        original_attributes = memory_data.get('original_attributes', [])
        combined_attributes = memory_data.get('combined_attributes', [])
        inferred_attributes = memory_data.get('inferred_attributes', [])
        
        # Process all attributes - using the EXACT same logic as in memory_visualizer
        all_attributes = []
        
        # Process original attributes
        for attr in original_attributes:
            attr_copy = attr.copy()
            attr_copy['type'] = 'original'
            if 'conversation_content' in attr_copy:
                attr_copy['has_conversation'] = True
            else:
                attr_copy['has_conversation'] = False
                
            if 'timestamp' in attr_copy:
                try:
                    attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M:%S')
                    all_attributes.append(attr_copy)
                except ValueError:
                    try:
                        attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M')
                        all_attributes.append(attr_copy)
                    except ValueError:
                        attr_copy['datetime_obj'] = datetime.min
                        all_attributes.append(attr_copy)
        
        # Process combined attributes
        for attr in combined_attributes:
            attr_copy = attr.copy()
            attr_copy['type'] = 'combined'
            
            attr_copy['has_conversation'] = False
            if 'source_attributes' in attr_copy:
                for source_attr in attr_copy['source_attributes']:
                    if 'conversation_content' in source_attr:
                        attr_copy['has_conversation'] = True
                        break
            
            if 'timestamp' not in attr_copy and 'source_attributes' in attr_copy:
                earliest_timestamp = None
                for source_attr in attr_copy['source_attributes']:
                    if 'timestamp' in source_attr:
                        try:
                            timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M:%S')
                            if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                                earliest_timestamp = timestamp_obj
                                attr_copy['timestamp'] = source_attr['timestamp']
                        except ValueError:
                            try:
                                timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M')
                                if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                                    earliest_timestamp = timestamp_obj
                                    attr_copy['timestamp'] = source_attr['timestamp']
                            except ValueError:
                                continue
            
            if 'timestamp' in attr_copy:
                try:
                    attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M:%S')
                    all_attributes.append(attr_copy)
                except ValueError:
                    try:
                        attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M')
                        all_attributes.append(attr_copy)
                    except ValueError:
                        attr_copy['datetime_obj'] = datetime.min
                        all_attributes.append(attr_copy)
            else:
                attr_copy['datetime_obj'] = datetime.min
                all_attributes.append(attr_copy)
        
        # Process inferred attributes
        for attr in inferred_attributes:
            attr_copy = attr.copy()
            attr_copy['type'] = 'inferred'
            
            attr_copy['has_conversation'] = False
            if 'source_attributes' in attr_copy:
                for source_attr in attr_copy['source_attributes']:
                    if 'conversation_content' in source_attr:
                        attr_copy['has_conversation'] = True
                        break
            
            if 'timestamp' not in attr_copy and 'source_attributes' in attr_copy:
                earliest_timestamp = None
                for source_attr in attr_copy['source_attributes']:
                    if 'timestamp' in source_attr:
                        try:
                            timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M:%S')
                            if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                                earliest_timestamp = timestamp_obj
                                attr_copy['timestamp'] = source_attr['timestamp']
                        except ValueError:
                            try:
                                timestamp_obj = datetime.strptime(source_attr['timestamp'], '%Y-%m-%d %H:%M')
                                if earliest_timestamp is None or timestamp_obj < earliest_timestamp:
                                    earliest_timestamp = timestamp_obj
                                    attr_copy['timestamp'] = source_attr['timestamp']
                            except ValueError:
                                continue
            
            if 'timestamp' in attr_copy:
                try:
                    attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M:%S')
                    all_attributes.append(attr_copy)
                except ValueError:
                    try:
                        attr_copy['datetime_obj'] = datetime.strptime(attr_copy['timestamp'], '%Y-%m-%d %H:%M')
                        all_attributes.append(attr_copy)
                    except ValueError:
                        attr_copy['datetime_obj'] = datetime.min
                        all_attributes.append(attr_copy)
            else:
                attr_copy['datetime_obj'] = datetime.min
                all_attributes.append(attr_copy)
        
        # Sort all attributes by timestamp (newest first) - EXACT same sorting as in memory_visualizer
        sorted_attributes = sorted(all_attributes, key=lambda x: x.get('datetime_obj', datetime.min), reverse=True)
    
    if attr_index < 0 or attr_index >= len(sorted_attributes):
        return "Attribute index out of range"
    
    attribute = sorted_attributes[attr_index]
    
    # Get conversation content
    conversation = None
    if 'conversation_content' in attribute:
        conversation = attribute['conversation_content']
    elif 'source_attributes' in attribute:
        # For combined/inferred attributes, get conversations from source attributes
        conversations = []
        for source_attr in attribute['source_attributes']:
            if 'conversation_content' in source_attr:
                conversations.append({
                    'attribute': source_attr.get('attribute', 'Unknown'),
                    'content': source_attr['conversation_content']
                })
        if conversations:
            return render_template('conversation_view.html', 
                                  attribute=attribute, 
                                  multiple_conversations=True,
                                  conversations=conversations)
    
    if conversation:
        return render_template('conversation_view.html', 
                              attribute=attribute, 
                              multiple_conversations=False,
                              conversation=conversation)
    else:
        return "No conversation content found for this attribute"

if __name__ == '__main__':
    app.run(debug=True)
