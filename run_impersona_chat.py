from flask import Flask, render_template, request, jsonify, session
from IMPersona.agents import BasicAgent
from IMPersona.example_module import ExampleModule
from IMPersona.memory_module import MemoryModule
import os
from dotenv import load_dotenv
import uuid
import glob

# Load environment variables
load_dotenv()

# Constants
MEMORY_BANK_PATH = "data/memory_bank.json"
CONVERSATION_STORE_PATH = "data/conversation_store.json"

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.urandom(24).hex())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_chat', methods=['POST'])
def start_chat():
    data = request.json
    user_name = data.get('user_name')
    impersonation_name = data.get('impersonation_name')
    model_name = data.get('model_name')
    use_icl = data.get('use_icl', False)
    use_memory = data.get('use_memory', False)
    memory_type = data.get('memory_type', 'normal')  # Add memory type selection with 'normal' as default
    custom_model_path = data.get('custom_model_path', "")
    adapter_path = data.get('adapter_path', "")

    print(f"USING IMPERSONATION NAME: {impersonation_name}")
    # Generate a unique session ID
    session_id = str(uuid.uuid4())
    
    # Store chat configuration in session
    session[session_id] = {
        'user_name': user_name,
        'impersonation_name': impersonation_name,
        'model_name': model_name,
        'use_icl': use_icl,
        'use_memory': use_memory,
        'memory_type': memory_type,  # Store memory type in session
        'custom_model_path': custom_model_path,
        'adapter_path': adapter_path,
        'messages': []  # Initialize empty message history
    }
    
    return jsonify({'session_id': session_id, 'status': 'success'})

@app.route('/chat/<session_id>', methods=['POST'])
def chat(session_id):
    if session_id not in session:
        return jsonify({'error': 'Invalid session'}), 400
    
    chat_config = session[session_id]
    user_input = request.json.get('message')
    
    # Initialize modules based on configuration
    example_module = None
    memory_module = None
    conversation_store_path = request.json.get('conversation_store_path', CONVERSATION_STORE_PATH)
    
    if chat_config['use_icl']:
        example_module = ExampleModule(conversation_store_path=conversation_store_path)
        # If impersonation name wasn't set earlier, try to extract it from the filename
        if not chat_config['impersonation_name'] and conversation_store_path:
            filename = os.path.basename(conversation_store_path)
            name_parts = filename.split('_')
            if name_parts and len(name_parts) > 0:
                chat_config['impersonation_name'] = name_parts[0]
    
    if chat_config['use_memory']:
        # Initialize the appropriate memory module based on the selected type
        memory_type = chat_config.get('memory_type', 'normal')
        if memory_type == 'hierarchical':
            from IMPersona.memory_module import HierarchicalMemoryModule
            memory_module = HierarchicalMemoryModule(attribute_path=MEMORY_BANK_PATH)
        else:
            from IMPersona.memory_module import MemoryModule
            memory_module = MemoryModule(attribute_path=MEMORY_BANK_PATH)
    
    # Initialize agent
    agent = BasicAgent(
        chat_config['model_name'], 
        chat_config['impersonation_name'], 
        chat_config['user_name'], 
        example_module=example_module, 
        memory_module=memory_module, 
        custom_model_path=chat_config['custom_model_path'],
        adapter_path=chat_config['adapter_path']
    )
    
    # Generate response
    response = agent.generate_response(user_input)
    
    # Update message history
    chat_config['messages'].append({'role': 'user', 'content': user_input})
    chat_config['messages'].append({'role': 'assistant', 'content': response})
    session[session_id] = chat_config
    
    return jsonify({
        'response': response,
        'impersonation_name': chat_config['impersonation_name']
    })

@app.route('/get_history/<session_id>')
def get_history(session_id):
    if session_id not in session:
        return jsonify({'error': 'Invalid session'}), 400
    
    return jsonify({'messages': session[session_id]['messages']})

@app.route('/get_default_impersonation_name')
def get_default_impersonation_name():
    data_dir = "data"
    matching_files = glob.glob(f"{data_dir}/*_impersona_imessage_0buffer*.jsonl")
    default_name = ""
    
    if matching_files:
        filename = os.path.basename(matching_files[0])
        # Extract the name part (everything before the first underscore)
        name_part = filename.split('_')[0] if '_' in filename else ''
        if name_part:
            default_name = name_part
    
    return jsonify({'default_name': default_name})

if __name__ == '__main__':
    app.run(debug=True) 