from dotenv import load_dotenv
load_dotenv()

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import anthropic
from openai import OpenAI
from together import Together
import os
import sys
import numpy as np
from .memory_module import MemoryModule
from .example_module import ExampleModule
from .run_custom_model import load_model, run_custom_model
from datetime import datetime

openai_api_key = os.environ.get('OPENAI_API_KEY', '')
anthropic_api_key = os.environ.get('ANTHROPIC_API_KEY', '')
deepseek_api_key = os.environ.get('DEEPSEEK_API_KEY', '')
together_api_key = os.environ.get('TOGETHER_API_KEY', '')

today_date = datetime.now().strftime("%Y-%m-%d")

STANDARD_SYSTEM_MESSAGE = "Today's date is {today_date}. You are a human being named {persona_name}. You are not an AI. Respond as yourself. You will first be given the topic of conversation, then any existing conversation history if there is any. Be sure to reply in the style of {persona_name}. Use the '<|msg|>' token to send mulitple messages at once if you wish. "
EXAMPLE_SYSTEM_MESSAGE = "Here is an example conversation between {persona_name} and another individual. This conversation is not relevant to the current conversation. Use this conversation to help aid you to emulate stylistically on how to communicate in your conversations. 'Me' is the user that you are imitating.\n\n [BEGIN EXAMPLE CONVERSATION]\n{examples}\n[END EXAMPLE CONVERSATION]"
MEMORY_SYSTEM_MESSAGE = "Here are some of your relevant memories + facts about yourself that may help you respond authentically. Pay careful attention to the date of the memories, as events have occurred in the past, and you should make reference to them in the appropriate time manner. \n\n[BEGIN MEMORIES]\n{memory_text}\n[END MEMORIES]"
FINAL_SYSTEM_MESSAGE = "Now you may begin the conversation."

class BaseAgent(ABC):
    """
    Abstract base class for agents that imitate human behavior and content generation.
    This class defines the interface that all agent implementations must follow.
    """
    
    def __init__(self, model_name: str, impersonation_name: str, custom_model_path: str, adapter_path: Optional[str] = None):
        """
        Initialize the base agent.
        
        Args:
            model_name (str): Name of the model to use
            inference_args_path (str): Path to the inference arguments
        """
        self.model_name = model_name
        if 'custom' in self.model_name.lower():
            assert custom_model_path, "Custom model path must be provided for custom models"
            self.custom_model, self.tokenizer = load_model(custom_model_path, adapter_path=adapter_path)
        else:
            self.custom_model = None
        self.impersonation_name = impersonation_name
        self.history = []
        self.system_message = ""
        self.verbose = False
    
    @abstractmethod
    def generate_response(self, input_text: str) -> str:
        """
        Generate a response based on the input text.
        
        Args:
            input_text (str): The input text to respond to
            
        Returns:
            str: The generated response
        """
        pass
    
    @abstractmethod
    def generate_response_impersona(self, history: List[Dict]) -> str:
        """
        Generate a response based on a provided conversation history without maintaining state.
        
        Args:
            history (List[Dict]): List of conversation messages in the format [{'role': 'user|assistant', 'content': str}]
            
        Returns:
            str: The generated response
        """
        pass
    
    def reset(self) -> None:
        """
        Reset the agent's state to initial conditions.
        """
        self.history = []

    def _run_inference(self) -> str:
        """
        Runs inference over a specific model on the current history state
        """
        if self.verbose:  # Temporarily force logging
            print("\n=== SYSTEM PROMPT LOGGING ===")
            if self.history:
                print(f"System prompt: {self.history[0]['content']}")
            else:
                print("No system prompt found in history")
            print("=== END SYSTEM PROMPT LOGGING ===\n")
        
        try:
            # For models that don't support system messages, prepend to first user message
            if self.custom_model:
                return run_custom_model(self.custom_model, self.tokenizer, self.history)
            elif 'o1' in self.model_name:
                messages = self.history.copy()
                if messages[0]['role'] == 'system':
                    messages[0]['role'] = 'user'
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                )
                return response.choices[0].message.content
            elif 'gpt' in self.model_name or 'o3' in self.model_name:
                client = OpenAI(api_key=openai_api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                )
                return response.choices[0].message.content
            elif 'deepseek' in self.model_name:
                client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                )
                return response.choices[0].message.content
            elif 'claude' in self.model_name:
                client = anthropic.Anthropic(api_key=anthropic_api_key)
                # Convert messages to Anthropic format
                anthropic_messages = [
                    {
                        "role": msg["role"],
                        "content": [{"type": "text", "text": msg["content"]}]
                    }
                    for msg in self.history
                ]
                if self.history[0]['role'] == 'system':
                    system_message = self.history[0]['content']
                else:
                    system_message = ''
                response = client.messages.create(
                    model=self.model_name,
                    messages=anthropic_messages,
                    max_tokens=4096,
                    system=system_message
                )
                return response.content[0].text
            else:
                client = Together(api_key=together_api_key)
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=self.history,
                    temperature=0.8,
                    top_p=0.9,
                )
                return response.choices[0].message.content
        except Exception as e:
            error_message = str(e)
            if "Incorrect API key provided" in error_message:
                return "Error: Invalid API key"
            elif "Rate limit reached" in error_message:
                return "Error: Rate limit exceeded"
            else:
                return f"Error: {error_message}"

    def parse_conversation(self, conversation_text: str, admin_name: str = "admin") -> List[Dict]:
        """
        Parse a Discord-like conversation into the format needed for the agent's history.
        
        Args:
            conversation_text (str): Raw conversation text with <|> delimiters
            admin_name (str): Name of the admin/assistant in the conversation
            
        Returns:
            List[Dict]: Formatted conversation history
        """
        # Split the text into parts using the delimiter
        parts = conversation_text.split('<|>')
        
        # Extract metadata (parts before the first timestamp)
        metadata = []
        conversation_start = 0
        for i, part in enumerate(parts):
            if '[' in part and ']' in part:  # Found first timestamp
                conversation_start = i
                break
            # Extract the metadata value after the colon
            if ':' in part:
                metadata.append(part.strip())
        
        # Combine metadata into system message
        metadata_text = "\n".join(metadata)
        
        # Format the conversation similar to the training examples
        conversation_lines = []
        
        # Process the actual conversation messages
        for part in parts[conversation_start:]:
            # Skip empty parts
            if not part.strip():
                continue
                
            # Parse the part
            try:
                # Extract username from the format [datetime] username: message
                message_parts = part.split(': ', 1)
                if len(message_parts) < 2:
                    continue  # Skip parts that don't have the expected format
                
                header = message_parts[0]
                message = message_parts[1].strip()
                
                # Replace admin_name with "Me" in the header
                if admin_name in header:
                    header = header.replace(admin_name, "Me")
                
                conversation_lines.append(f"{header}: {message}")
                
            except Exception as e:
                print(f"Error parsing part: {part}")
                continue
        
        # Format the conversation as a single string
        conversation_text = "\n".join(conversation_lines)
        
        # Create a system message with metadata
        system_message = f"Here is a conversation topic that you can begin conversation about with people. Make sure to respond to current conversation, and only use this as a guideline if there hasn't been any conversation yets. [BEGIN TOPIC METADATA]\n{metadata_text}\n[END TOPIC METADATA]."
        
        # Create a user message with the conversation and ending with "Me: " to prompt the model to respond
        user_message = f"Current conversation:\n{conversation_text}\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Me: "
        
        # Return the formatted history
        return [
            {'role': 'system', 'content': system_message},
            {'role': 'user', 'content': user_message}
        ]

class BasicAgent(BaseAgent):
    def __init__(self, model_name: str, impersonation_name: str, user_name: str, example_module: Optional[ExampleModule] = None, memory_module: Optional[MemoryModule] = None, custom_model_path: Optional[str] = None, adapter_path: Optional[str] = None):
        super().__init__(model_name, impersonation_name, custom_model_path, adapter_path=adapter_path)
        self.example_module = example_module
        self.memory_module = memory_module
        self.impersonation_name = impersonation_name
        self.user_name = user_name

        self.system_message = STANDARD_SYSTEM_MESSAGE.format(persona_name=self.impersonation_name, today_date=today_date)
        self.history = [{'role': 'system', 'content': self.system_message}]
    
    def _get_system_message(self) -> str:
        curr_system_message = self.system_message
        if self.example_module:
            example_conversation = self.example_module.search_conversation_store(self.user_name)
            curr_system_message = curr_system_message + "\n\n" + EXAMPLE_SYSTEM_MESSAGE.format(persona_name=self.impersonation_name, examples=example_conversation)
        if self.memory_module:
            history_text = ""
            # Get only the last 3 user messages
            user_messages = [msg['content'] for msg in self.history[1:] if msg['role'] == 'user']
            last_three_user_messages = user_messages[-3:] if len(user_messages) >= 3 else user_messages
            history_text = "\n\n".join(last_three_user_messages)
            memories = self.memory_module.search_memory(history_text)
            curr_system_message = curr_system_message + "\n\n" + MEMORY_SYSTEM_MESSAGE.format(persona_name=self.impersonation_name, memory_text=memories)
        
        # Add the final system message at the end
        curr_system_message = curr_system_message + "\n\n" + FINAL_SYSTEM_MESSAGE
        
        return curr_system_message
        
    def generate_response(self, input_text: str) -> str:
        # Dynamically changing the system message to include the example conversation
        self.history.append({'role': 'user', 'content': input_text + "/n Me: "})
        if self.history and self.history[0]['role'] == 'system':
            self.history[0]['content'] = self._get_system_message()
        
        response = self._run_inference()
        self.history.append({'role': 'assistant', 'content': response})
        return response
    
    def generate_response_impersona(self, history: List[Dict]) -> str:
        original_history = self.history
        original_system_message = self.system_message
        self.history = history
        curr_system_message = self._get_system_message()
        if self.history and self.history[0]['role'] == 'system':
            topic_information = self.history[0]['content']
            self.history[0]['content'] = curr_system_message + "\n\n" + topic_information
            self.history[0]['role'] = 'user'
        
        response = self._run_inference()
        self.history = original_history
        self.system_message = original_system_message
        
        return response