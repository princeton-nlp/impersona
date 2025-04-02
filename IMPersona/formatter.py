from typing import List, Dict
from abc import ABC, abstractmethod

class ModelFormatter(ABC):
    """Base class for model-specific formatting strategies"""
    
    @abstractmethod
    def format_conversation(self, conversation: List[Dict]) -> Dict:
        """Format a single conversation for the specific model"""
        pass

class LlamaFormatter(ModelFormatter):
    """Formatter for llama-8b-instruct model"""

    def format_conversation(
        self,
        system_prompt: str,
        metadata: str,
        prior_conversations: List[List[Dict]],
        current_conversation: List[Dict],
        target_response: str,
        timestamp: str
    ) -> Dict:
        """
        Format a conversation for Llama training.
        
        Args:
            system_prompt: System instruction for the model
            metadata: String containing participant information
            prior_conversations: List of previous conversations for context
            current_conversation: Current conversation history
            target_response: The target response to generate
            
        Returns:
            Dictionary containing formatted input and target output
        """
        # Build the context string starting with system prompt and metadata
        context_parts = []
        
        # Add prior conversations as context only if they exist
        if prior_conversations and len(prior_conversations) > 0:
            context_parts.append("Previous conversations:\n")
            for conv in prior_conversations:
                conv_text = "\n".join(
                    f"[{msg.get('timestamp', 'No timestamp')}] {msg['speaker']}: {msg['content']}"
                    for msg in conv
                )
                context_parts.append(f"{conv_text}\n---\n")
        
        # Add current conversation
        context_parts.append("Current conversation:\n")
        for msg in current_conversation:
            context_parts.append(f"[{msg.get('timestamp', 'No timestamp')}] {msg['speaker']}: {msg['content']}\n")
        
        # Complete the instruction
        context_parts.append(f"[{timestamp}] Me: ")
        input_text = "".join(context_parts)
        output_text = f"{target_response}"
        
        return {
            "instruction": f"{system_prompt}{metadata}\n",
            "input": input_text,
            "output": output_text,
            "timestamp": timestamp,
        }

class OpenAIFormatter(ModelFormatter):
    """Formatter for OpenAI fine-tuning"""
    
    def format_conversation(self, conversation: List[Dict]) -> Dict:
        # TODO: Implement OpenAI-specific formatting
        pass