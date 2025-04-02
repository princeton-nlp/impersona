import json 
import os
import re
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

CACHE_DIR = "data/cache"

class MemoryManager():
    def __init__(self, first_order_memories, second_order_memories, top_order_memories, embeddings, model="gpt-4o"):
        """
        Initialize the memory manager with hierarchical memories.
        
        Args:
            first_order_memories: List of first-order memory attributes
            second_order_memories: List of second-order memory attributes
            top_order_memories: List of top-level memory abstractions
            embeddings: Dictionary containing embeddings for all memory levels
            model: The OpenAI model to use for all API requests (default: "gpt-4o")
        """
        self.first_order_memories = first_order_memories
        self.second_order_memories = second_order_memories
        self.top_order_memories = top_order_memories
        self.embeddings = embeddings
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.model = model
        # Simplified cache structure
        self.last_conversation = None
        self.last_memory_summary = None
        
    def _get_embedding(self, text):
        """Get embedding for a text string"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)
    
    def _retrieve_relevant_memories(self, query_embedding, memory_list, memory_embeddings, top_k=5):
        """Retrieve most relevant memories based on embedding similarity"""
        similarities = []
        for i, memory_embedding in enumerate(memory_embeddings):
            similarity = self._cosine_similarity(query_embedding, memory_embedding)
            similarities.append((similarity, memory_list[i]))
        
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in similarities[:top_k]]
    
    def _check_cache(self, conversation):
        """Check if the last retrieved memories are sufficient for the current conversation"""
        if not self.last_conversation or not self.last_memory_summary:
            return None
            
        # Get embedding for current conversation
        current_embedding = self._get_embedding(conversation)
        
        # Check if our last retrieved memories are sufficient for this query
        cached_embedding = self._get_embedding(self.last_conversation)
        similarity = self._cosine_similarity(current_embedding, cached_embedding)
        
        # If similarity is high enough, use LLM to determine if cached memories are sufficient
        if similarity > 0.7:  # Lower threshold to consider more potential matches
            prompt = f"""
            Given the following conversation:
            
            {conversation}
            
            And these previously retrieved memories:
            
            {self.last_memory_summary}
            
            Are these memories sufficient to address the current conversation?
            Answer with only 'Yes' or 'No'.
            """
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a memory management agent that helps determine if cached memories are sufficient."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            answer = response.choices[0].message.content.strip().lower()
            if "yes" in answer:
                return self.last_memory_summary
                
        return None
    
    def search_memory(self, conversation):
        """
        Search memories in an agentic fashion based on the current conversation.
        
        Args:
            conversation: String containing the current conversation
            max_memories: Maximum number of memories to return
            
        Returns:
            String containing relevant memories
        """
        # Check cache first
        cached_results = self._check_cache(conversation)
        if cached_results:
            return cached_results
        
        # Get conversation embedding
        conversation_embedding = self._get_embedding(conversation)
        
        # First, identify relevant top-level abstractions
        top_memories = self._retrieve_relevant_memories(
            conversation_embedding, 
            self.top_order_memories, 
            self.embeddings['top_order'],
            top_k=3
        )
        
        # Use LLM to determine which top memories are most relevant and need expansion
        top_memories_text = "\n".join([f"{i+1}. {m['attribute']}" for i, m in enumerate(top_memories)])
        
        prompt = f"""
        Given the following conversation:
        
        {conversation}
        
        And these top-level memory abstractions:
        
        {top_memories_text}
        
        Which of these memory abstractions are most relevant to the conversation? 
        Identify the numbers of the abstractions that should be expanded to retrieve more detailed memories.
        Explain your reasoning briefly for each selection.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a memory management agent that helps retrieve relevant memories."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        analysis = response.choices[0].message.content
        
        # Extract the numbers of memories to expand
        to_expand = []
        for i, memory in enumerate(top_memories):
            if f"{i+1}." in analysis and "not relevant" not in analysis.lower():
                to_expand.append(memory)
        
        # If no memories were selected, use the top 2
        if not to_expand and top_memories:
            to_expand = top_memories[:2]
        
        # For each selected top memory, retrieve related second-order memories
        second_order_results = []
        for top_memory in to_expand:
            # Create embedding for this specific memory
            memory_embedding = self._get_embedding(top_memory['attribute'])
            
            # Find related second-order memories
            related_second = self._retrieve_relevant_memories(
                memory_embedding,
                self.second_order_memories,
                self.embeddings['second_order'],
                top_k=3
            )
            
            second_order_results.extend(related_second)
        
        # For each second-order memory, find related first-order memories
        first_order_results = []
        for second_memory in second_order_results:
            memory_embedding = self._get_embedding(second_memory['attribute'])
            
            related_first = self._retrieve_relevant_memories(
                memory_embedding,
                self.first_order_memories,
                self.embeddings['first_order'],
                top_k=2
            )
            
            first_order_results.extend(related_first)
        
        # Combine all retrieved memories
        all_memories = top_memories + second_order_results + first_order_results
        
        # Use LLM to create a summary of relevant information instead of just selecting memory numbers
        all_memories_text = "\n".join([
            f"{i+1}. [{m.get('timestamp', 'Unknown date')}] {m['attribute']}" 
            for i, m in enumerate(all_memories)
        ])
        
        final_prompt = f"""
        Given the following conversation:
        
        {conversation}
        
        And these retrieved memories:
        
        {all_memories_text}
        
        Create a concise summary of the information from these memories that is most relevant to the conversation.
        Focus on providing helpful context that would assist in continuing this conversation effectively.
        Include specific details like dates, names, and facts when they are important.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a memory management agent that helps retrieve and summarize relevant memories."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.3
        )
        
        memory_summary = response.choices[0].message.content
        
        # Update cache with the latest conversation and memory summary
        self.last_conversation = conversation
        self.last_memory_summary = memory_summary
        
        return memory_summary

class HierarchicalMemoryModule():
    def __init__(self, attribute_path, model="gpt-4o"):
        assert attribute_path, "Attribute path is required"
        assert os.path.exists(attribute_path), "Attribute path does not exist"

        os.makedirs(CACHE_DIR, exist_ok=True)
        
        no_timestamp_count = 0
        with open(attribute_path, 'r') as f:
            attribute_dict = json.load(f)
        
        first_order_attributes = attribute_dict['original_attributes']
        second_order_attributes = attribute_dict['combined_attributes']
        second_order_attributes.extend(attribute_dict['inferred_attributes'])
        top_order_attributes = attribute_dict['top_memories']

        for attribute in second_order_attributes:
            if 'Timestamp' in attribute:
                timestamp = attribute.pop('Timestamp')
                attribute['timestamp'] = timestamp
            
            if 'citation' in attribute:
                attribute['citations'] = attribute.pop('citation')
            
            if 'timestamp' not in attribute:
                attribute['timestamp'] = extract_timestamp(attribute)
                if not attribute['timestamp']:
                    no_timestamp_count += 1
            
            if 'citations' not in attribute:
                attribute['citations'] = []
        
        for attribute in first_order_attributes:
            if 'Timestamp' in attribute:
                timestamp = attribute.pop('Timestamp')
                attribute['timestamp'] = timestamp
            
            if 'citation' in attribute:
                attribute['citations'] = attribute.pop('citation')
            
            if 'timestamp' not in attribute:
                attribute['timestamp'] = extract_timestamp(attribute)
                if not attribute['timestamp']:
                    no_timestamp_count += 1
            
            if 'citations' not in attribute:
                attribute['citations'] = []
        
        for attribute in top_order_attributes:
            if 'Timestamp' in attribute:
                timestamp = attribute.pop('Timestamp')
                attribute['timestamp'] = timestamp
            
            if 'citations' not in attribute:
                attribute['citations'] = []
        
        print(f"No timestamp count: {no_timestamp_count}")

        self.first_order_attributes = first_order_attributes
        self.second_order_attributes = second_order_attributes
        self.top_order_attributes = top_order_attributes


        # Use the cache directory for storing embeddings
        base_filename = os.path.basename(attribute_path)
        cache_filename = f"{os.path.splitext(base_filename)[0]}_embeddings.json"
        self._embeddings_cache_path = os.path.join(CACHE_DIR, cache_filename)
        
        # Load or calculate embeddings during initialization
        self._embeddings = self._load_embeddings()

        # Initialize the memory manager with all memory levels
        self.memory_manager = MemoryManager(
            self.first_order_attributes, 
            self.second_order_attributes, 
            self.top_order_attributes, 
            self._embeddings,
            model=model
        )

    def _load_embeddings(self, batch_size=100, force_recalculate=False):
        """Load embeddings from cache or calculate them if needed"""
        # Check if embeddings cache file exists
        if os.path.exists(self._embeddings_cache_path) and not force_recalculate:
            try:
                print(f"Loading embeddings from cache: {self._embeddings_cache_path}")
                with open(self._embeddings_cache_path, 'r') as f:
                    cached_data = json.load(f)
                    return {
                        'first_order': [np.array(emb) for emb in cached_data['first_order']],
                        'second_order': [np.array(emb) for emb in cached_data['second_order']],
                        'top_order': [np.array(emb) for emb in cached_data['top_order']]
                    }
            except Exception as e:
                print(f"Error loading cached embeddings: {e}. Recalculating...")
        
        # Calculate embeddings if not cached or force_recalculate is True
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        embeddings = {
            'first_order': [],
            'second_order': [],
            'top_order': []
        }
        
        # Process first order attributes
        for i in range(0, len(self.first_order_attributes), batch_size):
            batch = self.first_order_attributes[i:i+batch_size]
            texts = [memory['attribute'] for memory in batch]
            
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            embeddings['first_order'].extend(batch_embeddings)
        
        # Process second order attributes
        for i in range(0, len(self.second_order_attributes), batch_size):
            batch = self.second_order_attributes[i:i+batch_size]
            texts = [memory['attribute'] for memory in batch]
            
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            embeddings['second_order'].extend(batch_embeddings)
        
        # Process top order attributes
        for i in range(0, len(self.top_order_attributes), batch_size):
            batch = self.top_order_attributes[i:i+batch_size]
            texts = [memory['attribute'] for memory in batch]
            
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            embeddings['top_order'].extend(batch_embeddings)
        
        # Save embeddings to cache file
        try:
            # Convert numpy arrays to lists for JSON serialization
            embeddings_list = {
                'first_order': [emb.tolist() for emb in embeddings['first_order']],
                'second_order': [emb.tolist() for emb in embeddings['second_order']],
                'top_order': [emb.tolist() for emb in embeddings['top_order']]
            }
            with open(self._embeddings_cache_path, 'w') as f:
                json.dump(embeddings_list, f)
            print(f"Saved embeddings to cache: {self._embeddings_cache_path}")
        except Exception as e:
            print(f"Error saving embeddings to cache: {e}")
            
        return embeddings

    # Add a method to search memories using the memory manager
    def search_memory(self, query):
        """Search memories using the hierarchical memory manager"""
        return self.memory_manager.search_memory(query)

class MemoryModule():
    def __init__(self, attribute_path):
        assert attribute_path, "Attribute path is required"
        assert os.path.exists(attribute_path), "Attribute path does not exist"

        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        
        no_timestamp_count = 0
        with open(attribute_path, 'r') as f:
            attribute_dict = json.load(f)
        
        attributes = []
        attributes.extend(attribute_dict['combined_attributes'])
        attributes.extend(attribute_dict['inferred_attributes'])
        # attributes.extend(attribute_dict['original_attributes'])
        # no_timestamp = []
        for i, attribute in enumerate(attributes):
            if 'Timestamp' in attribute:
                timestamp = attribute.pop('Timestamp')
                attribute['timestamp'] = timestamp
            
            if 'citation' in attribute:
                attribute['citations'] = attribute.pop('citation')
            
            if 'timestamp' not in attribute:
                attribute['timestamp'] = extract_timestamp(attribute)
                if not attribute['timestamp']:
                    no_timestamp_count += 1
                    # no_timestamp.append(i)
            
            if 'citations' not in attribute:
                attribute['citations'] = []
            
            assert 'attribute' in attribute, "Attribute must contain an attribute key"
            assert 'citations' in attribute, "Attribute must contain a citation key"
            assert 'timestamp' in attribute, "Attribute must contain a timestamp key"
        
        print(f"No timestamp count: {no_timestamp_count}")

        # no_timestamp = sorted(no_timestamp, reverse=True)
        # for i in no_timestamp:
        #     attributes.pop(i)

        assert attributes

        self.memories = attributes
        
        # Use the cache directory for storing embeddings
        base_filename = os.path.basename(attribute_path)
        cache_filename = f"{os.path.splitext(base_filename)[0]}_embeddings.json"
        self._embeddings_cache_path = os.path.join(CACHE_DIR, cache_filename)
        
        # Load or calculate embeddings during initialization
        self._embeddings = self._load_embeddings()
    
    def _load_embeddings(self, batch_size=100, force_recalculate=False):
        """Load embeddings from cache or calculate them if needed"""
        # Check if embeddings cache file exists
        if os.path.exists(self._embeddings_cache_path) and not force_recalculate:
            try:
                print(f"Loading embeddings from cache: {self._embeddings_cache_path}")
                with open(self._embeddings_cache_path, 'r') as f:
                    return [np.array(emb) for emb in json.load(f)]
            except Exception as e:
                print(f"Error loading cached embeddings: {e}. Recalculating...")
        
        # Calculate embeddings if not cached or force_recalculate is True
        embeddings = []
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        for i in range(0, len(self.memories), batch_size):
            batch = self.memories[i:i+batch_size]
            texts = [memory['attribute'] for memory in batch]
            
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            
            batch_embeddings = [np.array(item.embedding) for item in response.data]
            embeddings.extend(batch_embeddings)
        
        # Save embeddings to cache file
        try:
            # Convert numpy arrays to lists for JSON serialization
            embeddings_list = [emb.tolist() for emb in embeddings]
            with open(self._embeddings_cache_path, 'w') as f:
                json.dump(embeddings_list, f)
            print(f"Saved embeddings to cache: {self._embeddings_cache_path}")
        except Exception as e:
            print(f"Error saving embeddings to cache: {e}")
            
        return embeddings

    def get_embeddings(self, batch_size=100, force_recalculate=False):
        """Return embeddings, recalculating if requested"""
        if force_recalculate:
            self._embeddings = self._load_embeddings(batch_size, force_recalculate=True)
        return self._embeddings

    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        import numpy as np
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        return dot_product / (norm_a * norm_b)
    
    def _keyword_search(self, query, top_k):
        """Simple fallback search using keyword matching"""
        query_words = set(query.lower().split())
        results = []
        
        for memory in self.memories:
            attribute = memory['attribute'].lower()
            # Count how many query words appear in the attribute
            matches = sum(1 for word in query_words if word in attribute)
            if matches > 0:
                results.append((matches, memory))
        
        # Sort by number of matches (highest first)
        results.sort(reverse=True, key=lambda x: x[0])
        return [item[1] for item in results[:top_k]]

    def search_memory(self, query: str, top_k: int = 6, recency_weight: float = 0.15) -> str:
        """
        Search memories for relevant attributes based on the query.
        
        Args:
            query: The search query string
            top_k: Number of top results to return (default: 10)
            recency_weight: Weight given to recency in the final score (0-1, default: 0.15)
            
        Returns:
            List of top_k most relevant attribute dictionaries
        """
        # Get embeddings (already loaded during initialization)
        embeddings = self._embeddings
        
        # Verify embeddings match memories
        if len(embeddings) != len(self.memories):
            print(f"Warning: Embeddings count ({len(embeddings)}) doesn't match memories count ({len(self.memories)})")
            # Recalculate embeddings to ensure they match
            embeddings = self._load_embeddings(force_recalculate=True)
        
        if len(embeddings) == 0:  # Check length instead of boolean evaluation
            # Fallback to simple keyword matching if embeddings failed
            return self._keyword_search(query, top_k)
        
        # Get query embedding
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"  # Use the same model as for memories
        )
        query_embedding = np.array(response.data[0].embedding)
        
        # Calculate similarity scores and incorporate recency
        # Find the newest and oldest timestamps to normalize recency scores
        timestamps = []
        for memory in self.memories:
            try:
                # Try to parse the timestamp
                ts = memory['timestamp']
                if ts:
                    dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(dt)
            except (ValueError, TypeError):
                # Skip invalid timestamps
                continue
        
        if timestamps:
            newest = max(timestamps)
            oldest = min(timestamps)
            time_range = (newest - oldest).total_seconds()
            
            # If all memories have the same timestamp, avoid division by zero
            if time_range == 0:
                time_range = 1
        
        # Calculate combined scores
        combined_scores = []
        for i, memory in enumerate(self.memories):
            # Calculate similarity score
            similarity = self._cosine_similarity(query_embedding, embeddings[i])
            
            # Calculate recency score (0 to 1, where 1 is most recent)
            recency_score = 0.0  # Default for memories with no valid timestamp
            try:
                if timestamps and memory['timestamp']:
                    dt = datetime.strptime(memory['timestamp'], "%Y-%m-%d %H:%M:%S")
                    recency_score = (dt - oldest).total_seconds() / time_range
            except (ValueError, TypeError):
                # Use default recency score for invalid timestamps
                pass
            
            # Combine scores: (1-w)*similarity + w*recency
            combined_score = (1 - recency_weight) * similarity + recency_weight * recency_score
            combined_scores.append((combined_score, memory))
        
        # Sort by combined score (highest first) and get top_k results
        combined_scores.sort(reverse=True, key=lambda x: x[0])
        top_results = [item[1] for item in combined_scores[:top_k]]
        
        output_string = ""
        for result in top_results:
            timestamp = result.get('timestamp', '')
            output_string = output_string + f"[{timestamp}] {result['attribute']}\n"
        return output_string

def extract_timestamp(attribute):
    """
    Extract timestamp from citation text when the timestamp field is missing.
    
    Args:
        attribute: Dictionary containing attribute information
        
    Returns:
        Extracted timestamp string or empty string if not found
    """
    try:
        if 'source_attributes' in attribute:
            for a in attribute['source_attributes']:
                if 'timestamp' in a:
                    return a['timestamp']
        # If we have citations array
        if 'citations' in attribute and isinstance(attribute['citations'], list) and len(attribute['citations']) > 0:
            for citation_text in attribute['citations']:
                if isinstance(citation_text, str):
                    # Look for explicit timestamp pattern
                    timestamp_match = re.search(r'Timestamp:\s*(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?)', citation_text)
                    if timestamp_match:
                        return timestamp_match.group(1)
                    
                    # Try to find any date-time pattern
                    date_match = re.search(r'\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?', citation_text)
                    if date_match:
                        return date_match.group(0)
        
        # Check if we have a single citation field
        if 'citation' in attribute:
            citation = attribute['citation']
            # Look for timestamp pattern in citation
            timestamp_match = re.search(r'Timestamp:\s*(\d{4}-\d{2}-\d{2}\s+\d{1,2}:\d{2}(?::\d{2})?)', citation)
            if timestamp_match:
                return timestamp_match.group(1)
            
            # Try to find any date-time pattern in the citation
            date_match = re.search(r'\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?', citation)
            if date_match:
                return date_match.group(0)
    except Exception as e:
        print(f"Error extracting timestamp: {e}")
        return ""
    # Return empty string if no timestamp found
    return ""