"""
Memory module for NPC system using FAISS for long-term memory storage.
Implements significance filtering and time decay for memory retrieval.
"""
import os
import time
import json
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from openai import OpenAI

class FaissMemory:
    """
    Long-term memory system using FAISS vector database.
    Implements significance filtering and time-based decay for memory retrieval.
    """
    
    def __init__(
        self, 
        index_path: Optional[str] = None,
        embedding_dim: int = 1536,
        significance_threshold: float = 0.5,
        recent_time_boost: float = 1.2,
        recent_time_window: timedelta = timedelta(hours=24),
        client: Optional[OpenAI] = None
    ):
        """
        Initialize the FAISS memory system.
        
        Args:
            index_path: Path to save/load the FAISS index
            embedding_dim: Dimension of the embedding vectors
            significance_threshold: Threshold for significance filtering (0.0-1.0)
            recent_time_boost: Boost factor for recent memories
            recent_time_window: Time window for recent memory boost
            client: OpenAI client for embeddings and significance filtering
        """
        self.embedding_dim = embedding_dim
        self.significance_threshold = significance_threshold
        self.recent_time_boost = recent_time_boost
        self.recent_time_window = recent_time_window
        self.client = client or OpenAI()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # Memory storage
        self.memories = []
        
        # Load existing index if provided
        if index_path and os.path.exists(index_path):
            self.load(index_path)
            
        self.index_path = index_path
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def _check_significance(self, memory_text: str) -> float:
        """
        Check if a memory is significant enough to be stored.
        Uses GPT-3.5 to evaluate significance with the prompt:
        "Will this fact be useful later?"
        
        Returns:
            float: Significance score between 0.0 and 1.0
        """
        prompt = f"Will this fact be useful to remember later for a blacksmith NPC in a game?\n\nFact: {memory_text}\n\nRate from 0.0 to 1.0:"
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that evaluates the significance of memories. Respond only with a number between 0.0 and 1.0."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        # Extract the score from the response
        try:
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return min(max(score, 0.0), 1.0)  # Ensure score is between 0.0 and 1.0
        except (ValueError, AttributeError):
            # Default to threshold if parsing fails
            return self.significance_threshold
    
    def add(self, player_id: str, memory_text: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a memory to the database if it passes significance filtering.
        
        Args:
            player_id: ID of the player this memory is associated with
            memory_text: The text content of the memory
            metadata: Additional metadata for the memory
            
        Returns:
            bool: True if memory was added, False if filtered out
        """
        # Check significance
        significance = self._check_significance(memory_text)
        if significance < self.significance_threshold:
            return False
        
        # Get embedding
        embedding = self._get_embedding(memory_text)
        
        # Prepare memory object
        memory = {
            "player_id": player_id,
            "text": memory_text,
            "timestamp": datetime.now().isoformat(),
            "significance": significance,
            "metadata": metadata or {}
        }
        
        # Add to index and memories list
        self.index.add(np.array([embedding], dtype=np.float32))
        self.memories.append(memory)
        
        # Save index if path is provided
        if self.index_path:
            self.save(self.index_path)
            
        return True
    
    def search(
        self, 
        query: str, 
        player_id: Optional[str] = None,
        limit: int = 5,
        apply_time_decay: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using the query text.
        
        Args:
            query: The search query text
            player_id: Optional player ID to filter results
            limit: Maximum number of results to return
            apply_time_decay: Whether to apply time-based decay to scores
            
        Returns:
            List of memory objects with similarity scores
        """
        if not self.memories:
            return []
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Search in FAISS index
        D, I = self.index.search(
            np.array([query_embedding], dtype=np.float32), 
            min(limit * 2, len(self.memories))  # Get more results for filtering
        )
        
        # Process results
        results = []
        for i, (distance, idx) in enumerate(zip(D[0], I[0])):
            if idx >= len(self.memories):
                continue
                
            memory = self.memories[idx]
            
            # Filter by player_id if provided
            if player_id and memory["player_id"] != player_id:
                continue
            
            # Convert distance to similarity score (FAISS returns L2 distance)
            similarity = 1.0 / (1.0 + distance)
            
            # Apply time decay if enabled
            if apply_time_decay:
                memory_time = datetime.fromisoformat(memory["timestamp"])
                now = datetime.now()
                
                # Apply boost for recent memories
                if now - memory_time <= self.recent_time_window:
                    similarity *= self.recent_time_boost
            
            # Add to results
            results.append({
                **memory,
                "score": similarity
            })
            
            # Stop if we have enough results after filtering
            if len(results) >= limit:
                break
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:limit]
    
    def save(self, path: str) -> None:
        """Save the FAISS index and memories to disk."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save memories
        with open(f"{path}.json", "w") as f:
            json.dump(self.memories, f)
    
    def load(self, path: str) -> None:
        """Load the FAISS index and memories from disk."""
        # Load FAISS index
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        
        # Load memories
        if os.path.exists(f"{path}.json"):
            with open(f"{path}.json", "r") as f:
                self.memories = json.load(f)