#!/usr/bin/env python
"""
Quick test script for NPC functionality.
Tests memory, lore search, and inventory management.
This is a standalone script that doesn't rely on the agno package.
"""
import os
import sys
import json
import logging
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

import numpy as np
import faiss
import os
import json
from datetime import datetime

class SimpleFaissMemory:
    """
    Simplified FAISS memory system that saves memories to a file.
    """
    
    def __init__(self, index_path: str = "memory_faiss"):
        """
        Initialize the FAISS memory system.
        
        Args:
            index_path: Path to save/load the FAISS index
        """
        self.index_path = index_path
        self.embedding_dim = 1536  # OpenAI embedding dimension
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Memory storage
        self.memories = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)
        
        # Load existing index if it exists
        if os.path.exists(f"{index_path}.index") and os.path.exists(f"{index_path}.json"):
            self.load(index_path)
            print(f"[FAISS] Loaded {len(self.memories)} memories from {index_path}")
        else:
            print(f"[FAISS] Created new memory index at {index_path}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Get a simple embedding for text.
        
        In a real implementation, this would use OpenAI's embedding API.
        Here we just create a random vector for demonstration.
        """
        # Create a random embedding (for demonstration only)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def add(self, player_id: str, fact: str) -> bool:
        """
        Add a memory to the database.
        
        Args:
            player_id: ID of the player this memory is associated with
            fact: The text content of the memory
            
        Returns:
            bool: True if memory was added
        """
        # Get embedding
        embedding = self._get_embedding(fact)
        
        # Prepare memory object
        memory = {
            "player_id": player_id,
            "text": fact,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to index and memories list
        self.index.add(np.array([embedding], dtype=np.float32))
        self.memories.append(memory)
        
        # Save index
        self.save(self.index_path)
        
        # Print all memories for this player
        self.print_memories(player_id)
        
        return True
    
    def search(self, query: str, player_id: str = None, limit: int = 5) -> list:
        """
        Search for relevant memories using the query text.
        
        Args:
            query: The search query text
            player_id: Optional player ID to filter results
            limit: Maximum number of results to return
            
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
        
        # Print all memories for this player
        self.print_memories(player_id)
        
        return results[:limit]
    
    def print_memories(self, player_id: str = None):
        """Print all memories for a player."""
        if player_id:
            player_memories = [m for m in self.memories if m["player_id"] == player_id]
            print(f"\n[FAISS] All memories for player '{player_id}' ({len(player_memories)} total):")
            for i, memory in enumerate(player_memories, 1):
                print(f"  {i}. {memory['text']} (Added: {memory['timestamp']})")
        else:
            print(f"\n[FAISS] All memories ({len(self.memories)} total):")
            for i, memory in enumerate(self.memories, 1):
                print(f"  {i}. [{memory['player_id']}] {memory['text']} (Added: {memory['timestamp']})")
    
    def save(self, path: str):
        """Save the FAISS index and memories to disk."""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save memories
        with open(f"{path}.json", "w") as f:
            json.dump(self.memories, f)
        
        print(f"[FAISS] Saved {len(self.memories)} memories to {path}")
    
    def load(self, path: str):
        """Load the FAISS index and memories from disk."""
        # Load FAISS index
        if os.path.exists(f"{path}.index"):
            self.index = faiss.read_index(f"{path}.index")
        
        # Load memories
        if os.path.exists(f"{path}.json"):
            with open(f"{path}.json", "r") as f:
                self.memories = json.load(f)

# Initialize the FAISS memory system
faiss_memory = SimpleFaissMemory("memory_faiss")

def remember_fact(player_id: str, fact: str) -> str:
    """Implementation of remember_fact tool using FAISS."""
    success = faiss_memory.add(player_id, fact)
    print(f"[Tool] remember_fact: Stored '{fact}' for player '{player_id}'")
    return f"I'll remember that {fact}"

def recall_memory(player_id: str, query: str) -> str:
    """Implementation of recall_memory tool using FAISS."""
    memories = faiss_memory.search(query, player_id=player_id)
    
    if not memories:
        print(f"[Tool] recall_memory: No memories found for player '{player_id}'")
        return "I don't recall anything about that."
    
    print(f"[Tool] recall_memory: Found {len(memories)} memories for player '{player_id}'")
    
    # Format memories
    memory_texts = []
    for i, mem in enumerate(memories, 1):
        memory_texts.append(f"[{i}] {mem['text']} (Relevance: {mem['score']:.2f})")
    
    return "Here's what I remember:\n" + "\n".join(memory_texts)

def search_lore(query: str, tag_filter: str = "weapon_recipe") -> str:
    """Stub implementation of search_lore tool."""
    # Simplified lore database for testing
    lore_db = {
        "破甲长矛": {
            "title": "破甲长矛制作配方",
            "content": "制作破甲长矛需要以下材料：\n- 精炼钢锭 x5\n- 秘银合金 x2\n- 龙骨碎片 x1\n- 精工木柄 x1",
            "relevance": 0.92
        },
        "钢剑": {
            "title": "钢剑制作配方",
            "content": "制作钢剑需要以下材料：\n- 钢锭 x3\n- 皮革条 x1\n- 木柄 x1",
            "relevance": 0.85
        }
    }
    
    # Simple keyword matching
    results = []
    for key, item in lore_db.items():
        if key in query or query in key:
            results.append(item)
    
    if not results:
        print(f"[Tool] search_lore: No results found for query '{query}'")
        return "No relevant information found in the lore database."
    
    # Format results
    print(f"[Tool] search_lore: Found {len(results)} results for query '{query}'")
    formatted_results = []
    for i, result in enumerate(results, 1):
        formatted_results.append(
            f"[{i}] {result['title']}\n"
            f"Relevance: {result['relevance']:.2f}\n"
            f"{result['content']}\n"
        )
    
    return "\n".join(formatted_results)

def update_inventory(player_id: str, item_id: str, quantity: int = 1) -> str:
    """Stub implementation of update_inventory tool."""
    print(f"[Stub] pretend to add {quantity}x{item_id} to {player_id}")
    return "已假装写入"

def main():
    """
    Run a quick test of the NPC functionality.
    
    Tests:
    1. Memory: Remember player name
    2. Memory: Recall player name
    3. Lore: Search for weapon recipe
    4. Inventory: Add items to inventory
    """
    # Check for OPENAI_API_KEY
    if "OPENAI_API_KEY" not in os.environ:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it with: export OPENAI_API_KEY=your_api_key")
        sys.exit(1)
    
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: OpenAI package not found. Please install it with: pip install openai>=1.0.0")
        sys.exit(1)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    # System message for the NPC
    system_message = """You are a Blacksmith, a skilled blacksmith who can craft weapons and armor.

You are an NPC in a fantasy game world. You interact with players through conversation.
You can remember important facts about players, search for weapon recipes and lore,
and add items to players' inventories.

When a player asks about crafting a weapon, use the search_lore tool to find recipes.
When you need to remember something about a player, use the remember_fact tool.
When you need to recall something about a player, use the recall_memory tool.
When you craft an item for a player, use the update_inventory tool to add it to their inventory.

Always stay in character as a blacksmith. Be helpful but don't break character.
"""
    
    # Test messages
    test_messages = [
        "师傅，我叫李逍遥，请记住我！",
        "你还记得我是谁吗？",
        "我要做破甲长矛，需要哪些材料？",
        "给我 10 块钢锭"
    ]
    
    # Send messages and print responses
    player_id = "player123"  # Example player ID
    conversation_history = [{"role": "system", "content": system_message}]
    
    for i, msg_text in enumerate(test_messages, 1):
        print(f"\n--- Test Message {i} ---")
        print(f"Player: {msg_text}")
        
        # Add user message to conversation
        conversation_history.append({"role": "user", "content": msg_text})
        
        # Get response from OpenAI
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-0125",  # Use a model with function calling
                messages=conversation_history,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "remember_fact",
                            "description": "Remember an important fact about a player",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "player_id": {
                                        "type": "string",
                                        "description": "ID of the player"
                                    },
                                    "fact": {
                                        "type": "string",
                                        "description": "The fact to remember"
                                    }
                                },
                                "required": ["player_id", "fact"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "recall_memory",
                            "description": "Recall memories related to a player",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "player_id": {
                                        "type": "string",
                                        "description": "ID of the player"
                                    },
                                    "query": {
                                        "type": "string",
                                        "description": "What to recall"
                                    }
                                },
                                "required": ["player_id", "query"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "search_lore",
                            "description": "Search for lore in the knowledge base",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query"
                                    },
                                    "tag_filter": {
                                        "type": "string",
                                        "description": "Tag to filter results by",
                                        "default": "weapon_recipe"
                                    }
                                },
                                "required": ["query"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "update_inventory",
                            "description": "Add items to a player's inventory",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "player_id": {
                                        "type": "string",
                                        "description": "ID of the player"
                                    },
                                    "item_id": {
                                        "type": "string",
                                        "description": "ID of the item to add"
                                    },
                                    "quantity": {
                                        "type": "integer",
                                        "description": "Number of items to add",
                                        "default": 1
                                    }
                                },
                                "required": ["player_id", "item_id"]
                            }
                        }
                    }
                ],
                tool_choice="auto"
            )
            
            # Process the response
            message = response.choices[0].message
            
            # Add the assistant's message to the conversation history
            conversation_history.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
            })
            
            # Handle tool calls
            if message.tool_calls:
                # Process each tool call
                tool_results = []
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Call the appropriate function
                    if function_name == "remember_fact":
                        tool_result = remember_fact(
                            player_id=function_args.get("player_id", player_id),
                            fact=function_args.get("fact", "")
                        )
                    elif function_name == "recall_memory":
                        tool_result = recall_memory(
                            player_id=function_args.get("player_id", player_id),
                            query=function_args.get("query", "")
                        )
                    elif function_name == "search_lore":
                        tool_result = search_lore(
                            query=function_args.get("query", ""),
                            tag_filter=function_args.get("tag_filter", "weapon_recipe")
                        )
                    elif function_name == "update_inventory":
                        tool_result = update_inventory(
                            player_id=function_args.get("player_id", player_id),
                            item_id=function_args.get("item_id", ""),
                            quantity=function_args.get("quantity", 1)
                        )
                    else:
                        tool_result = f"Error: Unknown function {function_name}"
                    
                    # Add tool response to conversation
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": tool_result
                    })
                    
                    tool_results.append(f"{function_name}: {tool_result}")
                
                # Get a new response from the model
                second_response = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=conversation_history
                )
                
                # Add the final assistant message to the conversation history
                assistant_message = second_response.choices[0].message.content
                conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                
                # Print tool results for debugging
                for result in tool_results:
                    print(f"[Tool Result] {result}")
            else:
                # No tool calls, just use the content
                assistant_message = message.content
            
            # Print the response
            print(f"NPC: {assistant_message}")
            
        except Exception as e:
            logger.error(f"Error in test dialog: {str(e)}", exc_info=True)
            print(f"[ERROR]: {str(e)}")
            break

if __name__ == "__main__":
    main()