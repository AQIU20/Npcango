"""
Agent core module for NPC system.
Implements the build_npc function to create and configure the NPC agent.
"""
import os
from typing import Dict, Any, Optional, List, Union, Callable

from openai import OpenAI
import agno
from agno import Agent, Tool, Message
from agno.tools import ToolRegistry

from .memory import FaissMemory
from .lore_kb import LoreKnowledgeBase
from .tools import GameTools

def build_npc(
    npc_name: str = "Blacksmith",
    npc_description: str = "A skilled blacksmith who can craft weapons and armor.",
    memory_path: Optional[str] = "memory",
    lore_db_path: Optional[str] = "lancedb",
    api_base_url: str = "http://localhost:8000",
    openai_api_key: Optional[str] = None,
    openai_base_url: Optional[str] = None,
    model: str = "gpt-4o",
    use_local_llama: bool = False,
    local_llama_url: Optional[str] = None,
    debug_mode: bool = False
) -> Agent:
    """
    Build and configure an NPC agent with memory, knowledge base, and tools.
    
    Args:
        npc_name: Name of the NPC
        npc_description: Description of the NPC
        memory_path: Path to store/load memory
        lore_db_path: Path to LanceDB database
        api_base_url: Base URL for game API
        openai_api_key: OpenAI API key (defaults to env var)
        openai_base_url: OpenAI API base URL (for proxies)
        model: Model to use (default: gpt-4o)
        use_local_llama: Whether to use a local Llama model
        local_llama_url: URL for local Llama API
        debug_mode: Whether to enable debug mode
        
    Returns:
        Configured Agno agent
    """
    # Set up OpenAI client
    openai_client = OpenAI(
        api_key=openai_api_key or os.environ.get("OPENAI_API_KEY"),
        base_url=openai_base_url
    )
    
    # Initialize memory system
    memory = FaissMemory(
        index_path=memory_path,
        client=openai_client
    )
    
    # Initialize knowledge base
    lore_kb = LoreKnowledgeBase(
        db_path=lore_db_path,
        client=openai_client
    )
    
    # Initialize game tools
    game_tools = GameTools(
        api_base_url=api_base_url
    )
    
    # Create tool registry
    tool_registry = ToolRegistry()
    
    # Register memory tools
    @tool_registry.register
    def remember_fact(player_id: str, fact: str) -> str:
        """
        Remember an important fact about a player.
        
        Args:
            player_id: ID of the player
            fact: The fact to remember
            
        Returns:
            String indicating whether the fact was stored
        """
        success = memory.add(player_id, fact)
        if success:
            return f"I'll remember that {fact}"
        else:
            return "I don't think that's important enough to remember."
    
    @tool_registry.register
    def recall_memory(player_id: str, query: str) -> str:
        """
        Recall memories related to a player.
        
        Args:
            player_id: ID of the player
            query: What to recall
            
        Returns:
            String with relevant memories
        """
        memories = memory.search(query, player_id=player_id)
        
        if not memories:
            return "I don't recall anything about that."
        
        # Format memories
        memory_texts = []
        for i, mem in enumerate(memories, 1):
            memory_texts.append(f"[{i}] {mem['text']} (Relevance: {mem['score']:.2f})")
        
        return "Here's what I remember:\n" + "\n".join(memory_texts)
    
    # Register lore search tool
    tool_registry.register(lore_kb.create_search_lore_tool(), name="search_lore")
    
    # Register inventory tool
    tool_registry.register(game_tools.create_update_inventory_tool(), name="update_inventory")
    
    # Create system message
    system_message = f"""You are {npc_name}, {npc_description}

You are an NPC in a fantasy game world. You interact with players through conversation.
You can remember important facts about players, search for weapon recipes and lore,
and add items to players' inventories.

When a player asks about crafting a weapon, use the search_lore tool to find recipes.
When you need to remember something about a player, use the remember_fact tool.
When you need to recall something about a player, use the recall_memory tool.
When you craft an item for a player, use the update_inventory tool to add it to their inventory.

Always stay in character as a {npc_name.lower()}. Be helpful but don't break character.
"""

    # Configure the agent
    if use_local_llama and local_llama_url:
        # Use local Llama model
        agent = Agent(
            name=npc_name,
            system_message=system_message,
            tools=tool_registry.get_tools(),
            model="local",  # Use local model
            model_kwargs={
                "url": local_llama_url,
                "temperature": 0.7,
                "max_tokens": 1024
            },
            debug=debug_mode
        )
    else:
        # Use OpenAI model
        agent = Agent(
            name=npc_name,
            system_message=system_message,
            tools=tool_registry.get_tools(),
            model=model,
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 1024
            },
            openai_client=openai_client,
            debug=debug_mode
        )
    
    return agent