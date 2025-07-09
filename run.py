"""
Run module for NPC system.
Provides a demo_dialog function to demonstrate the NPC system in action.
"""
import os
import json
import logging
import uuid
import time
from typing import Dict, Any, Optional, List, Union

from .agent_core import build_npc
from .orchestrator import run_loop
from .lore_kb import LoreKnowledgeBase, LoreItem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def setup_demo_lore() -> None:
    """
    Set up demo lore for the blacksmith NPC.
    Creates a LanceDB database with weapon recipes.
    """
    # Create lore knowledge base
    lore_kb = LoreKnowledgeBase(
        db_path="lancedb_demo",
        table_name="lore"
    )
    
    # Define weapon recipes
    weapon_recipes = [
        {
            "id": "steel_sword",
            "title": "Steel Sword Recipe",
            "content": "To craft a steel sword, you need:\n- 2 steel ingots\n- 1 leather strip\n- 1 wooden handle\n\nForge the steel into a blade, attach the handle, and wrap with leather.",
            "tags": ["weapon_recipe", "sword", "steel"]
        },
        {
            "id": "iron_dagger",
            "title": "Iron Dagger Recipe",
            "content": "To craft an iron dagger, you need:\n- 1 iron ingot\n- 1 leather strip\n- 1 small wooden handle\n\nForge the iron into a small blade, attach the handle, and wrap with leather.",
            "tags": ["weapon_recipe", "dagger", "iron"]
        },
        {
            "id": "elven_bow",
            "title": "Elven Bow Recipe",
            "content": "To craft an elven bow, you need:\n- 2 flexible wood pieces\n- 1 elven string\n- 1 leather grip\n\nCarve the wood into a curved shape, attach the string, and wrap the grip with leather.",
            "tags": ["weapon_recipe", "bow", "elven"]
        },
        {
            "id": "dwarven_hammer",
            "title": "Dwarven Hammer Recipe",
            "content": "To craft a dwarven hammer, you need:\n- 3 dwarven metal ingots\n- 1 oak handle\n- 1 leather grip\n\nForge the metal into a hammer head, attach the handle, and wrap the grip with leather.",
            "tags": ["weapon_recipe", "hammer", "dwarven"]
        },
        {
            "id": "magic_staff",
            "title": "Magic Staff Recipe",
            "content": "To craft a magic staff, you need:\n- 1 enchanted wood\n- 1 magic crystal\n- 2 silver bands\n\nCarve the wood into a staff shape, attach the crystal at the top, and secure with silver bands.",
            "tags": ["weapon_recipe", "staff", "magic"]
        }
    ]
    
    # Add weapon recipes to knowledge base
    lore_kb.add_lore(weapon_recipes)
    
    logger.info(f"Added {len(weapon_recipes)} weapon recipes to lore database")

def mock_inventory_api(player_id: str, items: List[Dict[str, Any]], tx_id: str) -> Dict[str, Any]:
    """
    Mock inventory API for demo purposes.
    
    Args:
        player_id: ID of the player
        items: List of items to add to inventory
        tx_id: Transaction ID for idempotency
        
    Returns:
        Mock API response
    """
    logger.info(f"Mock inventory update for player {player_id}: {items}")
    return {
        "success": True,
        "player_id": player_id,
        "items_added": len(items),
        "tx_id": tx_id
    }

def demo_dialog() -> None:
    """
    Run a demo dialog with the blacksmith NPC.
    Sets up a demo environment and simulates a conversation.
    """
    # Set up demo lore
    setup_demo_lore()
    
    # Create NPC agent
    npc_agent = build_npc(
        npc_name="Master Hephaestus",
        npc_description="A legendary blacksmith known for crafting magical weapons and armor.",
        memory_path="memory_demo",
        lore_db_path="lancedb_demo",
        api_base_url="http://localhost:8000",  # Not used in demo
        debug_mode=True
    )
    
    # Mock the inventory API
    npc_agent.tools["update_inventory"].func = mock_inventory_api
    
    # Define demo conversation
    demo_conversation = [
        "Hello, blacksmith! My name is Aldric. I'm looking for a new weapon.",
        "Can you tell me about the different types of weapons you can craft?",
        "I'm interested in learning more about magic staffs. Do you have any recipes for those?",
        "That sounds perfect! Can you craft a magic staff for me?",
        "Thank you! I'll remember that you're the best blacksmith in town."
    ]
    
    # Generate player ID and conversation ID
    player_id = f"demo_player_{uuid.uuid4().hex[:8]}"
    conversation_id = f"demo_conversation_{uuid.uuid4().hex[:8]}"
    
    # Run demo conversation
    print("\n" + "="*50)
    print(f"DEMO CONVERSATION WITH {npc_agent.name}")
    print("="*50 + "\n")
    
    for i, user_input in enumerate(demo_conversation, 1):
        print(f"\n[USER {i}/{len(demo_conversation)}]: {user_input}\n")
        
        # Process message with NPC agent
        try:
            response, metadata = run_loop(
                agent=npc_agent,
                user_input=user_input,
                player_id=player_id,
                conversation_id=conversation_id,
                max_tokens_per_round=6000,
                max_tool_calls=4,
                max_loops=10,
                debug_mode=True
            )
            
            # Print response
            print(f"[NPC]: {response}\n")
            
            # Print metadata
            print(f"Metadata:")
            print(f"- Token usage: {metadata['token_usage']['total']} tokens")
            print(f"- Tool calls: {metadata['tool_calls']}")
            print(f"- Loops: {metadata['loops']}")
            print(f"- Duration: {metadata['duration']:.2f} seconds")
            
            if metadata.get("langtrace_url"):
                print(f"- Langtrace URL: {metadata['langtrace_url']}")
            
            # Add a delay between messages
            if i < len(demo_conversation):
                print("\nWaiting for next message...\n")
                time.sleep(2)
                
        except Exception as e:
            logger.error(f"Error in demo dialog: {str(e)}", exc_info=True)
            print(f"[ERROR]: {str(e)}")
            break
    
    print("\n" + "="*50)
    print("DEMO CONVERSATION COMPLETED")
    print("="*50 + "\n")

if __name__ == "__main__":
    demo_dialog()