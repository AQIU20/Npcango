#!/usr/bin/env python
"""
Quick test script for NPC functionality.
Tests memory, lore search, and inventory management.
"""
import os
import sys
from NPCango.agent_core import build_npc
from agno import Message

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
    
    print("Initializing NPC in debug mode...")
    # Initialize NPC with debug_mode=True
    npc = build_npc(debug_mode=True)
    
    # Test messages
    test_messages = [
        "师傅，我叫李逍遥，请记住我！",
        "你还记得我是谁吗？",
        "我要做破甲长矛，需要哪些材料？",
        "给我 10 块钢锭"
    ]
    
    # Send messages and print responses
    player_id = "player123"  # Example player ID
    for i, msg_text in enumerate(test_messages, 1):
        print(f"\n--- Test Message {i} ---")
        print(f"Player: {msg_text}")
        
        # Create message object
        message = Message(role="user", content=msg_text)
        
        # Get response from NPC
        response = npc.chat(message, metadata={"player_id": player_id})
        
        print(f"NPC: {response.content}")

if __name__ == "__main__":
    main()