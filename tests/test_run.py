"""
Tests for the NPC system.
Verifies that the conversation loop can return a string.
"""
import os
import pytest
from unittest.mock import MagicMock, patch

import agno
from agno import Agent, Message

from NPCango.agent_core import build_npc
from NPCango.orchestrator import run_loop

# Mock agent response
class MockAgentResponse:
    def __init__(self, content, tool_calls=None, usage=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.usage = usage or {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

# Test that the conversation loop can return a string
@patch("agno.Agent")
def test_conversation_loop_returns_string(mock_agent_class):
    """Test that the conversation loop can return a string."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent
    
    # Set up mock agent response
    mock_agent.respond.return_value = MockAgentResponse(
        content="Hello, I'm the blacksmith. How can I help you today?"
    )
    
    # Create agent
    agent = build_npc(
        npc_name="Test Blacksmith",
        npc_description="A test blacksmith",
        memory_path=None,
        lore_db_path=None,
        api_base_url="http://localhost:8000",
        model="gpt-4o",
        debug_mode=True
    )
    
    # Run conversation loop
    response, metadata = run_loop(
        agent=agent,
        user_input="Hello, blacksmith!",
        player_id="test_player",
        conversation_id="test_conversation",
        max_tokens_per_round=6000,
        max_tool_calls=4,
        max_loops=10,
        debug_mode=True
    )
    
    # Assert response is a string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Assert metadata is correct
    assert metadata["player_id"] == "test_player"
    assert metadata["conversation_id"] == "test_conversation"
    assert "token_usage" in metadata
    assert "tool_calls" in metadata
    assert "loops" in metadata
    assert "duration" in metadata

# Test with tool calls
@patch("agno.Agent")
def test_conversation_loop_with_tool_calls(mock_agent_class):
    """Test that the conversation loop handles tool calls correctly."""
    # Create mock agent
    mock_agent = MagicMock()
    mock_agent_class.return_value = mock_agent
    
    # Set up mock agent responses
    mock_agent.respond.side_effect = [
        # First response with tool call
        MockAgentResponse(
            content="Let me check my knowledge about swords.",
            tool_calls=[
                MagicMock(name="search_lore", arguments="sword")
            ]
        ),
        # Second response after tool call
        MockAgentResponse(
            content="I can craft a steel sword for you. Would you like one?"
        )
    ]
    
    # Set up mock tool execution
    mock_agent.execute_tool.return_value = "I found information about steel swords."
    
    # Create agent
    agent = build_npc(
        npc_name="Test Blacksmith",
        npc_description="A test blacksmith",
        memory_path=None,
        lore_db_path=None,
        api_base_url="http://localhost:8000",
        model="gpt-4o",
        debug_mode=True
    )
    
    # Run conversation loop
    response, metadata = run_loop(
        agent=agent,
        user_input="Can you tell me about swords?",
        player_id="test_player",
        conversation_id="test_conversation",
        max_tokens_per_round=6000,
        max_tool_calls=4,
        max_loops=10,
        debug_mode=True
    )
    
    # Assert response is a string
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Assert tool calls were made
    assert metadata["tool_calls"] == 1
    assert metadata["loops"] == 2

# Run tests if file is executed directly
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])