"""
Orchestrator module for NPC system.
Implements the run_loop function for managing conversation with token budget and loop limits.
"""
import time
import logging
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
import json

from openai import OpenAI
import agno
from agno import Agent, Message
from prometheus_client import Counter, Gauge, Histogram

# Set up logging
logger = logging.getLogger(__name__)

# Prometheus metrics
TOKEN_USAGE_GAUGE = Gauge(
    "npc_token_usage", 
    "Token usage per conversation",
    ["model", "type"]  # type: prompt, completion, total
)

CONVERSATION_DURATION_HISTOGRAM = Histogram(
    "npc_conversation_duration_seconds",
    "Duration of NPC conversations in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

TOOL_CALLS_COUNTER = Counter(
    "npc_tool_calls_total",
    "Total number of tool calls made by the NPC",
    ["tool_name"]
)

class TokenBudgetExceededError(Exception):
    """Exception raised when the token budget is exceeded."""
    pass

class MaxLoopsExceededError(Exception):
    """Exception raised when the maximum number of loops is exceeded."""
    pass

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text.
    This is a rough estimate based on words/4.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # Simple estimation: ~4 chars per token on average
    return len(text) // 4

def run_loop(
    agent: Agent,
    user_input: str,
    player_id: str,
    conversation_id: Optional[str] = None,
    max_tokens_per_round: int = 6000,
    max_tool_calls: int = 4,
    max_loops: int = 10,
    debug_mode: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Run a conversation loop with the NPC agent.
    
    Args:
        agent: The NPC agent
        user_input: The user's input message
        player_id: ID of the player
        conversation_id: ID for the conversation (for tracking)
        max_tokens_per_round: Maximum tokens per conversation round
        max_tool_calls: Maximum number of tool calls per conversation
        max_loops: Maximum number of conversation loops
        debug_mode: Whether to enable debug mode
        
    Returns:
        Tuple of (final_response, metadata)
        
    Raises:
        TokenBudgetExceededError: If the token budget is exceeded
        MaxLoopsExceededError: If the maximum number of loops is exceeded
    """
    start_time = time.time()
    
    # Initialize conversation
    conversation = [
        Message(role="user", content=user_input)
    ]
    
    # Initialize metadata
    metadata = {
        "player_id": player_id,
        "conversation_id": conversation_id,
        "token_usage": {
            "prompt": 0,
            "completion": 0,
            "total": 0
        },
        "tool_calls": 0,
        "loops": 0,
        "duration": 0,
        "langtrace_url": None
    }
    
    # Run conversation loop
    loop_count = 0
    tool_call_count = 0
    final_response = None
    
    while loop_count < max_loops:
        loop_count += 1
        metadata["loops"] = loop_count
        
        # Check if we've exceeded the maximum number of tool calls
        if tool_call_count >= max_tool_calls:
            logger.warning(f"Maximum tool calls ({max_tool_calls}) exceeded")
            # Force the agent to respond without using tools
            forced_response = agent.respond(
                conversation,
                force_no_tools=True
            )
            final_response = forced_response.content
            break
        
        # Get response from agent
        response = agent.respond(conversation)
        
        # Update token usage
        if hasattr(response, "usage") and response.usage:
            metadata["token_usage"]["prompt"] += response.usage.get("prompt_tokens", 0)
            metadata["token_usage"]["completion"] += response.usage.get("completion_tokens", 0)
            metadata["token_usage"]["total"] += response.usage.get("total_tokens", 0)
            
            # Update Prometheus metrics
            TOKEN_USAGE_GAUGE.labels(model=agent.model, type="prompt").set(metadata["token_usage"]["prompt"])
            TOKEN_USAGE_GAUGE.labels(model=agent.model, type="completion").set(metadata["token_usage"]["completion"])
            TOKEN_USAGE_GAUGE.labels(model=agent.model, type="total").set(metadata["token_usage"]["total"])
        else:
            # Estimate token usage if not provided
            estimated_tokens = estimate_tokens(response.content)
            metadata["token_usage"]["completion"] += estimated_tokens
            metadata["token_usage"]["total"] += estimated_tokens
        
        # Check if we've exceeded the token budget
        if metadata["token_usage"]["total"] > max_tokens_per_round:
            logger.warning(f"Token budget ({max_tokens_per_round}) exceeded: {metadata['token_usage']['total']}")
            raise TokenBudgetExceededError(
                f"Token budget exceeded: {metadata['token_usage']['total']} > {max_tokens_per_round}"
            )
        
        # Check if response has tool calls
        if response.tool_calls:
            tool_call_count += len(response.tool_calls)
            metadata["tool_calls"] = tool_call_count
            
            # Update Prometheus metrics for tool calls
            for tool_call in response.tool_calls:
                TOOL_CALLS_COUNTER.labels(tool_name=tool_call.name).inc()
            
            # Add response to conversation
            conversation.append(response)
            
            # Process tool calls and add results to conversation
            for tool_call in response.tool_calls:
                # Log tool call
                logger.info(f"Tool call: {tool_call.name}({tool_call.arguments})")
                
                # Execute tool and get result
                tool_result = agent.execute_tool(tool_call)
                
                # Add tool result to conversation
                conversation.append(Message(
                    role="tool",
                    content=tool_result,
                    name=tool_call.name
                ))
        else:
            # No tool calls, this is the final response
            final_response = response.content
            conversation.append(response)
            break
    
    # Check if we've exceeded the maximum number of loops
    if loop_count >= max_loops and final_response is None:
        logger.warning(f"Maximum loops ({max_loops}) exceeded")
        raise MaxLoopsExceededError(f"Maximum loops ({max_loops}) exceeded")
    
    # Calculate duration
    metadata["duration"] = time.time() - start_time
    
    # Update Prometheus metrics for duration
    CONVERSATION_DURATION_HISTOGRAM.observe(metadata["duration"])
    
    # Add Langtrace URL if in debug mode
    if debug_mode and hasattr(agent, "langtrace_url"):
        metadata["langtrace_url"] = agent.langtrace_url
        logger.info(f"Langtrace URL: {metadata['langtrace_url']}")
    
    return final_response, metadata