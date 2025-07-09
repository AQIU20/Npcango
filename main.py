"""
Main module for NPC system.
Implements FastAPI application with WebSocket support for NPC interactions.
"""
import os
import json
import logging
import uuid
from typing import Dict, Any, Optional, List, Union

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import make_asgi_app

from .agent_core import build_npc
from .orchestrator import run_loop, TokenBudgetExceededError, MaxLoopsExceededError

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="NPC System",
    description="AI-powered NPC system for game interactions",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Configuration
DEFAULT_CONFIG = {
    "npc_name": "Blacksmith",
    "npc_description": "A skilled blacksmith who can craft weapons and armor.",
    "memory_path": "memory",
    "lore_db_path": "lancedb",
    "api_base_url": "http://localhost:8000",
    "model": "gpt-4o",
    "use_local_llama": False,
    "local_llama_url": None,
    "max_tokens_per_round": 6000,
    "max_tool_calls": 4,
    "max_loops": 10,
    "debug_mode": False
}

# Load config from environment variables
def load_config() -> Dict[str, Any]:
    """Load configuration from environment variables."""
    config = DEFAULT_CONFIG.copy()
    
    # Override with environment variables
    if os.environ.get("NPC_NAME"):
        config["npc_name"] = os.environ.get("NPC_NAME")
    
    if os.environ.get("NPC_DESCRIPTION"):
        config["npc_description"] = os.environ.get("NPC_DESCRIPTION")
    
    if os.environ.get("MEMORY_PATH"):
        config["memory_path"] = os.environ.get("MEMORY_PATH")
    
    if os.environ.get("LORE_DB_PATH"):
        config["lore_db_path"] = os.environ.get("LORE_DB_PATH")
    
    if os.environ.get("API_BASE_URL"):
        config["api_base_url"] = os.environ.get("API_BASE_URL")
    
    if os.environ.get("MODEL"):
        config["model"] = os.environ.get("MODEL")
    
    if os.environ.get("USE_LOCAL_LLAMA"):
        config["use_local_llama"] = os.environ.get("USE_LOCAL_LLAMA").lower() == "true"
    
    if os.environ.get("LOCAL_LLAMA_URL"):
        config["local_llama_url"] = os.environ.get("LOCAL_LLAMA_URL")
    
    if os.environ.get("MAX_TOKENS_PER_ROUND"):
        config["max_tokens_per_round"] = int(os.environ.get("MAX_TOKENS_PER_ROUND"))
    
    if os.environ.get("MAX_TOOL_CALLS"):
        config["max_tool_calls"] = int(os.environ.get("MAX_TOOL_CALLS"))
    
    if os.environ.get("MAX_LOOPS"):
        config["max_loops"] = int(os.environ.get("MAX_LOOPS"))
    
    if os.environ.get("DEBUG_MODE"):
        config["debug_mode"] = os.environ.get("DEBUG_MODE").lower() == "true"
    
    return config

# Initialize NPC agent
config = load_config()
npc_agent = build_npc(
    npc_name=config["npc_name"],
    npc_description=config["npc_description"],
    memory_path=config["memory_path"],
    lore_db_path=config["lore_db_path"],
    api_base_url=config["api_base_url"],
    model=config["model"],
    use_local_llama=config["use_local_llama"],
    local_llama_url=config["local_llama_url"],
    debug_mode=config["debug_mode"]
)

# WebSocket connection manager
class ConnectionManager:
    """Manager for WebSocket connections."""
    
    def __init__(self):
        """Initialize the connection manager."""
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Connect a WebSocket client.
        
        Args:
            websocket: The WebSocket connection
            client_id: ID of the client
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str) -> None:
        """
        Disconnect a WebSocket client.
        
        Args:
            client_id: ID of the client
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: str) -> None:
        """
        Send a message to a client.
        
        Args:
            client_id: ID of the client
            message: Message to send
        """
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)
    
    async def send_json(self, client_id: str, data: Dict[str, Any]) -> None:
        """
        Send JSON data to a client.
        
        Args:
            client_id: ID of the client
            data: JSON data to send
        """
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)

# Create connection manager
manager = ConnectionManager()

# WebSocket endpoint for NPC interactions
@app.websocket("/npc")
async def npc_websocket(
    websocket: WebSocket,
    player_id: str = Query(..., description="ID of the player"),
    conversation_id: Optional[str] = Query(None, description="ID for the conversation")
):
    """
    WebSocket endpoint for NPC interactions.
    
    Args:
        websocket: The WebSocket connection
        player_id: ID of the player
        conversation_id: Optional ID for the conversation
    """
    # Generate conversation ID if not provided
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())
    
    # Generate client ID
    client_id = f"{player_id}:{conversation_id}"
    
    # Connect WebSocket
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            message = await websocket.receive_text()
            
            try:
                # Process message with NPC agent
                response, metadata = run_loop(
                    agent=npc_agent,
                    user_input=message,
                    player_id=player_id,
                    conversation_id=conversation_id,
                    max_tokens_per_round=config["max_tokens_per_round"],
                    max_tool_calls=config["max_tool_calls"],
                    max_loops=config["max_loops"],
                    debug_mode=config["debug_mode"]
                )
                
                # Send response to client
                await manager.send_json(client_id, {
                    "response": response,
                    "metadata": metadata
                })
                
            except TokenBudgetExceededError as e:
                # Handle token budget exceeded
                logger.warning(f"Token budget exceeded: {str(e)}")
                await manager.send_json(client_id, {
                    "error": "token_budget_exceeded",
                    "message": "The conversation is too long. Please try a shorter message."
                })
                
            except MaxLoopsExceededError as e:
                # Handle max loops exceeded
                logger.warning(f"Max loops exceeded: {str(e)}")
                await manager.send_json(client_id, {
                    "error": "max_loops_exceeded",
                    "message": "The conversation is too complex. Please try a simpler request."
                })
                
            except Exception as e:
                # Handle other errors
                logger.error(f"Error processing message: {str(e)}", exc_info=True)
                await manager.send_json(client_id, {
                    "error": "internal_error",
                    "message": "An error occurred while processing your message."
                })
    
    except WebSocketDisconnect:
        # Handle disconnect
        manager.disconnect(client_id)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

# Run the server
def run_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Run the FastAPI server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()