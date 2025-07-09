"""
Tools module for NPC system.
Implements game-specific tools like inventory management with retry logic and metrics.
"""
import time
import random
import logging
from typing import Dict, Any, Optional, Callable, List, Union
import httpx
from prometheus_client import Counter

# Set up logging
logger = logging.getLogger(__name__)

# Prometheus metrics
TOOL_ERROR_COUNTER = Counter(
    "tool_error_total", 
    "Total number of tool execution errors",
    ["tool_name", "error_type"]
)

class ToolExecutionError(Exception):
    """Exception raised when a tool execution fails."""
    pass

def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0,
    jitter: bool = True
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        jitter: Whether to add random jitter to delay
        
    Returns:
        Wrapped function with retry logic
    """
    def wrapper(*args, **kwargs):
        retries = 0
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    # Log and re-raise the exception
                    logger.error(f"Failed after {max_retries} retries: {str(e)}")
                    raise
                
                # Calculate delay with exponential backoff
                delay = min(base_delay * (2 ** (retries - 1)), max_delay)
                
                # Add jitter if enabled
                if jitter:
                    delay = delay * (0.5 + random.random())
                
                logger.warning(f"Retry {retries}/{max_retries} after {delay:.2f}s: {str(e)}")
                time.sleep(delay)
    
    return wrapper

class GameTools:
    """
    Game-specific tools for NPC interactions.
    Implements inventory management and other game mechanics.
    """
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        timeout: float = 10.0,
        max_retries: int = 3
    ):
        """
        Initialize game tools.
        
        Args:
            api_base_url: Base URL for the game API
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_base_url = api_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.http_client = httpx.Client(timeout=timeout, base_url=api_base_url)
        
        # Track processed transactions to ensure idempotency
        self.processed_transactions = set()
    
    @retry_with_exponential_backoff(max_retries=3)
    def _call_inventory_api(
        self,
        player_id: str,
        items: List[Dict[str, Any]],
        tx_id: str
    ) -> Dict[str, Any]:
        """
        Call the inventory API with retry logic.
        
        Args:
            player_id: ID of the player
            items: List of items to add to inventory
            tx_id: Transaction ID for idempotency
            
        Returns:
            API response data
            
        Raises:
            ToolExecutionError: If the API call fails after retries
        """
        try:
            response = self.http_client.post(
                "/inventory",
                json={
                    "player_id": player_id,
                    "items": items,
                    "tx_id": tx_id
                }
            )
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            TOOL_ERROR_COUNTER.labels(
                tool_name="update_inventory",
                error_type="http_error"
            ).inc()
            raise ToolExecutionError(f"HTTP error: {e.response.status_code} - {e.response.text}")
            
        except httpx.RequestError as e:
            TOOL_ERROR_COUNTER.labels(
                tool_name="update_inventory",
                error_type="request_error"
            ).inc()
            raise ToolExecutionError(f"Request error: {str(e)}")
    
    def update_inventory(
        self,
        player_id: str,
        items: List[Dict[str, Any]],
        tx_id: Optional[str] = None
    ) -> str:
        """
        Stub implementation of update_inventory.
        Prints a message instead of making HTTP requests.
        
        Args:
            player_id: ID of the player
            items: List of items to add to inventory
                Each item should have:
                - item_id: ID of the item
                - quantity: Number of items to add
                - properties: Optional item properties
            tx_id: Transaction ID for idempotency (generated if not provided)
            
        Returns:
            String with result of the operation
        """
        # Generate transaction ID if not provided (for consistency)
        if tx_id is None:
            tx_id = f"tx_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Format items for response
        items_text = ", ".join([
            f"{item.get('quantity', 1)}x {item.get('name', item['item_id'])}"
            for item in items
        ])
        
        # Print stub message
        print(f"[Stub] pretend to add {items_text} to {player_id}")
        
        return "已假装写入"
    
    def create_update_inventory_tool(self) -> Callable:
        """
        Create an update_inventory tool function for use with Agno.
        
        Returns:
            Callable: A tool function that can be registered with an Agno agent
        """
        def update_inventory_tool(
            player_id: str,
            item_id: str,
            quantity: int = 1,
            properties: Optional[Dict[str, Any]] = None,
            tx_id: Optional[str] = None
        ) -> str:
            """
            Add items to a player's inventory.
            
            Args:
                player_id: ID of the player
                item_id: ID of the item to add
                quantity: Number of items to add (default: 1)
                properties: Optional item properties
                tx_id: Transaction ID for idempotency
                
            Returns:
                String with result of the operation
            """
            item = {
                "item_id": item_id,
                "quantity": quantity
            }
            
            if properties:
                item["properties"] = properties
            
            return self.update_inventory(player_id, [item], tx_id)
        
        return update_inventory_tool