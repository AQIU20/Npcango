"""
Knowledge base module for NPC system using LanceDB for hybrid vector search.
Implements dense vector + BM25 hybrid search for weapon recipes and lore.
"""
import os
import json
from typing import List, Dict, Any, Optional, Union, Callable

import lancedb
import numpy as np
from lancedb.embeddings import OpenAIEmbeddings
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, Field

from openai import OpenAI

class LoreItem(LanceModel):
    """Pydantic model for lore items in LanceDB."""
    id: str
    content: str
    title: str
    tags: List[str]
    embedding: Vector[1536] = Field(embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"))

class LoreKnowledgeBase:
    """
    Knowledge base for NPC lore using LanceDB with hybrid search.
    Implements dense vector + BM25 hybrid search with configurable weights.
    """
    
    def __init__(
        self,
        db_path: str = "lancedb",
        table_name: str = "lore",
        dense_weight: float = 0.8,
        bm25_weight: float = 0.2,
        client: Optional[OpenAI] = None
    ):
        """
        Initialize the LanceDB knowledge base.
        
        Args:
            db_path: Path to the LanceDB database
            table_name: Name of the table to use
            dense_weight: Weight for dense vector search (0.0-1.0)
            bm25_weight: Weight for BM25 search (0.0-1.0)
            client: OpenAI client for embeddings
        """
        self.db_path = db_path
        self.table_name = table_name
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.client = client or OpenAI()
        
        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)
        
        # Connect to LanceDB
        self.db = lancedb.connect(db_path)
        
        # Create or get table
        if table_name in self.db.table_names():
            self.table = self.db.open_table(table_name)
        else:
            # Create empty table with schema
            self.table = self.db.create_table(
                table_name,
                schema=LoreItem,
                mode="overwrite"
            )
    
    def add_lore(self, items: List[Dict[str, Any]]) -> None:
        """
        Add lore items to the knowledge base.
        
        Args:
            items: List of lore items to add
                Each item should have:
                - id: Unique identifier
                - content: Main text content
                - title: Title of the lore item
                - tags: List of tags for filtering
        """
        # Convert to LoreItem objects
        lore_items = [LoreItem(**item) for item in items]
        
        # Add to table
        self.table.add(lore_items)
    
    def search_lore(
        self,
        query: str,
        limit: int = 5,
        tag_filter: Optional[str] = None,
        metric: str = "cosine"
    ) -> List[Dict[str, Any]]:
        """
        Search for lore items using hybrid search (vector + BM25).
        
        Args:
            query: The search query text
            limit: Maximum number of results to return
            tag_filter: Optional tag to filter results
            metric: Distance metric to use ('cosine', 'l2', etc.)
            
        Returns:
            List of lore items with hybrid similarity scores
        """
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Start with vector search
        vector_search = self.table.search(
            query_embedding,
            vector_column_name="embedding"
        ).limit(limit * 2)  # Get more for filtering
        
        # Add BM25 text search
        if self.bm25_weight > 0:
            vector_search = vector_search.hybrid(
                query,
                text_column_name="content",
                alpha=self.bm25_weight / self.dense_weight if self.dense_weight > 0 else 1.0
            )
        
        # Add tag filter if provided
        if tag_filter:
            vector_search = vector_search.where(f"tags CONTAINS '{tag_filter}'")
        
        # Execute search
        results = vector_search.to_list()
        
        # Process and format results
        processed_results = []
        for item in results:
            # Extract relevant fields
            processed_item = {
                "id": item["id"],
                "title": item["title"],
                "content": item["content"],
                "tags": item["tags"],
                "score": item["_score"] if "_score" in item else 0.0
            }
            processed_results.append(processed_item)
        
        return processed_results[:limit]
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    
    def create_search_lore_tool(self) -> Callable:
        """
        Create a search_lore tool function for use with Agno.
        
        Returns:
            Callable: A tool function that can be registered with an Agno agent
        """
        def search_lore_tool(query: str, tag_filter: str = "weapon_recipe") -> str:
            """
            Search for lore in the knowledge base.
            
            Args:
                query: The search query
                tag_filter: Tag to filter results by (default: weapon_recipe)
                
            Returns:
                String with formatted search results
            """
            results = self.search_lore(query, tag_filter=tag_filter)
            
            if not results:
                return "No relevant information found in the lore database."
            
            # Format results
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"[{i}] {result['title']}\n"
                    f"Relevance: {result['score']:.2f}\n"
                    f"{result['content']}\n"
                )
            
            return "\n".join(formatted_results)
        
        return search_lore_tool