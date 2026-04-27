import os
import hashlib
import json
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

class HintManager:
    """Manages SQL hints storage and retrieval using Qdrant vector database."""
    
    def __init__(self):
        """Initialize HintManager (lazy initialization of heavy components)."""
        self.client = None
        self.model = None
        self.collection_name = "sql_hints"
        self._initialized = False
    
    def _initialize(self):
        """Lazy initialize Qdrant client and sentence transformer model."""
        if self._initialized:
            return True
            
        try:
            # Initialize Qdrant client with persistent storage
            # Store data in ./qdrant_data directory
            self.client = QdrantClient(path="./qdrant_data")
            
            # Import sentence-transformers only when needed (lazy import)
            from sentence_transformers import SentenceTransformer
            
            # Initialize sentence transformer for embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create collection if it doesn't exist
            self._ensure_collection()
            
            self._initialized = True
            logging.info("HintManager initialized successfully with persistent storage")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize HintManager: {e}")
            self.client = None
            self.model = None
            return False
    
    def _ensure_collection(self):
        """Ensure the Qdrant collection exists."""
        try:
            if not self.client.collection_exists(self.collection_name):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 384 is the dimension for all-MiniLM-L6-v2
                )
                logging.info(f"Created collection: {self.collection_name}")
        except Exception as e:
            logging.error(f"Failed to create collection: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for the given text."""
        try:
            if not self._initialize():
                return []
            if self.model is None:
                return []
            
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logging.error(f"Failed to generate embedding: {e}")
            return []
    
    def _generate_hint_id(self, connection_id: str, content: str) -> str:
        """Generate unique hint ID based on connection and content."""
        combined = f"{connection_id}:{content}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def add_hint(self, connection_id: str, content: str, sql_query: str, schema_context: str = "") -> bool:
        """Add a hint to the vector database."""
        try:
            if not self._initialize():
                return False
            if self.client is None or self.model is None:
                logging.error("HintManager not properly initialized")
                return False
            
            # Generate embedding for the content
            embedding = self._generate_embedding(content)
            if not embedding:
                return False
            
            # Generate unique point ID
            point_id = self._generate_hint_id(connection_id, content)
            
            # Create point with metadata
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "connection_id": connection_id,
                    "content": content,
                    "sql_query": sql_query,
                    "schema_context": schema_context,
                    "usage_count": 0,
                    "created_at": str(os.times())
                }
            )
            
            # Insert into Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logging.info(f"Added hint for connection {connection_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to add hint: {e}")
            return False
    
    def get_hints(self, connection_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant hints for a given query and connection.
        
        Args:
            connection_id: Unique identifier for the database connection
            query: User question or query description
            limit: Maximum number of hints to return
        
        Returns:
            List of relevant hints with metadata
        """
        try:
            if not self._initialize():
                return []
            if self.client is None or self.model is None:
                logging.error("HintManager not properly initialized")
                return []
            
            # Generate embedding for the query
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            # Create filter for connection_id
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="connection_id",
                        match=MatchValue(value=connection_id)
                    )
                ]
            )
            
            # Search for similar hints
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=limit,
                with_payload=True
            )
            
            # Format results
            hints = []
            for hit in search_result.points:
                payload = hit.payload
                hints.append({
                    "content": payload.get("content", ""),
                    "sql_query": payload.get("sql_query", ""),
                    "schema_context": payload.get("schema_context", ""),
                    "similarity_score": hit.score,
                    "usage_count": payload.get("usage_count", 0)
                })
                
                # Increment usage count (optional - could be done asynchronously)
                self._increment_usage(hit.id)
            
            logging.info(f"Retrieved {len(hints)} hints for connection {connection_id}")
            return hints
            
        except Exception as e:
            logging.error(f"Failed to get hints: {e}")
            return []
    
    def _increment_usage(self, point_id: str):
        """Increment usage count for a hint."""
        try:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"usage_count": 1},
                points=[point_id]
            )
        except Exception as e:
            logging.error(f"Failed to increment usage count: {e}")
    
    def delete_hints_by_connection(self, connection_id: str) -> bool:
        """Delete all hints for a specific connection."""
        try:
            if not self._initialize():
                return False
            if self.client is None:
                return False
            
            # Create filter for connection_id
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="connection_id",
                        match=MatchValue(value=connection_id)
                    )
                ]
            )
            
            # Delete points
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=query_filter
            )
            
            logging.info(f"Deleted hints for connection {connection_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete hints: {e}")
            return False
    
    def get_connection_stats(self, connection_id: str) -> Dict[str, Any]:
        """Get statistics for hints in a connection."""
        try:
            if not self._initialize():
                return {}
            if self.client is None:
                return {}
            
            # Create filter for connection_id
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="connection_id",
                        match=MatchValue(value=connection_id)
                    )
                ]
            )
            
            # Count points
            count_result = self.client.count(
                collection_name=self.collection_name,
                count_filter=query_filter
            )
            
            return {
                "total_hints": count_result.count,
                "connection_id": connection_id
            }
            
        except Exception as e:
            logging.error(f"Failed to get connection stats: {e}")
            return {}

# Global instance for easy access
hint_manager = HintManager()
