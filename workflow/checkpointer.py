"""
PostgreSQL checkpointer setup for LangGraph state persistence.
"""

import os
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

load_dotenv()


def create_checkpointer():
    """
    Create PostgreSQL checkpointer for LangGraph.
    
    Uses Supabase connection string from .env
    """
    db_url = os.environ.get('SUPABASE_DB_URL')
    
    if not db_url:
        raise ValueError("SUPABASE_DB_URL not found in environment")
    
    # Create connection pool
    pool = ConnectionPool(
        conninfo=db_url,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
        }
    )
    
    # Create checkpointer
    checkpointer = PostgresSaver(pool)
    
    # Setup tables (first time only)
    checkpointer.setup()
    
    return checkpointer


if __name__ == "__main__":
    """Test checkpointer connection"""
    print("Testing PostgreSQL checkpointer...")
    
    try:
        checkpointer = create_checkpointer()
        print("✓ Checkpointer created successfully")
        print("✓ Database tables initialized")
    except Exception as e:
        print(f"✗ Error: {e}")