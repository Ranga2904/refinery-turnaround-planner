"""
Audit logging for turnaround planning workflow.

Logs every human decision (approval/rejection) with:
- Who made the decision
- When
- What they approved/rejected
- Full state snapshot at that moment
"""

import os
from datetime import datetime
from dotenv import load_dotenv
import psycopg
from typing import Optional, Dict, List
import json

load_dotenv()


def get_db_connection():
    """Get PostgreSQL connection"""
    return psycopg.connect(os.environ['SUPABASE_DB_URL'])


def create_audit_table():
    """
    Create audit log table if it doesn't exist.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS approval_log (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            thread_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            action TEXT NOT NULL,
            user_email TEXT NOT NULL,
            feedback TEXT,
            timestamp TIMESTAMP DEFAULT NOW(),
            state_snapshot JSONB,
            metadata JSONB
        );
        
        CREATE INDEX IF NOT EXISTS idx_approval_session ON approval_log(session_id);
        CREATE INDEX IF NOT EXISTS idx_approval_thread ON approval_log(thread_id);
        CREATE INDEX IF NOT EXISTS idx_approval_timestamp ON approval_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_approval_user ON approval_log(user_email);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("✓ Audit log table created")


def log_approval(
    session_id: str,
    thread_id: str,
    agent_name: str,
    action: str,
    user_email: str,
    feedback: Optional[str] = None,
    state_snapshot: Optional[Dict] = None,
    metadata: Optional[Dict] = None
):
    """
    Log an approval or rejection action.
    
    Args:
        session_id: User session identifier
        thread_id: LangGraph thread identifier
        agent_name: Which agent was approved/rejected (agent1, agent2, agent3)
        action: 'approved' or 'rejected'
        user_email: Who made the decision
        feedback: Optional user comment
        state_snapshot: Full state at time of decision
        metadata: Additional context (e.g., IP address, browser)
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO approval_log 
        (session_id, thread_id, agent_name, action, user_email, feedback, state_snapshot, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        session_id,
        thread_id,
        agent_name,
        action,
        user_email,
        feedback,
        json.dumps(state_snapshot) if state_snapshot else None,
        json.dumps(metadata) if metadata else None
    ))
    
    conn.commit()
    cursor.close()
    conn.close()


def get_audit_log(
    session_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    user_email: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """
    Retrieve audit log entries.
    
    Args:
        session_id: Filter by session
        thread_id: Filter by thread
        user_email: Filter by user
        limit: Max entries to return
    
    Returns:
        List of audit log entries
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    query = "SELECT * FROM approval_log WHERE 1=1"
    params = []
    
    if session_id:
        query += " AND session_id = %s"
        params.append(session_id)
    
    if thread_id:
        query += " AND thread_id = %s"
        params.append(thread_id)
    
    if user_email:
        query += " AND user_email = %s"
        params.append(user_email)
    
    query += " ORDER BY timestamp DESC LIMIT %s"
    params.append(limit)
    
    cursor.execute(query, params)
    
    columns = [desc[0] for desc in cursor.description]
    results = []
    
    for row in cursor.fetchall():
        entry = dict(zip(columns, row))
        # JSONB already returns dict, only parse if it's a string
        if entry.get('state_snapshot') and isinstance(entry['state_snapshot'], str):
            entry['state_snapshot'] = json.loads(entry['state_snapshot'])
        if entry.get('metadata') and isinstance(entry['metadata'], str):
            entry['metadata'] = json.loads(entry['metadata'])
        results.append(entry)
    
    cursor.close()
    conn.close()
    
    return results


def get_session_timeline(session_id: str) -> List[Dict]:
    """
    Get chronological timeline of decisions for a session.
    """
    entries = get_audit_log(session_id=session_id)
    
    timeline = []
    for entry in reversed(entries):  # Chronological order
        timeline.append({
            'timestamp': entry['timestamp'].isoformat(),
            'agent': entry['agent_name'],
            'action': entry['action'],
            'user': entry['user_email'],
            'feedback': entry['feedback']
        })
    
    return timeline


def generate_audit_report(session_id: str) -> str:
    """
    Generate human-readable audit report for a session.
    """
    entries = get_audit_log(session_id=session_id)
    
    if not entries:
        return f"No audit log found for session {session_id}"
    
    report = f"AUDIT REPORT: Session {session_id}\n"
    report += "=" * 60 + "\n\n"
    
    for entry in reversed(entries):
        report += f"[{entry['timestamp']}]\n"
        report += f"  Agent: {entry['agent_name']}\n"
        report += f"  Action: {entry['action'].upper()}\n"
        report += f"  User: {entry['user_email']}\n"
        
        if entry['feedback']:
            report += f"  Feedback: {entry['feedback']}\n"
        
        report += "\n"
    
    return report


if __name__ == "__main__":
    """Test audit logging"""
    print("=" * 60)
    print("AUDIT LOGGING SETUP")
    print("=" * 60)
    
    # Create table
    print("\n[1] Creating audit log table...")
    create_audit_table()
    
    # Test logging
    print("\n[2] Testing audit log entry...")
    log_approval(
        session_id="test-session-001",
        thread_id="test-thread-001",
        agent_name="agent1",
        action="approved",
        user_email="test@refinery.com",
        feedback="Risk analysis looks reasonable",
        state_snapshot={'current_step': 'agent1_approval', 'items_analyzed': 30},
        metadata={'ip_address': '192.168.1.1', 'browser': 'Chrome'}
    )
    print("✓ Logged approval")
    
    # Test retrieval
    print("\n[3] Retrieving audit log...")
    entries = get_audit_log(session_id="test-session-001")
    print(f"✓ Retrieved {len(entries)} entries")
    
    if entries:
        print("\nSample entry:")
        entry = entries[0]
        print(f"  Timestamp: {entry['timestamp']}")
        print(f"  Agent: {entry['agent_name']}")
        print(f"  Action: {entry['action']}")
        print(f"  User: {entry['user_email']}")
    
    # Generate report
    print("\n[4] Generating audit report...")
    report = generate_audit_report("test-session-001")
    print(report)
    
    print("=" * 60)
    print("✓ AUDIT LOGGING READY")
    print("=" * 60) 
