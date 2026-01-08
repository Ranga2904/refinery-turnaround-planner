 
"""
LangGraph workflow with human approval gates.

Flow:
  Input → Agent 1 → [Human Approval] → Agent 2 → [Human Approval] → Agent 3 → Output
"""

from typing import TypedDict, Literal, Optional, Annotated
from datetime import datetime
import pandas as pd
from langgraph.graph import StateGraph, END
# from langgraph.checkpoint.postgres import PostgresSaver
import operator
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # Add project root to path
from workflow.checkpointer import create_checkpointer
from utils.audit import log_approval, create_audit_table


def sanitize_for_serialization(obj):
    """
    Convert NumPy types to Python native types for serialization.
    """
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: sanitize_for_serialization(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_serialization(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def log_agent_approval(state: TurnaroundState, agent_name: str, action: str):
    """
    Log approval/rejection to audit table.
    """
    try:
        from utils.audit import log_approval
        
        log_approval(
            session_id=state['session_id'],
            thread_id=state['session_id'],  # Using session_id as thread_id
            agent_name=agent_name,
            action=action,
            user_email=state.get(f'{agent_name}_approved_by', state['user_email']),
            feedback=state.get(f'{agent_name}_rejection_reason'),
            state_snapshot={
                'current_step': state['current_step'],
                f'{agent_name}_complete': state.get(f'{agent_name}_complete'),
                'constraints': state['constraints']
            }
        )
    except Exception as e:
        print(f"Warning: Audit logging failed: {e}")

class TurnaroundState(TypedDict):
    """
    Complete state for turnaround planning workflow.
    All agent outputs and approval tracking stored here.
    """
    
    # Input data
    equipment_data: dict  # Equipment registry as dict
    failure_history: dict  # Failure events as dict
    scope_proposals: dict  # Proposed scope as dict
    constraints: dict  # Budget, duration, margin constraints
    
    # User context
    user_email: str
    session_id: str
    
    # Agent 1 outputs
    risk_analysis: Optional[dict]  # Risk-ranked items
    agent1_complete: bool
    agent1_approved: bool
    agent1_approved_by: Optional[str]
    agent1_approved_at: Optional[str]
    agent1_rejection_reason: Optional[str]
    
    # Agent 2 outputs
    optimized_scope: Optional[dict]  # Included items
    deferred_items: Optional[dict]  # Deferred items
    optimization_summary: Optional[dict]  # Summary statistics
    agent2_complete: bool
    agent2_approved: bool
    agent2_approved_by: Optional[str]
    agent2_approved_at: Optional[str]
    agent2_rejection_reason: Optional[str]
    
    # Agent 3 outputs
    executive_memo: Optional[str]  # Final memo text
    agent3_complete: bool
    
    # Workflow control
    current_step: Literal[
        "start",
        "agent1_running",
        "agent1_approval",
        "agent2_running", 
        "agent2_approval",
        "agent3_running",
        "complete"
    ]
    
    # Error tracking
    errors: Annotated[list[str], operator.add]  # Accumulate errors


def create_initial_state(
    equipment_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    constraints: dict,
    user_email: str,
    session_id: str
) -> TurnaroundState:
    """
    Convert input DataFrames to initial state dict.
    """
    return {
        # Input data (convert to dict for JSON serialization)
        'equipment_data': equipment_df.to_dict('records'),
        'failure_history': failures_df.to_dict('records'),
        'scope_proposals': scope_df.to_dict('records'),
        'constraints': constraints,
        
        # User context
        'user_email': user_email,
        'session_id': session_id,
        
        # Agent 1
        'risk_analysis': None,
        'agent1_complete': False,
        'agent1_approved': False,
        'agent1_approved_by': None,
        'agent1_approved_at': None,
        'agent1_rejection_reason': None,
        
        # Agent 2
        'optimized_scope': None,
        'deferred_items': None,
        'optimization_summary': None,
        'agent2_complete': False,
        'agent2_approved': False,
        'agent2_approved_by': None,
        'agent2_approved_at': None,
        'agent2_rejection_reason': None,
        
        # Agent 3
        'executive_memo': None,
        'agent3_complete': False,
        
        # Workflow
        'current_step': 'start',
        'errors': []
    }


# Node functions (agents will be called here)

def run_agent1_node(state: TurnaroundState) -> TurnaroundState:
    """
    Execute Agent 1: Risk Analysis
    """
    print("[Agent 1] Running risk analysis...")
    
    try:
        from agents.risk_analyst import run_agent1
        
        # Convert dict back to DataFrame
        equipment_df = pd.DataFrame(state['equipment_data'])
        failures_df = pd.DataFrame(state['failure_history'])
        scope_df = pd.DataFrame(state['scope_proposals'])
        
        # Run Agent 1
        risk_results = run_agent1(
            equipment_df,
            failures_df,
            scope_df,
            production_capacity_bbl_day=state['constraints']['production_capacity_bbl_day'],
            margin_per_bbl=state['constraints']['margin_per_bbl']
        )
        
        # Update state
        # Update state
        state['risk_analysis'] = sanitize_for_serialization(risk_results.to_dict('records'))
        state['agent1_complete'] = True
        state['current_step'] = 'agent1_approval'
        
        print(f"[Agent 1] ✓ Analyzed {len(risk_results)} items")
        
    except Exception as e:
        print(f"[Agent 1] ✗ Error: {e}")
        state['errors'].append(f"Agent 1 failed: {str(e)}")
        state['current_step'] = 'complete'
    
    if state.get('agent1_approved'):
        log_agent_approval(state, 'agent1', 'approved')
    elif state.get('agent1_rejection_reason'):
        log_agent_approval(state, 'agent1', 'rejected')

    return state


def wait_for_agent1_approval(state: TurnaroundState) -> TurnaroundState:
    """
    Pause point: waiting for human to approve Agent 1 results.
    This node does nothing - execution pauses here until human updates state.
    """
    print("[Approval Gate 1] Waiting for human review of risk analysis...")
    return state


def run_agent2_node(state: TurnaroundState) -> TurnaroundState:
    """
    Execute Agent 2: Scope Optimization
    """
    print("[Agent 2] Running scope optimization...")
    
    try:
        from agents.optimizer import run_agent2
        
        # Convert dict back to DataFrame
        risk_df = pd.DataFrame(state['risk_analysis'])
        
        # Run Agent 2
        results = run_agent2(
            risk_df,
            state['constraints']['budget_cap'],
            state['constraints']['duration_cap']
        )
        
        # Update state
        # Update state
        state['optimized_scope'] = sanitize_for_serialization(results['included_scope'].to_dict('records'))
        state['deferred_items'] = sanitize_for_serialization(results['deferred_items'].to_dict('records'))
        state['optimization_summary'] = sanitize_for_serialization(results['summary'])
        state['agent2_complete'] = True
        state['current_step'] = 'agent2_approval'
        
        print(f"[Agent 2] ✓ Included {results['summary']['total_included']} items")
        
    except Exception as e:
        print(f"[Agent 2] ✗ Error: {e}")
        state['errors'].append(f"Agent 2 failed: {str(e)}")
        state['current_step'] = 'complete'
    
    if state.get('agent2_approved'):
        log_agent_approval(state, 'agent2', 'approved')
    elif state.get('agent2_rejection_reason'):
        log_agent_approval(state, 'agent2', 'rejected')

    return state


def wait_for_agent2_approval(state: TurnaroundState) -> TurnaroundState:
    """
    Pause point: waiting for human to approve Agent 2 results.
    """
    print("[Approval Gate 2] Waiting for human review of optimized scope...")
    return state


def run_agent3_node(state: TurnaroundState) -> TurnaroundState:
    """
    Execute Agent 3: Executive Translator
    """
    print("[Agent 3] Generating executive memo...")
    
    try:
        from agents.translator import run_agent3
        
        # Convert dicts back to DataFrames
        risk_df = pd.DataFrame(state['risk_analysis'])
        optimized_df = pd.DataFrame(state['optimized_scope'])
        
        # Handle empty deferred items
        if state['deferred_items']:
            deferred_df = pd.DataFrame(state['deferred_items'])
        else:
            deferred_df = pd.DataFrame(columns=[
                'equipment_id', 'work_type', 'expected_loss_avoided', 'deferral_reason'
            ])
        
        # Run Agent 3
        memo = run_agent3(
            risk_df,
            optimized_df,
            deferred_df,
            state['optimization_summary'],
            state['constraints']
        )
        
        state['executive_memo'] = memo
        state['agent3_complete'] = True
        state['current_step'] = 'complete'
        
        print("[Agent 3] ✓ Memo generated")
        
    except Exception as e:
        print(f"[Agent 3] ✗ Error: {e}")
        state['errors'].append(f"Agent 3 failed: {str(e)}")
        state['current_step'] = 'complete'
    
    return state


# Conditional routing functions

def route_after_agent1_approval(state: TurnaroundState) -> Literal["agent2", "end"]:
    """
    Route based on Agent 1 approval status.
    For testing without checkpointer, we'll manually handle approval in invoke calls.
    """
    if state['agent1_approved']:
        return "agent2"
    else:
        # Without checkpointer, we can't pause - just end here
        return "end"


def route_after_agent2_approval(state: TurnaroundState) -> Literal["agent3", "end"]:
    """
    Route based on Agent 2 approval status.
    """
    if state['agent2_approved']:
        return "agent3"
    else:
        return "end"


# Build the graph

def create_workflow(checkpointer=None):
    """
    Create LangGraph workflow with approval gates.
    """
    workflow = StateGraph(TurnaroundState)
    
    # Add nodes
    workflow.add_node("agent1", run_agent1_node)
    workflow.add_node("approval1", wait_for_agent1_approval)
    workflow.add_node("agent2", run_agent2_node)
    workflow.add_node("approval2", wait_for_agent2_approval)
    workflow.add_node("agent3", run_agent3_node)
    
    # Define edges
    workflow.set_entry_point("agent1")
    
    # Agent 1 → Approval Gate 1
    workflow.add_edge("agent1", "approval1")
    
    # Approval Gate 1 → Agent 2 (if approved) or END (if rejected)
    workflow.add_conditional_edges(
        "approval1",
        route_after_agent1_approval,
        {
            "agent2": "agent2",
            "end": END
        }
    )
    
    # Agent 2 → Approval Gate 2
    workflow.add_edge("agent2", "approval2")
    
    # Approval Gate 2 → Agent 3 (if approved) or END (if rejected)
    workflow.add_conditional_edges(
        "approval2",
        route_after_agent2_approval,
        {
            "agent3": "agent3",
            "end": END
        }
    )
    
    # Agent 3 → END
    workflow.add_edge("agent3", END)
    
    # Compile
    app = workflow.compile(checkpointer=checkpointer)
    
    return app


if __name__ == "__main__":
    """Test workflow without checkpointing"""
    import sqlite3
    
    print("=" * 60)
    print("TESTING LANGGRAPH WORKFLOW (NO CHECKPOINTING)")
    print("=" * 60)
    
    # Load data
    conn = sqlite3.connect('data/demo.db')
    equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
    failures_df = pd.read_sql('SELECT * FROM failures', conn)
    scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
    conn.close()
    
    # Create initial state
    initial_state = create_initial_state(
        equipment_df,
        failures_df,
        scope_df,
        constraints={
            'budget_cap': 18_500_000,
            'duration_cap': 42,
            'production_capacity_bbl_day': 100000,
            'margin_per_bbl': 25.0
        },
        user_email="test@refinery.com",
        session_id="test-session-001"
    )
    
    # Create workflow (no checkpointer for testing)
    # Create workflow WITH checkpointer
    print("\n[Setup] Creating checkpointer...")
    checkpointer = create_checkpointer()
    app = create_workflow(checkpointer=checkpointer)
    print("✓ Checkpointer configured")
    
    # Run Agent 1
    print("\n[Step 1] Running Agent 1...")
    config = {"configurable": {"thread_id": "test-thread"}}
    state = app.invoke(initial_state, config)
    
    print(f"\nCurrent step: {state['current_step']}")
    print(f"Agent 1 complete: {state['agent1_complete']}")
    
    # Simulate approval
    print("\n[Step 2] Simulating human approval...")
    state['agent1_approved'] = True
    state['agent1_approved_by'] = "test@refinery.com"
    state['agent1_approved_at'] = datetime.now().isoformat()

    
    # Continue to Agent 2
    print("\n[Step 3] Continuing to Agent 2...")
    state = app.invoke(state, config)
    
    print(f"\nCurrent step: {state['current_step']}")
    print(f"Agent 2 complete: {state['agent2_complete']}")
    
    # Simulate approval
    print("\n[Step 4] Simulating human approval...")
    state['agent2_approved'] = True
    state['agent2_approved_by'] = "test@refinery.com"
    state['agent2_approved_at'] = datetime.now().isoformat()
    
    # Continue to Agent 3
    print("\n[Step 5] Continuing to Agent 3...")
    state = app.invoke(state, config)
    
    print(f"\nCurrent step: {state['current_step']}")
    print(f"Agent 3 complete: {state['agent3_complete']}")
    
    print("\n" + "=" * 60)
    print("✓ WORKFLOW TEST COMPLETE")
    print("=" * 60)
    
    if state['executive_memo']:
        print("\nExecutive Memo:")
        print(state['executive_memo'])