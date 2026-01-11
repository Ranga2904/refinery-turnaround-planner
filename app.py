"""
Gradio UI for Refinery Turnaround Planning System

Multi-tab interface with human approval gates:
1. Upload data & set constraints
2. Review Agent 1 (Risk Analysis)
3. Review Agent 2 (Scope Optimization)
4. View final outputs & download
"""

import gradio as gr
import pandas as pd
import sqlite3
from datetime import datetime
import os
from typing import Optional, Tuple

# Import our agents and utilities
from workflow.graph import create_workflow, create_initial_state
from workflow.checkpointer import create_checkpointer
from utils.outputs import create_excel_workbook, save_excel_workbook, create_pdf_memo
from utils.audit import get_audit_log, generate_audit_report


# Global state management
current_session_id = None
current_thread_id = None
workflow_app = None


def initialize_workflow():
    """Initialize LangGraph workflow with checkpointer"""
    global workflow_app
    if workflow_app is None:
        checkpointer = create_checkpointer()
        workflow_app = create_workflow(checkpointer=checkpointer)
    return workflow_app


def start_new_session(
    equipment_file,
    scope_file,
    budget_cap: float,
    duration_cap: float,
    capacity: float,
    margin: float,
    user_email: str
) -> Tuple[str, pd.DataFrame, str]:
    """
    Start new turnaround planning session.
    
    Returns:
        (session_id, risk_analysis_df, status_message)
    """
    global current_session_id, current_thread_id
    
    try:
        # Generate session ID
        current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        current_thread_id = current_session_id
        
        status_msg = "⏳ Loading data..."
        
        # Load data
        if equipment_file is None or scope_file is None:
            # Use demo data
            conn = sqlite3.connect('data/demo.db')
            equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
            failures_df = pd.read_sql('SELECT * FROM failures', conn)
            scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
            conn.close()
            status_msg = "✓ Demo data loaded\n⏳ Initializing Agent 1..."
        else:
            # Load uploaded files
            equipment_df = pd.read_csv(equipment_file.name)
            scope_df = pd.read_csv(scope_file.name)
            
            # Validate required columns
            required_equip_cols = ['equipment_id', 'criticality', 'condition_score']
            required_scope_cols = ['scope_item_id', 'equipment_id', 'cost_estimate_usd', 'duration_days']
            
            missing_equip = [col for col in required_equip_cols if col not in equipment_df.columns]
            missing_scope = [col for col in required_scope_cols if col not in scope_df.columns]
            
            if missing_equip or missing_scope:
                error_details = ""
                if missing_equip:
                    error_details += f"\nEquipment file missing: {missing_equip}"
                if missing_scope:
                    error_details += f"\nScope file missing: {missing_scope}"
                raise ValueError(f"Invalid file format{error_details}")
            
            # For demo, use failure history from database
            conn = sqlite3.connect('data/demo.db')
            failures_df = pd.read_sql('SELECT * FROM failures', conn)
            conn.close()
            
            status_msg = "✓ Custom data loaded\n⏳ Initializing Agent 1..."
        
        # Create initial state
        constraints = {
            'budget_cap': budget_cap,
            'duration_cap': duration_cap,
            'production_capacity_bbl_day': capacity,
            'margin_per_bbl': margin
        }
        
        initial_state = create_initial_state(
            equipment_df,
            failures_df,
            scope_df,
            constraints,
            user_email,
            current_session_id
        )
        
        # Initialize workflow
        app = initialize_workflow()
        
        # Run Agent 1
        config = {"configurable": {"thread_id": current_thread_id}}
        state = app.invoke(initial_state, config)
        
        # Extract risk analysis
        risk_df = pd.DataFrame(state['risk_analysis'])
        
        # Format for display
        display_df = risk_df[[
            'rank', 'equipment_id', 'work_type', 'expected_loss_avoided',
            'risk_tier', 'confidence'
        ]].head(20)  # Show top 20
        
        display_df.columns = [
            'Rank', 'Equipment', 'Work Type', 'Risk Avoided ($)',
            'Risk Tier', 'Confidence'
        ]
        
        status_msg += f"\n✓ Agent 1 complete: Analyzed {len(risk_df)} items"
        status_msg += f"\n📊 Session ID: {current_session_id}"
        
        return current_session_id, display_df, status_msg
        
    except Exception as e:
        error_msg = f"""❌ ERROR STARTING SESSION

{str(e)}

Common issues:
- Check that CSV files have required columns
- Ensure budget/duration values are positive
- Verify database connection

Try using demo data (leave file uploads blank)
"""
        return None, pd.DataFrame(), error_msg


def approve_agent1(session_id: str, user_email: str, feedback: str) -> Tuple[str, pd.DataFrame, str]:
    """
    Approve Agent 1 and run Agent 2.
    
    Returns:
        (session_id, optimized_scope_df, status_message)
    """
    global current_thread_id, workflow_app
    
    try:
        if session_id is None:
            return session_id, pd.DataFrame(), "✗ No active session"
        
        # Get current state
        config = {"configurable": {"thread_id": current_thread_id}}
        current_state = workflow_app.get_state(config)
        
        # Update with approval
        current_state.values['agent1_approved'] = True
        current_state.values['agent1_approved_by'] = user_email
        current_state.values['agent1_approved_at'] = datetime.now().isoformat()
        
        # Continue workflow
        workflow_app.update_state(config, current_state.values)
        state = workflow_app.invoke(None, config)
        
        # Extract optimized scope
        optimized_df = pd.DataFrame(state['optimized_scope'])
        
        # Format for display
        display_df = optimized_df[[
            'rank', 'equipment_id', 'work_type', 'cost_estimate_usd',
            'expected_loss_avoided', 'confidence'
        ]].head(20)
        
        display_df.columns = [
            'Rank', 'Equipment', 'Work Type', 'Cost ($)',
            'Risk Avoided ($)', 'Confidence'
        ]
        
        summary = state['optimization_summary']
        status_msg = f"✓ Agent 1 approved by {user_email}"
        status_msg += f"\n✓ Agent 2 complete: {summary['total_included']} items included"
        status_msg += f"\n💰 Total cost: ${summary['total_cost']:,.0f}"
        
        return session_id, display_df, status_msg
        
    except Exception as e:
        return session_id, pd.DataFrame(), f"✗ Error: {str(e)}"


def approve_agent2(session_id: str, user_email: str, feedback: str) -> Tuple[str, str, str, str]:
    """
    Approve Agent 2 and run Agent 3.
    
    Returns:
        (session_id, memo_text, excel_path, pdf_path, status_message)
    """
    global current_thread_id, workflow_app
    
    try:
        if session_id is None:
            return session_id, "", "", "", "✗ No active session"
        
        # Get current state
        config = {"configurable": {"thread_id": current_thread_id}}
        current_state = workflow_app.get_state(config)
        
        # Update with approval
        current_state.values['agent2_approved'] = True
        current_state.values['agent2_approved_by'] = user_email
        current_state.values['agent2_approved_at'] = datetime.now().isoformat()
        
        # Continue workflow
        workflow_app.update_state(config, current_state.values)
        state = workflow_app.invoke(None, config)
        
        # Extract final outputs
        memo_text = state['executive_memo']
        
        # Generate Excel
        risk_df = pd.DataFrame(state['risk_analysis'])
        optimized_df = pd.DataFrame(state['optimized_scope'])
        deferred_df = pd.DataFrame(state['deferred_items']) if state['deferred_items'] else pd.DataFrame()
        
        wb = create_excel_workbook(
            risk_df,
            optimized_df,
            deferred_df,
            state['optimization_summary'],
            state['constraints'],
            session_id
        )
        excel_path = save_excel_workbook(wb, session_id=session_id)
        
        # Generate PDF
        pdf_path = create_pdf_memo(
            memo_text,
            state['optimization_summary'],
            state['constraints'],
            session_id
        )
        
        status_msg = f"""✅ WORKFLOW COMPLETE

✓ Agent 2 approved by: {user_email}
✓ Executive memo generated
✓ Files ready for download

📄 Outputs:
   • Excel: {os.path.basename(excel_path)}
   • PDF: {os.path.basename(pdf_path)}

➡️ View memo and download files in Tab 4
"""
        
        return session_id, memo_text, excel_path, pdf_path, status_msg
        
    # Load audit log for this session
        audit_df, audit_report = load_audit_log_for_session(session_id)
        
        return session_id, memo_text, excel_path, pdf_path, status_msg, audit_df, audit_report
        
    except Exception as e:
        return session_id, "", "", "", f"✗ Error: {str(e)}", pd.DataFrame(), ""


def reject_agent(session_id: str, agent_name: str, user_email: str, reason: str) -> str:
    """
    Reject an agent's output.
    
    Returns:
        status_message
    """
    # Log rejection (audit trail)
    from utils.audit import log_approval
    
    try:
        log_approval(
            session_id=session_id,
            thread_id=session_id,
            agent_name=agent_name,
            action="rejected",
            user_email=user_email,
            feedback=reason
        )
        
        return f"✗ {agent_name} rejected by {user_email}\nReason: {reason}\n\nWorkflow stopped. Please start a new session."
        
    except Exception as e:
        return f"✗ Error logging rejection: {str(e)}"


def run_example_workflow(user_email: str) -> Tuple[str, pd.DataFrame, pd.DataFrame, str, str, str]:
    """
    Run complete workflow with demo data (auto-approve all steps).
    
    Returns:
        (session_id, risk_df, scope_df, memo, excel_path, pdf_path, status)
    """
    try:
        # Step 1: Start with demo data
        session_id, risk_df, status1 = start_new_session(
            None, None, 18_500_000, 42, 100_000, 25.0, user_email
        )
        
        # Step 2: Auto-approve Agent 1
        session_id, scope_df, status2 = approve_agent1(session_id, user_email, "Auto-approved for example")
        
        # Step 3: Auto-approve Agent 2
        session_id, memo, excel_path, pdf_path, status3 = approve_agent2(session_id, user_email, "Auto-approved for example")
        
        final_status = f"""✅ EXAMPLE WORKFLOW COMPLETE

{status1}

{status2}

{status3}

📊 Files ready for download in Tab 4.
"""
        
        return session_id, risk_df, scope_df, memo, excel_path, pdf_path, final_status
        
    except Exception as e:
        return None, pd.DataFrame(), pd.DataFrame(), "", "", "", f"✗ Example failed: {str(e)}"


def load_audit_log_for_session(session_id: str = None) -> Tuple[pd.DataFrame, str]:
    """
    Load audit log entries for a session.
    
    Returns:
        (audit_dataframe, full_report_text)
    """
    try:
        from utils.audit import get_audit_log, generate_audit_report
        
        # Get entries
        if session_id and session_id.strip():
            entries = get_audit_log(session_id=session_id.strip(), limit=100)
            report = generate_audit_report(session_id.strip())
        else:
            # Get recent entries
            entries = get_audit_log(limit=20)
            report = "Showing 20 most recent audit entries across all sessions.\n\n"
            report += "Enter a specific Session ID to see full report for that session."
        
        if not entries:
            return pd.DataFrame(), "No audit log entries found."
        
        # Format for display
        display_data = []
        for entry in entries:
            display_data.append({
                'Timestamp': entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'Agent': entry['agent_name'],
                'Action': entry['action'].upper(),
                'User': entry['user_email'],
                'Feedback': entry['feedback'] or '—'
            })
        
        df = pd.DataFrame(display_data)
        
        return df, report
        
    except Exception as e:
        return pd.DataFrame(), f"Error loading audit log: {str(e)}"

# Build Gradio interface
with gr.Blocks(title="Refinery Turnaround Planner", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🏭 Refinery Turnaround Planning System
    
    **AI-powered scope optimization with human oversight**
    
    This system analyzes equipment failure risk, optimizes maintenance scope under constraints,
    and generates executive recommendations with full audit trails.
    """)
    
    # Session state
    session_state = gr.State(value=None)
    
    with gr.Tabs() as tabs:
        
        # TAB 1: Upload & Setup
        with gr.Tab("1️⃣ Upload Data & Start Analysis", id=0):
            gr.Markdown("### Upload Equipment & Scope Data")
            gr.Markdown("*Leave blank to use demo data*")
            
            with gr.Row():
                with gr.Column():
                    equipment_upload = gr.File(label="Equipment Registry (CSV)", file_types=[".csv"])
                    scope_upload = gr.File(label="Proposed Scope (CSV)", file_types=[".csv"])
                
                with gr.Column():
                    budget_input = gr.Number(
                        label="Budget Cap ($)",
                        value=18_500_000,
                        info="Maximum budget for turnaround"
                    )
                    duration_input = gr.Number(
                        label="Duration Cap (days)",
                        value=42,
                        info="Maximum turnaround duration"
                    )
            
            with gr.Row():
                capacity_input = gr.Number(
                    label="Production Capacity (bbl/day)",
                    value=100_000
                )
                margin_input = gr.Number(
                    label="Margin ($/bbl)",
                    value=25.0
                )
                user_email_input = gr.Textbox(
                    label="Your Email",
                    value="user@refinery.com",
                    info="For audit trail"
                )
            
            start_button = gr.Button("🚀 Start Analysis (Run Agent 1)", variant="primary", size="lg")
            
            tab1_status = gr.Textbox(label="Status", lines=3)
            
            gr.Markdown("### Agent 1: Risk Analysis Results (Top 20)")
            risk_table = gr.Dataframe(label="Risk-Ranked Scope Items")
        
        # TAB 2: Review Agent 1
        with gr.Tab("2️⃣ Review Risk Analysis", id=1):
            gr.Markdown("### Review Agent 1 Output")
            gr.Markdown("Agent 1 has calculated failure probability and economic consequences for each scope item.")
            
            risk_table_review = gr.Dataframe(label="Risk Analysis Results")
            
            with gr.Row():
                approve1_button = gr.Button("✅ Approve & Continue to Agent 2", variant="primary")
                reject1_button = gr.Button("❌ Reject", variant="stop")
            
            feedback1_input = gr.Textbox(
                label="Feedback (optional)",
                placeholder="Why approving or rejecting?",
                lines=2
            )
            
            tab2_status = gr.Textbox(label="Status", lines=3)
            scope_table = gr.Dataframe(label="Agent 2: Optimized Scope (Top 20)")
        
        # TAB 3: Review Agent 2
        with gr.Tab("3️⃣ Review Scope Optimization", id=2):
            gr.Markdown("### Review Agent 2 Output")
            gr.Markdown("Agent 2 has optimized scope selection under budget and duration constraints.")
            
            scope_table_review = gr.Dataframe(label="Included Scope Items")
            
            with gr.Row():
                approve2_button = gr.Button("✅ Approve & Generate Final Outputs", variant="primary")
                reject2_button = gr.Button("❌ Reject", variant="stop")
            
            feedback2_input = gr.Textbox(
                label="Feedback (optional)",
                lines=2
            )
            
            tab3_status = gr.Textbox(label="Status", lines=3)
        
        # TAB 4: Final Outputs
        with gr.Tab("4️⃣ Final Outputs & Downloads", id=3):
            gr.Markdown("### Executive Memo")
            memo_display = gr.Textbox(label="Agent 3: Executive Memo", lines=20)
            
            gr.Markdown("### Download Files")
            with gr.Row():
                excel_download = gr.File(label="📊 Excel Workbook")
                pdf_download = gr.File(label="📄 PDF Memo")
            
            tab4_status = gr.Textbox(label="Status", lines=2)

        # TAB 5: Audit Log (ADD THIS NEW TAB)
        with gr.Tab("📋 Audit Log", id=4):
            gr.Markdown("### Decision Trail")
            gr.Markdown("Complete history of approvals and rejections for this session.")
            
            with gr.Row():
                audit_session_input = gr.Textbox(
                    label="Session ID",
                    placeholder="Enter session ID or leave blank for recent entries",
                    scale=3
                )
                load_audit_button = gr.Button("🔍 Load Audit Log", scale=1)
            
            audit_table = gr.Dataframe(
                label="Approval History",
                headers=["Timestamp", "Agent", "Action", "User", "Feedback"]
            )
            
            gr.Markdown("### Full Audit Report")
            audit_report = gr.Textbox(label="Detailed Report", lines=15)


        # Examples section
    gr.Markdown("---")
    gr.Markdown("### 🎯 Quick Start with Example")
    
    with gr.Accordion("Click to expand example workflow", open=False):
        gr.Markdown("""
        **Try the system with one click:**
        
        This will run the complete workflow using demo data:
        - 50 equipment items from a typical refinery
        - 180+ historical failure events
        - 30 proposed maintenance tasks
        - $18.5M budget, 42-day duration constraint
        
        The system will generate risk analysis, optimize scope, and create executive outputs automatically.
        """)
        
        example_button = gr.Button("▶️ Run Complete Example Workflow", variant="secondary", size="lg")
        example_status = gr.Textbox(label="Example Status", lines=5)
    
    # Event handlers
    
    # Tab 1: Start analysis
    start_button.click(
        fn=start_new_session,
        inputs=[
            equipment_upload, scope_upload, budget_input, duration_input,
            capacity_input, margin_input, user_email_input
        ],
        outputs=[session_state, risk_table, tab1_status]
    ).then(
        fn=lambda x: x,  # Copy to review tab
        inputs=[risk_table],
        outputs=[risk_table_review]
    )
    
    # Tab 2: Approve Agent 1
    approve1_button.click(
        fn=approve_agent1,
        inputs=[session_state, user_email_input, feedback1_input],
        outputs=[session_state, scope_table, tab2_status]
    ).then(
        fn=lambda x: x,  # Copy to review tab
        inputs=[scope_table],
        outputs=[scope_table_review]
    )
    
    # Tab 2: Reject Agent 1
    reject1_button.click(
        fn=lambda s, e, r: reject_agent(s, "agent1", e, r),
        inputs=[session_state, user_email_input, feedback1_input],
        outputs=[tab2_status]
    )
    
    # Tab 3: Approve Agent 2
    approve2_button.click(
        fn=approve_agent2,
        inputs=[session_state, user_email_input, feedback2_input],
        outputs=[session_state, memo_display, excel_download, pdf_download, tab3_status]
    ).then(
        fn=lambda x: x,  # Copy status to tab 4
        inputs=[tab3_status],
        outputs=[tab4_status]
    ).then(
        fn=lambda s: s,  # Populate audit session ID
        inputs=[session_state],
        outputs=[audit_session_input]
    )
    
    # Tab 3: Reject Agent 2
    reject2_button.click(
        fn=lambda s, e, r: reject_agent(s, "agent2", e, r),
        inputs=[session_state, user_email_input, feedback2_input],
        outputs=[tab3_status]
    )

    # Tab 5: Load audit log (ADD THIS)
    load_audit_button.click(
        fn=load_audit_log_for_session,
        inputs=[audit_session_input],
        outputs=[audit_table, audit_report]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)
