"""
Agent 3: Executive Translator

Converts technical risk analysis and optimization results into 
plain-language executive memos with explicit uncertainty statements.

Uses Llama 3.3 70B via Hugging Face Inference API.
"""

import os
from dotenv import load_dotenv
from groq import Groq
import pandas as pd
from typing import Dict

load_dotenv()


def create_llm_client():
    """
    Create Groq API client (free tier).
    """
    token = os.environ.get('GROQ_API_KEY')
    if not token:
        raise ValueError("GROQ_API_KEY not found in environment")
    
    return Groq(api_key=token)


def build_prompt(
    risk_analysis: pd.DataFrame,
    optimized_scope: pd.DataFrame,
    deferred_items: pd.DataFrame,
    optimization_summary: Dict,
    constraints: Dict
) -> str:
    """
    Build prompt for LLM to generate executive memo.
    """
    
    # Prepare data summaries
    total_items = len(risk_analysis)
    included_count = len(optimized_scope)
    deferred_count = len(deferred_items)
    
    budget_cap = constraints['budget_cap']
    duration_cap = constraints['duration_cap']
    
    total_cost = optimization_summary['total_cost']
    total_risk_avoided = optimization_summary['total_risk_avoided']
    residual_risk = optimization_summary['residual_risk']
    
    # Top 5 included items
    top_included = optimized_scope.nlargest(5, 'expected_loss_avoided')[
        ['equipment_id', 'work_type', 'expected_loss_avoided', 'confidence']
    ]
    
    # Top 5 deferred items (if any)
    top_deferred = deferred_items.nlargest(5, 'expected_loss_avoided')[
        ['equipment_id', 'work_type', 'expected_loss_avoided', 'deferral_reason']
    ] if len(deferred_items) > 0 else pd.DataFrame()
    
    # Confidence breakdown
    confidence_dist = optimized_scope['confidence'].value_counts().to_dict()
    
    prompt = f"""You are a technical writer creating an executive memo for refinery operations leaders.

INPUT DATA:
- Total scope items analyzed: {total_items}
- Items included in scope: {included_count}
- Items deferred: {deferred_count}
- Budget cap: ${budget_cap:,.0f}
- Actual cost: ${total_cost:,.0f} ({(total_cost/budget_cap)*100:.1f}% utilization)
- Duration cap: {duration_cap} days
- Risk avoided: ${total_risk_avoided:,.0f}
- Residual risk: ${residual_risk:,.0f}

TOP 5 INCLUDED ITEMS:
{top_included.to_string(index=False)}

TOP DEFERRED ITEMS (if any):
{top_deferred.to_string(index=False) if len(top_deferred) > 0 else "None - all items fit within constraints"}

CONFIDENCE DISTRIBUTION:
{confidence_dist}

CRITICAL RULES:
1. Use plain language - NO jargon (no "MTBF," "expected value," "optimization")
2. Every number must cite its source from the input data above
3. State uncertainty upfront: "High confidence in X of Y items due to [reason]"
4. Use accessible units: "Equivalent to 2 days of lost production" NOT "$4.2M NPV"
5. Structure: Recommendation → Key Inclusions → Accepted Risks → Confidence & Assumptions
6. Be concise - maximum 400 words total

OUTPUT FORMAT (markdown):

# Turnaround Scope Recommendation

## Recommendation
[2-3 sentences: budget, duration, risk coverage percentage]

## Key Inclusions
[3 highest-risk items with plain-language rationale - why each matters to operations]

## Accepted Risks
[Deferred items with residual exposure - what we're NOT doing and why]

## Confidence & Assumptions
[Data quality summary, key assumptions, conditions that would trigger revisiting this plan]

Generate the memo now:"""

    return prompt


def generate_memo(
    risk_analysis: pd.DataFrame,
    optimized_scope: pd.DataFrame,
    deferred_items: pd.DataFrame,
    optimization_summary: Dict,
    constraints: Dict,
    model: str = "llama-3.3-70b-versatile" # Groq's Llama 3.3 70B
) -> str:
    """
    Generate executive memo using Llama 3.3 70B.
    
    Returns:
        Memo text in markdown format
    """
    
    # Build prompt
    prompt = build_prompt(
        risk_analysis,
        optimized_scope,
        deferred_items,
        optimization_summary,
        constraints
    )
    
    # Call LLM
    client = create_llm_client()
    
    print(f"[Agent 3] Calling Groq with {model}...")
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
            top_p=0.9,
        )
        
        generated_text = response.choices[0].message.content.strip()
        
        print(f"[Agent 3] ✓ Memo generated ({len(generated_text)} characters)")
        
        return generated_text

    except Exception as e:
        print(f"[Agent 3] ✗ LLM error: {e}")
        print("[Agent 3] Falling back to template-based memo")
        
        return generate_fallback_memo(
            optimized_scope,
            deferred_items,
            optimization_summary,
            constraints
        )


def generate_fallback_memo(
    optimized_scope: pd.DataFrame,
    deferred_items: pd.DataFrame,
    optimization_summary: Dict,
    constraints: Dict
) -> str:
    """
    Fallback template-based memo if LLM fails.
    """
    
    memo = f"""# Turnaround Scope Recommendation

## Recommendation

Proposed scope includes {optimization_summary['total_included']} maintenance items at ${optimization_summary['total_cost']:,.0f} ({(optimization_summary['total_cost']/constraints['budget_cap'])*100:.1f}% of ${constraints['budget_cap']:,.0f} budget). Estimated duration is {optimization_summary['total_duration_parallel']:.0f} days with parallel execution.

This scope addresses ${optimization_summary['total_risk_avoided']:,.0f} in quantified failure risk, representing the expected cost of equipment failures if this work is deferred.

## Key Inclusions

The scope prioritizes equipment with highest failure risk and business impact:

"""
    
    # Add top 3 items
    top_3 = optimized_scope.nlargest(3, 'expected_loss_avoided')
    for idx, item in top_3.iterrows():
        memo += f"- **{item['equipment_id']}** ({item['work_type']}): Addresses ${item['expected_loss_avoided']:,.0f} expected loss\n"
    
    memo += f"""
## Accepted Risks

{optimization_summary['total_deferred']} items deferred due to budget and schedule constraints, carrying ${optimization_summary['residual_risk']:,.0f} residual risk.
"""
    
    if len(deferred_items) > 0:
        memo += "\nTop deferred items:\n"
        for idx, item in deferred_items.head(3).iterrows():
            memo += f"- {item['equipment_id']}: ${item['expected_loss_avoided']:,.0f} risk - {item['deferral_reason']}\n"
    
    memo += f"""
## Confidence & Assumptions

Analysis based on historical failure data and current equipment condition assessments. Cost estimates assume standard contractor rates and normal material availability. Recommend revisiting if equipment condition degrades or if additional failures occur before turnaround execution.
"""
    
    return memo


def run_agent3(
    risk_analysis: pd.DataFrame,
    optimized_scope: pd.DataFrame,
    deferred_items: pd.DataFrame,
    optimization_summary: Dict,
    constraints: Dict
) -> str:
    """
    Main Agent 3 function: Generate executive memo.
    
    Returns:
        Executive memo text (markdown)
    """
    
    print("=" * 60)
    print("AGENT 3: EXECUTIVE TRANSLATOR")
    print("=" * 60)
    
    memo = generate_memo(
        risk_analysis,
        optimized_scope,
        deferred_items,
        optimization_summary,
        constraints
    )
    
    return memo


if __name__ == "__main__":
    """Test Agent 3 with Agent 2 output"""
    import pandas as pd
    
    # Load Agent 2 outputs
    risk_analysis = pd.read_csv('data/agent1_risk_analysis.csv')
    optimized_scope = pd.read_csv('data/agent2_included_scope.csv')
    
    # Handle empty deferred items file
    try:
        deferred_items = pd.read_csv('data/agent2_deferred_items.csv')
        if len(deferred_items) == 0:
            # Create empty DataFrame with expected columns
            deferred_items = pd.DataFrame(columns=[
                'equipment_id', 'work_type', 'expected_loss_avoided', 'deferral_reason'
            ])
    except (pd.errors.EmptyDataError, FileNotFoundError):
        # File is empty or doesn't exist
        deferred_items = pd.DataFrame(columns=[
            'equipment_id', 'work_type', 'expected_loss_avoided', 'deferral_reason'
        ])
    
    # Load optimization summary (reconstruct from CSVs)
    optimization_summary = {
        'total_included': len(optimized_scope),
        'total_deferred': len(deferred_items),
        'total_cost': optimized_scope['cost_estimate_usd'].sum(),
        'total_duration_parallel': optimized_scope['duration_days'].sum() * 0.5,
        'total_risk_avoided': optimized_scope['expected_loss_avoided'].sum(),
        'residual_risk': deferred_items['expected_loss_avoided'].sum() if len(deferred_items) > 0 else 0
    }
    
    constraints = {
        'budget_cap': 18_500_000,
        'duration_cap': 42,
        'production_capacity_bbl_day': 100000,
        'margin_per_bbl': 25.0
    }
    
    # Generate memo
    memo = run_agent3(
        risk_analysis,
        optimized_scope,
        deferred_items,
        optimization_summary,
        constraints
    )
    
    print("\n" + "=" * 60)
    print("GENERATED MEMO")
    print("=" * 60)
    print(memo)
    
    # Save memo
    with open('data/agent3_executive_memo.md', 'w') as f:
        f.write(memo)
    
    print("\n✓ Memo saved to data/agent3_executive_memo.md") 
