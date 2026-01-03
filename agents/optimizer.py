"""
Agent 2: Scope / Portfolio Optimizer

Optimizes scope selection under budget, duration, and dependency constraints.
Uses integer linear programming (PuLP + CBC solver).
"""

import pandas as pd
import numpy as np
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatus, value
from typing import Dict, Tuple, List


def optimize_scope(
    risk_analysis: pd.DataFrame,
    budget_cap: float,
    duration_cap: float,
    regulatory_override: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Optimize scope selection to maximize risk reduction under constraints.
    
    Args:
        risk_analysis: Output from Agent 1 (risk-ranked items)
        budget_cap: Maximum budget ($)
        duration_cap: Maximum duration (days, assuming some parallelization)
        regulatory_override: Force include all regulatory items
    
    Returns:
        (included_scope, deferred_items, optimization_summary)
    """
    
    # Create optimization problem
    prob = LpProblem("Turnaround_Scope_Optimization", LpMaximize)
    
    # Decision variables (binary: include or not)
    items = risk_analysis.to_dict('records')
    include = {i: LpVariable(f"x_{i}", cat='Binary') for i in range(len(items))}
    
    # Objective: Maximize total expected loss avoided
    prob += lpSum([
        items[i]['expected_loss_avoided'] * include[i] 
        for i in range(len(items))
    ]), "Total_Risk_Avoided"
    
    # Constraint 1: Budget cap
    prob += lpSum([
        items[i]['cost_estimate_usd'] * include[i] 
        for i in range(len(items))
    ]) <= budget_cap, "Budget_Constraint"
    
    # Constraint 2: Duration cap (simplified - assumes 50% parallelization)
    # In reality, would need dependency graph
    prob += lpSum([
        items[i]['duration_days'] * include[i] * 0.5  # Parallel factor
        for i in range(len(items))
    ]) <= duration_cap, "Duration_Constraint"
    
    # Constraint 3: Regulatory items are mandatory
    if regulatory_override:
        for i in range(len(items)):
            if items[i]['is_regulatory']:
                prob += include[i] == 1, f"Regulatory_Mandatory_{i}"
    
    # Solve
    prob.solve()
    
    # Check if solution found
    status = LpStatus[prob.status]
    if status != 'Optimal':
        raise ValueError(f"Optimization failed: {status}")
    
    # Extract results
    included_items = []
    deferred_items = []
    
    for i in range(len(items)):
        item = items[i].copy()
        
        if value(include[i]) == 1:
            included_items.append(item)
        else:
            item['deferral_reason'] = _determine_deferral_reason(
                item, risk_analysis, budget_cap, duration_cap
            )
            deferred_items.append(item)
    
    included_df = pd.DataFrame(included_items)
    deferred_df = pd.DataFrame(deferred_items)
    
    # Calculate summary statistics
    summary = {
        'total_included': len(included_df),
        'total_deferred': len(deferred_df),
        'total_cost': included_df['cost_estimate_usd'].sum() if len(included_df) > 0 else 0,
        'total_duration_sequential': included_df['duration_days'].sum() if len(included_df) > 0 else 0,
        'total_duration_parallel': (included_df['duration_days'].sum() * 0.5) if len(included_df) > 0 else 0,
        'total_risk_avoided': included_df['expected_loss_avoided'].sum() if len(included_df) > 0 else 0,
        'residual_risk': deferred_df['expected_loss_avoided'].sum() if len(deferred_df) > 0 else 0,
        'budget_utilization_pct': (included_df['cost_estimate_usd'].sum() / budget_cap) * 100 if len(included_df) > 0 else 0,
        'optimization_status': status
    }
    
    return included_df, deferred_df, summary


def _determine_deferral_reason(
    item: Dict,
    risk_analysis: pd.DataFrame,
    budget_cap: float,
    duration_cap: float
) -> str:
    """
    Determine why an item was deferred.
    """
    # Check confidence level
    if item['confidence'] == 'low':
        return "Low confidence in risk assessment; recommend updated inspection"
    
    # Check if regulatory (shouldn't be deferred, but flag if it happens)
    if item['is_regulatory']:
        return "ERROR: Regulatory item deferred - manual review required"
    
    # Calculate marginal ROI (avoided loss per dollar)
    marginal_roi = item['expected_loss_avoided'] / item['cost_estimate_usd']
    
    # Compare to included items
    included_items = risk_analysis[
        risk_analysis['scope_item_id'].isin(
            risk_analysis[risk_analysis['expected_loss_avoided'] > item['expected_loss_avoided']]
            ['scope_item_id']
        )
    ]
    
    if len(included_items) > 0:
        avg_included_roi = (
            included_items['expected_loss_avoided'].sum() / 
            included_items['cost_estimate_usd'].sum()
        )
        
        if marginal_roi < avg_included_roi * 0.5:
            return f"Low marginal ROI (${marginal_roi:.2f} per $1K vs ${avg_included_roi:.2f} avg)"
    
    return "Budget exhausted; marginal ROI below inclusion threshold"


def calculate_marginal_analysis(
    included_df: pd.DataFrame,
    deferred_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate marginal ROI for each item (included and deferred).
    Shows "last item in" and "first item out" for sensitivity analysis.
    """
    all_items = pd.concat([included_df, deferred_df], ignore_index=True)
    
    all_items['marginal_roi_per_1k'] = (
        all_items['expected_loss_avoided'] / 
        (all_items['cost_estimate_usd'] / 1000)
    )
    
    all_items['marginal_roi_per_day'] = (
        all_items['expected_loss_avoided'] / 
        all_items['duration_days']
    )
    
    all_items['included'] = all_items['scope_item_id'].isin(included_df['scope_item_id'])
    
    # Sort by marginal ROI
    all_items = all_items.sort_values('marginal_roi_per_1k', ascending=False)
    
    return all_items


def generate_sensitivity_scenarios(
    risk_analysis: pd.DataFrame,
    base_budget: float,
    base_duration: float
) -> pd.DataFrame:
    """
    Run optimization under different budget/duration scenarios.
    """
    scenarios = []
    
    # Budget sensitivity
    for budget_delta in [-0.1, 0, 0.1, 0.2]:
        new_budget = base_budget * (1 + budget_delta)
        
        try:
            included, deferred, summary = optimize_scope(
                risk_analysis, new_budget, base_duration
            )
            
            scenarios.append({
                'scenario': f"Budget {budget_delta:+.0%}",
                'budget': new_budget,
                'duration': base_duration,
                'items_included': summary['total_included'],
                'total_cost': summary['total_cost'],
                'risk_avoided': summary['total_risk_avoided'],
                'residual_risk': summary['residual_risk']
            })
        except ValueError:
            # Infeasible scenario
            scenarios.append({
                'scenario': f"Budget {budget_delta:+.0%}",
                'budget': new_budget,
                'duration': base_duration,
                'items_included': 0,
                'total_cost': 0,
                'risk_avoided': 0,
                'residual_risk': risk_analysis['expected_loss_avoided'].sum()
            })
    
    # Duration sensitivity
    for duration_delta in [-0.1, 0.1, 0.2]:
        new_duration = base_duration * (1 + duration_delta)
        
        try:
            included, deferred, summary = optimize_scope(
                risk_analysis, base_budget, new_duration
            )
            
            scenarios.append({
                'scenario': f"Duration {duration_delta:+.0%}",
                'budget': base_budget,
                'duration': new_duration,
                'items_included': summary['total_included'],
                'total_cost': summary['total_cost'],
                'risk_avoided': summary['total_risk_avoided'],
                'residual_risk': summary['residual_risk']
            })
        except ValueError:
            scenarios.append({
                'scenario': f"Duration {duration_delta:+.0%}",
                'budget': base_budget,
                'duration': new_duration,
                'items_included': 0,
                'total_cost': 0,
                'risk_avoided': 0,
                'residual_risk': risk_analysis['expected_loss_avoided'].sum()
            })
    
    return pd.DataFrame(scenarios)


def run_agent2(
    risk_analysis: pd.DataFrame,
    budget_cap: float,
    duration_cap: float
) -> Dict:
    """
    Main Agent 2 function: Optimize scope and generate analysis.
    
    Returns dict with:
        - included_scope
        - deferred_items
        - marginal_analysis
        - sensitivity_scenarios
        - summary
    """
    
    # Run optimization
    included_df, deferred_df, summary = optimize_scope(
        risk_analysis, budget_cap, duration_cap
    )
    
    # Marginal analysis
    marginal_df = calculate_marginal_analysis(included_df, deferred_df)
    
    # Sensitivity scenarios
    sensitivity_df = generate_sensitivity_scenarios(
        risk_analysis, budget_cap, duration_cap
    )
    
    return {
        'included_scope': included_df,
        'deferred_items': deferred_df,
        'marginal_analysis': marginal_df,
        'sensitivity_scenarios': sensitivity_df,
        'summary': summary
    }


if __name__ == "__main__":
    """Test Agent 2 with Agent 1 output"""
    import pandas as pd
    
    # Load Agent 1 output
    risk_analysis = pd.read_csv('data/agent1_risk_analysis.csv')
    
    print("=" * 60)
    print("AGENT 2: SCOPE / PORTFOLIO OPTIMIZER")
    print("=" * 60)
    
    # Set constraints
    BUDGET_CAP = 18_500_000  # $18.5M
    DURATION_CAP = 42  # 42 days
    
    print(f"\nConstraints:")
    print(f"  Budget cap: ${BUDGET_CAP:,.0f}")
    print(f"  Duration cap: {DURATION_CAP} days")
    
    # Run optimization
    results = run_agent2(risk_analysis, BUDGET_CAP, DURATION_CAP)
    
    # Display results
    summary = results['summary']
    print(f"\n✓ Optimization complete: {summary['optimization_status']}")
    print(f"\nScope Summary:")
    print(f"  Items included: {summary['total_included']}")
    print(f"  Items deferred: {summary['total_deferred']}")
    print(f"  Total cost: ${summary['total_cost']:,.0f} ({summary['budget_utilization_pct']:.1f}% of budget)")
    print(f"  Estimated duration: {summary['total_duration_parallel']:.1f} days (parallel)")
    print(f"  Risk avoided: ${summary['total_risk_avoided']:,.0f}")
    print(f"  Residual risk: ${summary['residual_risk']:,.0f}")
    
    # Save results
    results['included_scope'].to_csv('data/agent2_included_scope.csv', index=False)
    results['deferred_items'].to_csv('data/agent2_deferred_items.csv', index=False)
    results['marginal_analysis'].to_csv('data/agent2_marginal_analysis.csv', index=False)
    results['sensitivity_scenarios'].to_csv('data/agent2_sensitivity.csv', index=False)
    
    print(f"\n✓ Saved results:")
    print(f"  • data/agent2_included_scope.csv")
    print(f"  • data/agent2_deferred_items.csv")
    print(f"  • data/agent2_marginal_analysis.csv")
    print(f"  • data/agent2_sensitivity.csv")
    
    if len(results['deferred_items']) > 0:
        print(f"\nTop 5 Deferred Items (by risk):")
        print(results['deferred_items'][['equipment_id', 'expected_loss_avoided', 'deferral_reason']].head().to_string(index=False))
    else:
        print(f"\n✓ All items fit within constraints - no deferrals needed!")
