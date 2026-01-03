"""
Agent 1: Risk & Economics Analyst

Deterministic risk calculations based on historical failure data.
No recommendations - only quantifies risk exposure.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple


def calculate_failure_probability(
    equipment_id: str,
    equipment_data: pd.DataFrame,
    failure_history: pd.DataFrame
) -> Tuple[float, str, str]:
    """
    Calculate probability of failure within next operating period.
    
    Returns:
        (probability, confidence_level, data_sources)
    """
    # Get equipment info
    equip = equipment_data[equipment_data['equipment_id'] == equipment_id].iloc[0]
    
    # Get failure history for this equipment
    failures = failure_history[failure_history['equipment_id'] == equipment_id].copy()
    
    if len(failures) == 0:
        # No history - use conservative industry default
        return 0.05, 'low', 'industry_default_no_history'
    
    # Calculate MTBF (Mean Time Between Failures)
    failures['failure_date'] = pd.to_datetime(failures['failure_date'])
    failures = failures.sort_values('failure_date')
    
    if len(failures) == 1:
        # Only one failure - use time since installation
        time_span_days = (datetime.now() - pd.to_datetime(equip['install_date'])).days
        mtbf_days = time_span_days
        confidence = 'low'
    else:
        # Multiple failures - calculate intervals
        intervals = failures['failure_date'].diff().dt.days.dropna()
        mtbf_days = intervals.mean()
        confidence = 'high' if len(failures) >= 5 else 'medium'
    
    # Time since last failure
    last_failure = failures['failure_date'].max()
    days_since_failure = (datetime.now() - last_failure).days
    
    # Base probability from MTBF (exponential distribution)
    # P(failure in next year) = 1 - exp(-365/MTBF)
    operating_period_days = 365  # Next operating period (1 year)
    base_prob = 1 - np.exp(-operating_period_days / mtbf_days)
    
    # Adjust for age (older equipment more likely to fail)
    age_years = equip['current_age_years']
    design_life = equip['design_life_years']
    age_factor = 1 + (age_years / design_life) * 0.5  # Up to 50% increase
    
    # Adjust for condition score (1=poor, 10=excellent)
    condition_score = equip['condition_score']
    condition_factor = (11 - condition_score) / 10  # Poor condition increases prob
    
    # Combined probability
    adjusted_prob = base_prob * age_factor * condition_factor
    adjusted_prob = min(adjusted_prob, 0.95)  # Cap at 95%
    
    # Confidence adjustment based on data recency
    data_age_days = (datetime.now() - last_failure).days
    if data_age_days > 1095:  # >3 years old
        confidence = 'low' if confidence == 'high' else 'medium'
    
    data_sources = f"mtbf_{len(failures)}_events"
    
    return round(adjusted_prob, 3), confidence, data_sources


def calculate_consequence(
    equipment_id: str,
    equipment_data: pd.DataFrame,
    failure_history: pd.DataFrame,
    production_capacity_bbl_day: float,
    margin_per_bbl: float
) -> Tuple[int, float, str]:
    """
    Calculate economic consequence of failure (repair cost + downtime impact).
    
    Returns:
        (total_consequence_usd, downtime_days, calculation_source)
    """
    # Get equipment info
    equip = equipment_data[equipment_data['equipment_id'] == equipment_id].iloc[0]
    
    # Get historical failures for this equipment
    failures = failure_history[failure_history['equipment_id'] == equipment_id]
    
    if len(failures) == 0:
        # No history - use criticality-based estimates
        criticality = equip['criticality']
        repair_cost_map = {1: 250000, 2: 100000, 3: 30000, 4: 10000}
        downtime_map = {1: 4.0, 2: 2.0, 3: 1.0, 4: 0.5}
        
        repair_cost = repair_cost_map[criticality]
        downtime_days = downtime_map[criticality]
        source = 'criticality_based_estimate'
    else:
        # Use historical averages
        repair_cost = int(failures['repair_cost_usd'].mean())
        downtime_days = failures['downtime_days'].mean()
        source = f'historical_avg_{len(failures)}_events'
    
    # Calculate production loss
    # Criticality 1 = full plant shutdown
    # Criticality 2 = partial impact (50%)
    # Criticality 3+ = localized (20%)
    impact_factors = {1: 1.0, 2: 0.5, 3: 0.2, 4: 0.1}
    impact_factor = impact_factors[equip['criticality']]
    
    daily_production_loss = production_capacity_bbl_day * impact_factor * margin_per_bbl
    downtime_cost = downtime_days * daily_production_loss
    
    total_consequence = int(repair_cost + downtime_cost)
    
    return total_consequence, round(downtime_days, 2), source


def calculate_expected_loss(
    failure_prob: float,
    consequence_usd: int
) -> int:
    """
    Expected loss = Probability × Consequence
    """
    return int(failure_prob * consequence_usd)


def assign_risk_tier(expected_loss: int) -> str:
    """
    Categorize risk into tiers for prioritization.
    """
    if expected_loss >= 5_000_000:
        return 'critical'
    elif expected_loss >= 1_000_000:
        return 'high'
    elif expected_loss >= 100_000:
        return 'medium'
    else:
        return 'low'


def run_agent1(
    equipment_data: pd.DataFrame,
    failure_history: pd.DataFrame,
    scope_items: pd.DataFrame,
    production_capacity_bbl_day: float = 100000,
    margin_per_bbl: float = 25.0
) -> pd.DataFrame:
    """
    Main Agent 1 function: Calculate risk for all scope items.
    
    Returns DataFrame with risk analysis for each scope item.
    """
    results = []
    
    for _, scope_item in scope_items.iterrows():
        equipment_id = scope_item['equipment_id']
        
        # Calculate failure probability
        prob, confidence, prob_source = calculate_failure_probability(
            equipment_id, equipment_data, failure_history
        )
        
        # Calculate consequence
        consequence, downtime, cons_source = calculate_consequence(
            equipment_id, equipment_data, failure_history,
            production_capacity_bbl_day, margin_per_bbl
        )
        
        # Calculate expected loss
        expected_loss = calculate_expected_loss(prob, consequence)
        
        # Assign risk tier
        risk_tier = assign_risk_tier(expected_loss)
        
        results.append({
            'scope_item_id': scope_item['scope_item_id'],
            'equipment_id': equipment_id,
            'work_type': scope_item['work_type'],
            'description': scope_item['description'],
            'cost_estimate_usd': scope_item['cost_estimate_usd'],
            'duration_days': scope_item['duration_days'],
            'is_regulatory': scope_item['is_regulatory'],
            'failure_probability': prob,
            'consequence_usd': consequence,
            'downtime_days': downtime,
            'expected_loss_avoided': expected_loss,
            'risk_tier': risk_tier,
            'confidence': confidence,
            'data_sources': f"{prob_source}|{cons_source}"
        })
    
    risk_df = pd.DataFrame(results)
    
    # Sort by expected loss (highest risk first)
    risk_df = risk_df.sort_values('expected_loss_avoided', ascending=False)
    risk_df = risk_df.reset_index(drop=True)
    
    # Add rank
    risk_df.insert(0, 'rank', range(1, len(risk_df) + 1))
    
    return risk_df


if __name__ == "__main__":
    """Test Agent 1 with synthetic data"""
    import sqlite3
    
    # Load data
    conn = sqlite3.connect('data/demo.db')
    equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
    failures_df = pd.read_sql('SELECT * FROM failures', conn)
    scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
    conn.close()
    
    # Run risk analysis
    print("=" * 60)
    print("AGENT 1: RISK & ECONOMICS ANALYST")
    print("=" * 60)
    
    risk_results = run_agent1(
        equipment_df, 
        failures_df, 
        scope_df,
        production_capacity_bbl_day=100000,
        margin_per_bbl=25.0
    )
    
    print(f"\n✓ Analyzed {len(risk_results)} scope items")
    print(f"\nRisk Tier Distribution:")
    print(risk_results['risk_tier'].value_counts())
    
    print(f"\nTop 10 Highest Risk Items:")
    print(risk_results[['rank', 'equipment_id', 'work_type', 'expected_loss_avoided', 
                        'risk_tier', 'confidence']].head(10).to_string(index=False))
    
    print(f"\nConfidence Distribution:")
    print(risk_results['confidence'].value_counts())
    
    # Save results
    risk_results.to_csv('data/agent1_risk_analysis.csv', index=False)
    print(f"\n✓ Saved results to data/agent1_risk_analysis.csv")
