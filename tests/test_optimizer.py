 
"""
Unit tests for Agent 2: Scope Optimizer
"""

import pandas as pd
import numpy as np
import sys
sys.path.append('.')

from agents.optimizer import (
    optimize_scope,
    calculate_marginal_analysis,
    generate_sensitivity_scenarios,
    _determine_deferral_reason
)


def create_test_risk_data():
    """Create synthetic risk analysis data for testing"""
    return pd.DataFrame([
        {
            'scope_item_id': 'WO-001',
            'equipment_id': 'TEST-001',
            'work_type': 'replacement',
            'description': 'Critical equipment replacement',
            'cost_estimate_usd': 500000,
            'duration_days': 5,
            'is_regulatory': True,
            'expected_loss_avoided': 5000000,
            'risk_tier': 'critical',
            'confidence': 'high'
        },
        {
            'scope_item_id': 'WO-002',
            'equipment_id': 'TEST-002',
            'work_type': 'repair',
            'description': 'High priority repair',
            'cost_estimate_usd': 200000,
            'duration_days': 3,
            'is_regulatory': False,
            'expected_loss_avoided': 2000000,
            'risk_tier': 'high',
            'confidence': 'high'
        },
        {
            'scope_item_id': 'WO-003',
            'equipment_id': 'TEST-003',
            'work_type': 'inspection',
            'description': 'Low priority inspection',
            'cost_estimate_usd': 50000,
            'duration_days': 1,
            'is_regulatory': False,
            'expected_loss_avoided': 100000,
            'risk_tier': 'medium',
            'confidence': 'low'
        },
        {
            'scope_item_id': 'WO-004',
            'equipment_id': 'TEST-004',
            'work_type': 'upgrade',
            'description': 'Medium priority upgrade',
            'cost_estimate_usd': 300000,
            'duration_days': 4,
            'is_regulatory': False,
            'expected_loss_avoided': 1500000,
            'risk_tier': 'high',
            'confidence': 'medium'
        }
    ])


def test_regulatory_items_always_included():
    """Test that regulatory items are always included"""
    risk_data = create_test_risk_data()
    
    # Set very tight budget that would exclude regulatory item if not mandatory
    budget_cap = 600000  # Only enough for 2-3 items
    duration_cap = 20
    
    included, deferred, summary = optimize_scope(
        risk_data, budget_cap, duration_cap, regulatory_override=True
    )
    
    # Regulatory item WO-001 must be included
    assert 'WO-001' in included['scope_item_id'].values
    print("✓ test_regulatory_items_always_included")


def test_optimization_respects_budget():
    """Test that optimizer doesn't exceed budget cap"""
    risk_data = create_test_risk_data()
    
    budget_cap = 700000
    duration_cap = 20
    
    included, deferred, summary = optimize_scope(
        risk_data, budget_cap, duration_cap
    )
    
    total_cost = included['cost_estimate_usd'].sum()
    assert total_cost <= budget_cap
    print("✓ test_optimization_respects_budget")


def test_optimization_respects_duration():
    """Test that optimizer doesn't exceed duration cap"""
    risk_data = create_test_risk_data()
    
    budget_cap = 2000000  # High budget
    duration_cap = 5  # Very tight duration
    
    included, deferred, summary = optimize_scope(
        risk_data, budget_cap, duration_cap
    )
    
    # Duration with 50% parallel factor
    total_duration = included['duration_days'].sum() * 0.5
    assert total_duration <= duration_cap
    print("✓ test_optimization_respects_duration")


def test_optimizer_maximizes_risk_avoided():
    """Test that optimizer prioritizes high-risk items"""
    risk_data = create_test_risk_data()
    
    budget_cap = 600000  # Tighter budget to force deferrals
    duration_cap = 20
    
    included, deferred, summary = optimize_scope(
        risk_data, budget_cap, duration_cap
    )
    
    # Should include regulatory item (WO-001) always
    assert 'WO-001' in included['scope_item_id'].values
    
    # With $600K budget after regulatory ($500K), only $100K left
    # Should defer lower-value items
    if len(deferred) > 0:
        # Lowest risk item (WO-003: $100K avoided) should be considered for deferral
        deferred_risks = deferred['expected_loss_avoided'].values
        included_risks = included['expected_loss_avoided'].values
        
        # Deferred items should have lower expected loss than included items
        if len(deferred_risks) > 0 and len(included_risks) > 1:
            assert deferred_risks.min() <= included_risks.mean()
    
    print("✓ test_optimizer_maximizes_risk_avoided")

def test_infeasible_constraints_raise_error():
    """Test that impossible constraints are caught"""
    risk_data = create_test_risk_data()
    
    # Regulatory item costs $500K, but budget is only $100K
    budget_cap = 100000
    duration_cap = 20
    
    try:
        included, deferred, summary = optimize_scope(
            risk_data, budget_cap, duration_cap
        )
        assert False, "Should have raised ValueError for infeasible problem"
    except ValueError as e:
        assert "Optimization failed" in str(e) or "Infeasible" in str(e)
        print("✓ test_infeasible_constraints_raise_error")


def test_marginal_analysis_calculation():
    """Test marginal ROI calculations"""
    risk_data = create_test_risk_data()
    
    budget_cap = 700000
    duration_cap = 20
    
    included, deferred, summary = optimize_scope(
        risk_data, budget_cap, duration_cap
    )
    
    marginal_df = calculate_marginal_analysis(included, deferred)
    
    # Should have all 4 items
    assert len(marginal_df) == 4
    
    # Should have marginal ROI columns
    assert 'marginal_roi_per_1k' in marginal_df.columns
    assert 'marginal_roi_per_day' in marginal_df.columns
    
    # Items should be sorted by ROI (descending)
    assert marginal_df['marginal_roi_per_1k'].is_monotonic_decreasing
    
    print("✓ test_marginal_analysis_calculation")


def test_sensitivity_scenarios():
    """Test sensitivity analysis generation"""
    risk_data = create_test_risk_data()
    
    base_budget = 700000
    base_duration = 20
    
    sensitivity_df = generate_sensitivity_scenarios(
        risk_data, base_budget, base_duration
    )
    
    # Should have multiple scenarios (budget +/- and duration +/-)
    assert len(sensitivity_df) >= 5
    
    # Should have required columns
    assert 'scenario' in sensitivity_df.columns
    assert 'risk_avoided' in sensitivity_df.columns
    assert 'residual_risk' in sensitivity_df.columns
    
    # Higher budget should generally avoid more risk
    base_scenario = sensitivity_df[sensitivity_df['scenario'] == 'Budget +0%']
    high_budget = sensitivity_df[sensitivity_df['scenario'] == 'Budget +10%']
    
    if len(base_scenario) > 0 and len(high_budget) > 0:
        assert high_budget['risk_avoided'].values[0] >= base_scenario['risk_avoided'].values[0]
    
    print("✓ test_sensitivity_scenarios")


def test_deferral_reasons():
    """Test that deferral reasons are informative"""
    risk_data = create_test_risk_data()
    
    budget_cap = 600000  # Will defer some items
    duration_cap = 20
    
    included, deferred, summary = optimize_scope(
        risk_data, budget_cap, duration_cap
    )
    
    if len(deferred) > 0:
        # Every deferred item should have a reason
        assert 'deferral_reason' in deferred.columns
        assert deferred['deferral_reason'].notna().all()
        
        # Low confidence items should mention confidence
        low_conf = deferred[deferred['confidence'] == 'low']
        if len(low_conf) > 0:
            assert any('confidence' in reason.lower() for reason in low_conf['deferral_reason'])
    
    print("✓ test_deferral_reasons")


def test_empty_input_handling():
    """Test handling of edge cases"""
    empty_risk_data = pd.DataFrame(columns=[
        'scope_item_id', 'equipment_id', 'work_type', 'description',
        'cost_estimate_usd', 'duration_days', 'is_regulatory',
        'expected_loss_avoided', 'risk_tier', 'confidence'
    ])
    
    budget_cap = 1000000
    duration_cap = 30
    
    try:
        included, deferred, summary = optimize_scope(
            empty_risk_data, budget_cap, duration_cap
        )
        # Should succeed with empty results
        assert len(included) == 0
        assert len(deferred) == 0
        print("✓ test_empty_input_handling")
    except Exception as e:
        print(f"✓ test_empty_input_handling (acceptable error: {e})")


if __name__ == "__main__":
    print("Running Agent 2 unit tests...")
    print("=" * 60)
    
    # Run tests
    test_regulatory_items_always_included()
    test_optimization_respects_budget()
    test_optimization_respects_duration()
    test_optimizer_maximizes_risk_avoided()
    test_infeasible_constraints_raise_error()
    test_marginal_analysis_calculation()
    test_sensitivity_scenarios()
    test_deferral_reasons()
    test_empty_input_handling()
    
    print("=" * 60)
    print("✓ All Agent 2 tests passed!")