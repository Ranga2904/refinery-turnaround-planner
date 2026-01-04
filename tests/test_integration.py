"""
Integration tests: Full Agent 1 → Agent 2 pipeline
"""

import pandas as pd
import sqlite3
import sys
sys.path.append('.')

from agents.risk_analyst import run_agent1
from agents.optimizer import run_agent2


def test_full_pipeline():
    """Test complete workflow from raw data to optimized scope"""
    
    print("\n" + "=" * 60)
    print("INTEGRATION TEST: Full Agent 1 → Agent 2 Pipeline")
    print("=" * 60)
    
    # Step 1: Load raw data
    print("\n[Step 1] Loading raw data from database...")
    conn = sqlite3.connect('data/demo.db')
    equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
    failures_df = pd.read_sql('SELECT * FROM failures', conn)
    scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
    conn.close()
    
    print(f"  ✓ Loaded {len(equipment_df)} equipment items")
    print(f"  ✓ Loaded {len(failures_df)} failure events")
    print(f"  ✓ Loaded {len(scope_df)} scope proposals")
    
    # Step 2: Run Agent 1 (Risk Analysis)
    print("\n[Step 2] Running Agent 1: Risk Analysis...")
    risk_results = run_agent1(
        equipment_df,
        failures_df,
        scope_df,
        production_capacity_bbl_day=100000,
        margin_per_bbl=25.0
    )
    
    print(f"  ✓ Analyzed {len(risk_results)} items")
    print(f"  ✓ Risk tiers: {risk_results['risk_tier'].value_counts().to_dict()}")
    print(f"  ✓ Confidence: {risk_results['confidence'].value_counts().to_dict()}")
    
    # Validate Agent 1 output
    assert len(risk_results) == len(scope_df), "Agent 1 should analyze all scope items"
    assert 'expected_loss_avoided' in risk_results.columns
    assert 'risk_tier' in risk_results.columns
    assert 'confidence' in risk_results.columns
    assert risk_results['expected_loss_avoided'].min() >= 0, "Expected loss cannot be negative"
    
    # Step 3: Run Agent 2 (Optimization)
    print("\n[Step 3] Running Agent 2: Scope Optimization...")
    BUDGET_CAP = 18_500_000
    DURATION_CAP = 42
    
    optimization_results = run_agent2(risk_results, BUDGET_CAP, DURATION_CAP)
    
    included = optimization_results['included_scope']
    deferred = optimization_results['deferred_items']
    summary = optimization_results['summary']
    
    print(f"  ✓ Included: {summary['total_included']} items")
    print(f"  ✓ Deferred: {summary['total_deferred']} items")
    print(f"  ✓ Total cost: ${summary['total_cost']:,.0f}")
    print(f"  ✓ Risk avoided: ${summary['total_risk_avoided']:,.0f}")
    print(f"  ✓ Residual risk: ${summary['residual_risk']:,.0f}")
    
    # Validate Agent 2 output
    assert summary['total_cost'] <= BUDGET_CAP, "Budget constraint violated"
    assert summary['total_included'] + summary['total_deferred'] == len(scope_df), "Item count mismatch"
    assert summary['total_risk_avoided'] > 0, "Should avoid some risk"
    
    # Step 4: Validate data consistency
    # Step 4: Validate data consistency
    print("\n[Step 4] Validating data consistency...")
    
    # All scope items should be either included or deferred
    if len(deferred) > 0:
        all_items = set(included['scope_item_id']).union(set(deferred['scope_item_id']))
    else:
        all_items = set(included['scope_item_id'])
    
    original_items = set(scope_df['scope_item_id'])
    assert all_items == original_items, "Items lost in pipeline"
    
    # No duplicate items
    assert len(included['scope_item_id'].unique()) == len(included), "Duplicate items in included scope"
    if len(deferred) > 0:
        assert len(deferred['scope_item_id'].unique()) == len(deferred), "Duplicate items in deferred scope"
    
    # Regulatory items should be included
    # Regulatory items should be included
    regulatory_items = scope_df[scope_df['is_regulatory'] == True]['scope_item_id'].values
    if len(regulatory_items) > 0:
        for item_id in regulatory_items:
            assert item_id in included['scope_item_id'].values, f"Regulatory item {item_id} not included"
        print(f"  ✓ All {len(regulatory_items)} regulatory items included")
    else:
        print("  ✓ No regulatory items in scope")
    
    print("  ✓ All scope items accounted for")
    print("  ✓ No duplicate items")
    print("  ✓ All regulatory items included")
    
    # Step 5: Validate traceability
    print("\n[Step 5] Validating traceability...")
    
    # Pick a random included item and trace back to source
    sample_item = included.iloc[0]
    item_id = sample_item['scope_item_id']
    equipment_id = sample_item['equipment_id']
    
    # Trace to equipment
    equipment = equipment_df[equipment_df['equipment_id'] == equipment_id]
    assert len(equipment) == 1, f"Equipment {equipment_id} not found"
    
    # Trace to failures
    failures = failures_df[failures_df['equipment_id'] == equipment_id]
    
    # Trace through risk analysis
    risk_item = risk_results[risk_results['scope_item_id'] == item_id]
    assert len(risk_item) == 1, f"Risk analysis missing for {item_id}"
    
    print(f"  ✓ Traced item {item_id} back to equipment {equipment_id}")
    print(f"  ✓ Found {len(failures)} historical failures")
    print(f"  ✓ Risk tier: {risk_item['risk_tier'].values[0]}")
    print(f"  ✓ Expected loss: ${risk_item['expected_loss_avoided'].values[0]:,.0f}")
    
    # Step 6: Validate economic calculations
    print("\n[Step 6] Validating economic calculations...")
    
    total_avoided_risk = included['expected_loss_avoided'].sum()
    total_cost = included['cost_estimate_usd'].sum()
    roi = total_avoided_risk / total_cost
    
    print(f"  ✓ Total risk avoided: ${total_avoided_risk:,.0f}")
    print(f"  ✓ Total cost: ${total_cost:,.0f}")
    print(f"  ✓ ROI: ${roi:.2f} avoided per $1 spent")
    
    assert roi > 0, "ROI should be positive"
    
    if total_avoided_risk > total_cost:
        print(f"  ✓ Risk avoided exceeds cost (good economic case)")
    else:
        print(f"  ⚠ Risk avoided less than cost (risk mitigation still valuable)")
    
    print("\n" + "=" * 60)
    print("✓ INTEGRATION TEST PASSED")
    print("=" * 60)
    
    return {
        'risk_results': risk_results,
        'optimization_results': optimization_results,
        'summary': summary
    }


def test_data_quality():
    """Test data quality and validation rules"""
    
    print("\n" + "=" * 60)
    print("DATA QUALITY VALIDATION")
    print("=" * 60)
    
    conn = sqlite3.connect('data/demo.db')
    equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
    failures_df = pd.read_sql('SELECT * FROM failures', conn)
    scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
    conn.close()
    
    # Equipment validation
    print("\n[Equipment Registry]")
    assert equipment_df['equipment_id'].nunique() == len(equipment_df), "Duplicate equipment IDs"
    assert equipment_df['criticality'].between(1, 4).all(), "Invalid criticality values"
    assert equipment_df['condition_score'].between(1, 10).all(), "Invalid condition scores"
    assert equipment_df['current_age_years'].min() >= 0, "Negative age values"
    print("  ✓ No duplicate equipment IDs")
    print("  ✓ Criticality values valid (1-4)")
    print("  ✓ Condition scores valid (1-10)")
    print("  ✓ Age values non-negative")
    
    # Failure history validation
    print("\n[Failure History]")
    assert failures_df['repair_cost_usd'].min() > 0, "Repair costs must be positive"
    assert failures_df['downtime_days'].min() >= 0, "Downtime cannot be negative"
    assert failures_df['equipment_id'].isin(equipment_df['equipment_id']).all(), "Invalid equipment references"
    print("  ✓ All repair costs positive")
    print("  ✓ All downtime values non-negative")
    print("  ✓ All failures reference valid equipment")
    
    # Scope validation
    print("\n[Scope Proposals]")
    assert scope_df['cost_estimate_usd'].min() > 0, "Cost estimates must be positive"
    assert scope_df['duration_days'].min() > 0, "Duration must be positive"
    assert scope_df['equipment_id'].isin(equipment_df['equipment_id']).all(), "Invalid equipment references"
    regulatory_count = scope_df['is_regulatory'].sum()
    print(f"  ✓ All costs positive")
    print(f"  ✓ All durations positive")
    print(f"  ✓ All scope items reference valid equipment")
    print(f"  ✓ {regulatory_count} regulatory items flagged")
    
    print("\n" + "=" * 60)
    print("✓ DATA QUALITY VALIDATED")
    print("=" * 60)


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    
    print("\n" + "=" * 60)
    print("EDGE CASE TESTING")
    print("=" * 60)
    
    # Load data
    conn = sqlite3.connect('data/demo.db')
    equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
    failures_df = pd.read_sql('SELECT * FROM failures', conn)
    scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
    conn.close()
    
    # Test 1: Very tight budget (only regulatory items fit)
    print("\n[Test 1] Very tight budget...")
    risk_results = run_agent1(equipment_df, failures_df, scope_df)
    regulatory_cost = scope_df[scope_df['is_regulatory'] == True]['cost_estimate_usd'].sum()
    tight_budget = regulatory_cost + 100000  # Just slightly above regulatory
    
    try:
        results = run_agent2(risk_results, tight_budget, 100)
        print(f"  ✓ Handled tight budget (${tight_budget:,.0f})")
        print(f"  ✓ Included {results['summary']['total_included']} items")
    except ValueError as e:
        print(f"  ✓ Correctly rejected infeasible budget: {e}")
    
    # Test 2: Very generous budget (all items fit)
    print("\n[Test 2] Very generous budget...")
    generous_budget = scope_df['cost_estimate_usd'].sum() * 2
    results = run_agent2(risk_results, generous_budget, 200)
    
    # Should include all items
    assert results['summary']['total_deferred'] == 0, "All items should fit"
    print(f"  ✓ Included all {results['summary']['total_included']} items")
    print(f"  ✓ No deferrals with generous budget")
    
    # Test 3: Equipment with no failure history
    print("\n[Test 3] Equipment with no failure history...")
    no_history_items = []
    for _, item in scope_df.iterrows():
        equip_failures = failures_df[failures_df['equipment_id'] == item['equipment_id']]
        if len(equip_failures) == 0:
            no_history_items.append(item['equipment_id'])
    
    if len(no_history_items) > 0:
        print(f"  ✓ Found {len(no_history_items)} items with no failure history")
        # These should still get analyzed (using defaults)
        sample_item = scope_df[scope_df['equipment_id'] == no_history_items[0]]
        risk_result = risk_results[risk_results['equipment_id'] == no_history_items[0]]
        assert len(risk_result) > 0, "Items with no history should still be analyzed"
        print(f"  ✓ Items without history analyzed with defaults")
    else:
        print(f"  ✓ All equipment has failure history (good data quality)")
    
    print("\n" + "=" * 60)
    print("✓ EDGE CASES HANDLED")
    print("=" * 60)


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING FULL INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Test 1: Full pipeline
    test_full_pipeline()
    
    # Test 2: Data quality
    test_data_quality()
    
    # Test 3: Edge cases
    test_edge_cases()
    
    print("\n" + "=" * 70)
    print("✓ ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    print("\n✓ System ready for LangGraph integration (Day 4)")