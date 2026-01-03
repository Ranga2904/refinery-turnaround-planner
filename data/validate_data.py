"""
Validate synthetic data quality and distributions
"""

import pandas as pd
import sqlite3

def validate_data():
    """Run validation checks on synthetic data"""
    
    # Load from database
    conn = sqlite3.connect('data/demo.db')
    
    equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
    failures_df = pd.read_sql('SELECT * FROM failures', conn)
    scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
    
    conn.close()
    
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    # Equipment validations
    print("\n[EQUIPMENT REGISTRY]")
    print(f"✓ Total items: {len(equipment_df)}")
    print(f"✓ Criticality distribution:")
    for crit in [1, 2, 3, 4]:
        count = (equipment_df['criticality'] == crit).sum()
        pct = count / len(equipment_df) * 100
        print(f"    Criticality {crit}: {count:2d} ({pct:4.1f}%)")
    
    print(f"✓ Age range: {equipment_df['current_age_years'].min()}-{equipment_df['current_age_years'].max()} years")
    print(f"✓ Condition range: {equipment_df['condition_score'].min()}-{equipment_df['condition_score'].max()}")
    
    # Check for data quality issues
    assert equipment_df['equipment_id'].nunique() == len(equipment_df), "Duplicate equipment IDs!"
    assert equipment_df['criticality'].between(1, 4).all(), "Invalid criticality values!"
    assert equipment_df['condition_score'].between(1, 10).all(), "Invalid condition scores!"
    print("✓ No data quality issues detected")
    
    # Failure validations
    print("\n[FAILURE HISTORY]")
    print(f"✓ Total events: {len(failures_df)}")
    print(f"✓ Date range: {failures_df['failure_date'].min()} to {failures_df['failure_date'].max()}")
    print(f"✓ Total costs: ${failures_df['repair_cost_usd'].sum():,.0f}")
    print(f"✓ Cost distribution:")
    print(f"    Mean: ${failures_df['repair_cost_usd'].mean():,.0f}")
    print(f"    Median: ${failures_df['repair_cost_usd'].median():,.0f}")
    print(f"    Min: ${failures_df['repair_cost_usd'].min():,.0f}")
    print(f"    Max: ${failures_df['repair_cost_usd'].max():,.0f}")
    
    # Failures per equipment
    failures_per_equip = failures_df.groupby('equipment_id').size()
    print(f"✓ Failures per equipment: {failures_per_equip.min()}-{failures_per_equip.max()}")
    
    # Scope validations
    print("\n[PROPOSED SCOPE]")
    print(f"✓ Total items: {len(scope_df)}")
    print(f"✓ Total cost: ${scope_df['cost_estimate_usd'].sum():,.0f}")
    print(f"✓ Total duration: {scope_df['duration_days'].sum():.1f} days")
    print(f"✓ Work type breakdown:")
    for work_type, count in scope_df['work_type'].value_counts().items():
        print(f"    {work_type}: {count}")
    print(f"✓ Regulatory items: {scope_df['is_regulatory'].sum()}")
    
    # Check scope items reference valid equipment
    invalid_refs = ~scope_df['equipment_id'].isin(equipment_df['equipment_id'])
    assert not invalid_refs.any(), f"Invalid equipment references in scope!"
    print("✓ All scope items reference valid equipment")
    
# Realistic constraints check
    print("\n[REALISTIC CONSTRAINTS CHECK]")
    
    # Typical refinery TAR constraints
    typical_budget_range = (15_000_000, 30_000_000)
    typical_duration_range = (30, 60)
    
    total_scope_cost = scope_df['cost_estimate_usd'].sum()
    total_scope_duration = scope_df['duration_days'].sum()
    
    if typical_budget_range[0] <= total_scope_cost <= typical_budget_range[1]:
        print(f"✓ Total scope cost ${total_scope_cost:,.0f} is within typical range")
    else:
        print(f"⚠ Total scope cost ${total_scope_cost:,.0f} is outside typical range {typical_budget_range}")
    
    # Duration assumes some parallelization
    estimated_duration = total_scope_duration * 0.5  # Rough parallel factor
    if typical_duration_range[0] <= estimated_duration <= typical_duration_range[1]:
        print(f"✓ Estimated duration {estimated_duration:.0f} days is realistic")
    else:
        print(f"⚠ Estimated duration {estimated_duration:.0f} days may need adjustment")
    
    # Criticality bias check (critical equipment should be over-represented in scope)
    crit_in_registry = (equipment_df['criticality'] <= 2).sum() / len(equipment_df)
    crit_in_scope = scope_df.merge(equipment_df[['equipment_id', 'criticality']], on='equipment_id')
    crit_in_scope_pct = (crit_in_scope['criticality'] <= 2).sum() / len(crit_in_scope)
    
    print(f"✓ Critical equipment in registry: {crit_in_registry:.1%}")
    print(f"✓ Critical equipment in scope: {crit_in_scope_pct:.1%}")
    
    if crit_in_scope_pct > crit_in_registry:
        print("✓ Scope correctly prioritizes critical equipment")
    else:
        print("⚠ Scope may not prioritize critical equipment enough")
    
    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE - DATA IS READY")
    print("=" * 60)


if __name__ == "__main__":
    validate_data()