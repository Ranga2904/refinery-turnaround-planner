"""
Unit tests for Agent 1: Risk Analyst
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('.')

from agents.risk_analyst import (
    calculate_failure_probability,
    calculate_consequence,
    calculate_expected_loss,
    assign_risk_tier
)


def test_failure_probability_no_history():
    """Test probability calculation with no failure history"""
    equipment_data = pd.DataFrame([{
        'equipment_id': 'TEST-001',
        'install_date': '2010-01-01',
        'current_age_years': 15,
        'design_life_years': 25,
        'condition_score': 7,
        'criticality': 2
    }])
    
    failure_history = failure_history = pd.DataFrame(columns=['equipment_id', 'failure_date'])
    
    prob, confidence, source = calculate_failure_probability(
        'TEST-001', equipment_data, failure_history
    )
    
    assert prob == 0.05  # Industry default
    assert confidence == 'low'
    assert 'no_history' in source


def test_failure_probability_with_history():
    """Test probability calculation with failure history"""
    equipment_data = pd.DataFrame([{
        'equipment_id': 'TEST-002',
        'install_date': '2010-01-01',
        'current_age_years': 15,
        'design_life_years': 25,
        'condition_score': 5,
        'criticality': 1
    }])
    
    # Create failure history (3 failures, ~5 years apart)
    failure_history = pd.DataFrame([
        {'equipment_id': 'TEST-002', 'failure_date': '2015-01-01'},
        {'equipment_id': 'TEST-002', 'failure_date': '2019-06-01'},
        {'equipment_id': 'TEST-002', 'failure_date': '2024-01-01'}
    ])
    
    prob, confidence, source = calculate_failure_probability(
        'TEST-002', equipment_data, failure_history
    )
    
    assert 0 < prob < 1  # Valid probability
    assert confidence == 'medium'  # 3 events
    assert 'mtbf' in source


def test_consequence_calculation():
    """Test consequence calculation"""
    equipment_data = pd.DataFrame([{
        'equipment_id': 'TEST-003',
        'criticality': 1,
        'current_age_years': 10,
        'design_life_years': 25,
        'condition_score': 8
    }])
    
    failure_history = pd.DataFrame([
        {'equipment_id': 'TEST-003', 'repair_cost_usd': 200000, 'downtime_days': 5.0},
        {'equipment_id': 'TEST-003', 'repair_cost_usd': 300000, 'downtime_days': 4.0}
    ])
    
    consequence, downtime, source = calculate_consequence(
        'TEST-003', equipment_data, failure_history,
        production_capacity_bbl_day=100000,
        margin_per_bbl=25.0
    )
    
    # Should use historical average
    expected_repair = 250000  # Average of 200k and 300k
    expected_downtime = 4.5  # Average of 5 and 4
    # Criticality 1 = 100% impact: 100k bbl/day * $25/bbl * 4.5 days = $11.25M
    expected_production_loss = 100000 * 25 * 4.5
    
    assert consequence > expected_repair  # Includes production loss
    assert downtime == 4.5
    assert 'historical' in source


def test_expected_loss():
    """Test expected loss calculation"""
    prob = 0.25  # 25% chance
    consequence = 1_000_000  # $1M consequence
    
    expected_loss = calculate_expected_loss(prob, consequence)
    
    assert expected_loss == 250_000  # 25% of $1M


def test_risk_tier_assignment():
    """Test risk tier categorization"""
    assert assign_risk_tier(10_000_000) == 'critical'
    assert assign_risk_tier(5_000_000) == 'critical'
    assert assign_risk_tier(2_000_000) == 'high'
    assert assign_risk_tier(500_000) == 'medium'
    assert assign_risk_tier(50_000) == 'low'


def test_confidence_degrades_with_old_data():
    """Test that confidence decreases with old data"""
    equipment_data = pd.DataFrame([{
        'equipment_id': 'TEST-004',
        'install_date': '2000-01-01',
        'current_age_years': 25,
        'design_life_years': 25,
        'condition_score': 6,
        'criticality': 2
    }])
    
    # Old failure data (>3 years ago)
    old_date = (datetime.now() - timedelta(days=1200)).strftime('%Y-%m-%d')
    failure_history = pd.DataFrame([
        {'equipment_id': 'TEST-004', 'failure_date': old_date},
        {'equipment_id': 'TEST-004', 'failure_date': old_date}
    ])
    
    prob, confidence, source = calculate_failure_probability(
        'TEST-004', equipment_data, failure_history
    )
    
    # Confidence should degrade due to old data
    assert confidence in ['low', 'medium']


if __name__ == "__main__":
    print("Running Agent 1 unit tests...")
    print("=" * 60)
    
    # Run tests
    test_failure_probability_no_history()
    print("✓ test_failure_probability_no_history")
    
    test_failure_probability_with_history()
    print("✓ test_failure_probability_with_history")
    
    test_consequence_calculation()
    print("✓ test_consequence_calculation")
    
    test_expected_loss()
    print("✓ test_expected_loss")
    
    test_risk_tier_assignment()
    print("✓ test_risk_tier_assignment")
    
    test_confidence_degrades_with_old_data()
    print("✓ test_confidence_degrades_with_old_data")
    
    print("=" * 60)
    print("✓ All tests passed!") 
