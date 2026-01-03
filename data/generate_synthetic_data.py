import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_equipment_registry(n_items=50):
    """
    Generate equipment registry with realistic age and criticality distributions.
    
    Real refineries have:
    - 5-10% Criticality 1 (mission-critical, single point of failure)
    - 15-20% Criticality 2 (high impact)
    - 50-60% Criticality 3 (moderate impact)
    - 20-30% Criticality 4 (low impact)
    """
    
    equipment = []
    
    # Equipment systems in a typical refinery
    systems = [
        'crude_unit', 'vacuum_unit', 'reformer', 'hydrotreater',
        'fluid_catalytic_cracker', 'alkylation', 'isomerization',
        'utilities', 'tank_farm', 'loading_rack'
    ]
    
    # Equipment types by system
    equipment_types = {
        'crude_unit': ['furnace', 'heat_exchanger', 'column', 'pump', 'compressor'],
        'vacuum_unit': ['furnace', 'column', 'ejector', 'pump'],
        'reformer': ['reactor', 'furnace', 'separator', 'compressor', 'heat_exchanger'],
        'hydrotreater': ['reactor', 'furnace', 'separator', 'pump', 'compressor'],
        'fluid_catalytic_cracker': ['reactor', 'regenerator', 'main_column', 'compressor'],
        'alkylation': ['reactor', 'settler', 'pump', 'heat_exchanger'],
        'isomerization': ['reactor', 'separator', 'pump'],
        'utilities': ['boiler', 'cooling_tower', 'pump', 'air_compressor'],
        'tank_farm': ['storage_tank', 'pump', 'valve'],
        'loading_rack': ['loading_arm', 'pump', 'meter']
    }
    
    for i in range(n_items):
        system = random.choice(systems)
        equip_type = random.choice(equipment_types[system])
        
        # Age follows Weibull distribution (realistic for refineries)
        # Mean age ~15 years, shape parameter 2
        age_years = int(np.random.weibull(2) * 10 + 5)
        age_years = min(age_years, 40)  # Cap at 40 years
        
        install_date = datetime.now() - timedelta(days=age_years * 365)
        
        # Last inspection within last 1-3 years
        days_since_inspection = random.randint(30, 1095)
        last_inspection = datetime.now() - timedelta(days=days_since_inspection)
        
        # Criticality follows realistic distribution
        criticality = random.choices(
            [1, 2, 3, 4],
            weights=[7, 18, 50, 25]  # Realistic distribution
        )[0]
        
        # Condition score (1=poor, 10=excellent)
        # Older equipment tends to have lower scores
        base_condition = 10 - (age_years / 40) * 8  # Age-based degradation
        noise = np.random.normal(0, 1.5)
        condition_score = int(np.clip(base_condition + noise, 1, 10))
        
        equipment.append({
            'equipment_id': f'{system[:3].upper()}-{equip_type[:3].upper()}-{i:03d}',
            'system': system,
            'equipment_type': equip_type,
            'criticality': criticality,
            'install_date': install_date.strftime('%Y-%m-%d'),
            'last_inspection_date': last_inspection.strftime('%Y-%m-%d'),
            'condition_score': condition_score,
            'design_life_years': 25,
            'current_age_years': age_years
        })
    
    df = pd.DataFrame(equipment)
    
    # Sort by criticality then age
    df = df.sort_values(['criticality', 'current_age_years'], ascending=[True, False])
    df = df.reset_index(drop=True)
    
    print(f"✓ Generated {len(df)} equipment items")
    print(f"  Criticality distribution: {df['criticality'].value_counts().sort_index().to_dict()}")
    print(f"  Age range: {df['current_age_years'].min()}-{df['current_age_years'].max()} years")
    print(f"  Condition range: {df['condition_score'].min()}-{df['condition_score'].max()}")
    
    return df

def generate_failure_history(equipment_df, n_events=200):
    """
    Generate failure events based on equipment criticality and age.
    
    Realistic patterns:
    - Critical equipment fails more often (monitored better, stressed more)
    - Older equipment fails more frequently
    - Repair costs correlate with equipment type and criticality
    - Downtime varies by failure severity
    """
    
    failures = []
    
    # Failure modes by equipment type
    failure_modes = {
        'furnace': ['tube_leak', 'burner_failure', 'refractory_damage', 'control_valve_stuck'],
        'heat_exchanger': ['tube_leak', 'fouling', 'gasket_failure', 'baffle_damage'],
        'column': ['tray_damage', 'packing_collapse', 'internals_failure', 'shell_corrosion'],
        'pump': ['seal_failure', 'bearing_failure', 'impeller_damage', 'shaft_failure'],
        'compressor': ['seal_failure', 'bearing_failure', 'valve_failure', 'rotor_imbalance'],
        'reactor': ['catalyst_deactivation', 'distributor_plugging', 'shell_corrosion', 'internals_failure'],
        'separator': ['level_control_failure', 'internals_damage', 'vessel_corrosion'],
        'valve': ['seat_erosion', 'stem_binding', 'actuator_failure'],
        'tank': ['shell_corrosion', 'roof_damage', 'foundation_settlement'],
        'default': ['mechanical_failure', 'corrosion', 'fatigue', 'erosion']
    }
    
    for _, equip in equipment_df.iterrows():
        # Number of failures based on criticality and age
        # Critical equipment: 3-6 failures over lifetime
        # Non-critical: 1-3 failures
        base_failures = {1: 5, 2: 4, 3: 2, 4: 1}[equip['criticality']]
        
        # Age factor: older equipment has more failures
        age_factor = equip['current_age_years'] / 25  # Normalize by design life
        n_failures = int(base_failures * age_factor * np.random.uniform(0.7, 1.3))
        n_failures = max(1, n_failures)  # At least 1 failure for history
        
        # Generate failures spread over equipment lifetime
        equipment_age_days = equip['current_age_years'] * 365
        
        for _ in range(n_failures):
            # Failure date somewhere in equipment's past
            days_ago = random.randint(180, equipment_age_days)  # At least 6 months ago
            failure_date = datetime.now() - timedelta(days=days_ago)
            
            # Failure mode
            equip_type = equip['equipment_type']
            modes = failure_modes.get(equip_type, failure_modes['default'])
            failure_mode = random.choice(modes)
            
            # Repair cost based on criticality and equipment type
            # Critical equipment: $50K-$500K
            # High: $20K-$200K
            # Medium: $5K-$50K
            # Low: $2K-$20K
            cost_ranges = {
                1: (50000, 500000),
                2: (20000, 200000),
                3: (5000, 50000),
                4: (2000, 20000)
            }
            min_cost, max_cost = cost_ranges[equip['criticality']]
            repair_cost = int(np.random.lognormal(np.log(min_cost + max_cost) / 2, 0.5))
            repair_cost = np.clip(repair_cost, min_cost, max_cost)
            
            # Downtime (days) - critical failures cause longer outages
            # Criticality 1: 2-7 days
            # Criticality 2: 1-4 days
            # Criticality 3: 0.5-2 days
            # Criticality 4: 0.25-1 day
            downtime_ranges = {
                1: (2, 7),
                2: (1, 4),
                3: (0.5, 2),
                4: (0.25, 1)
            }
            min_dt, max_dt = downtime_ranges[equip['criticality']]
            downtime_days = round(np.random.uniform(min_dt, max_dt), 2)
            
            failures.append({
                'failure_id': len(failures) + 1,
                'equipment_id': equip['equipment_id'],
                'failure_date': failure_date.strftime('%Y-%m-%d'),
                'failure_mode': failure_mode,
                'repair_cost_usd': int(repair_cost),
                'downtime_days': downtime_days,
                'root_cause': random.choice(['wear', 'corrosion', 'fatigue', 'design_issue', 'operational_error'])
            })
    
    df = pd.DataFrame(failures)
    df = df.sort_values('failure_date').reset_index(drop=True)
    
    print(f"\n✓ Generated {len(df)} failure events")
    print(f"  Date range: {df['failure_date'].min()} to {df['failure_date'].max()}")
    print(f"  Total repair costs: ${df['repair_cost_usd'].sum():,.0f}")
    print(f"  Average downtime: {df['downtime_days'].mean():.1f} days")
    print(f"  Cost per event: ${df['repair_cost_usd'].mean():,.0f} (median: ${df['repair_cost_usd'].median():,.0f})")
    
    return df

def generate_proposed_scope(equipment_df, failures_df, n_items=30):
    """
    Generate proposed turnaround scope items.
    
    Realistic patterns:
    - Critical equipment more likely to be in scope
    - Equipment with poor condition scores prioritized
    - Equipment with recent failures included
    - Mix of preventive and corrective work
    """
    
    scope_items = []
    
    # Work types and typical costs/durations
    work_types = {
        'inspection': {
            'furnace': (80000, 150000, 2, 4),
            'heat_exchanger': (30000, 80000, 1, 2),
            'column': (100000, 250000, 3, 5),
            'pump': (15000, 40000, 0.5, 1),
            'compressor': (120000, 300000, 3, 6),
            'reactor': (200000, 500000, 4, 7),
            'default': (20000, 100000, 1, 3)
        },
        'replacement': {
            'furnace': (500000, 2000000, 5, 10),
            'heat_exchanger': (150000, 600000, 2, 5),
            'pump': (50000, 200000, 1, 3),
            'compressor': (400000, 1500000, 4, 8),
            'valve': (10000, 50000, 0.5, 1),
            'default': (100000, 500000, 2, 5)
        },
        'repair': {
            'furnace': (200000, 800000, 3, 6),
            'heat_exchanger': (80000, 300000, 2, 4),
            'column': (150000, 600000, 3, 7),
            'pump': (30000, 120000, 1, 2),
            'default': (50000, 250000, 1, 4)
        },
        'upgrade': {
            'default': (100000, 500000, 2, 5)
        }
    }
    
    # Select equipment for scope (biased toward critical and poor condition)
    equipment_df['score'] = (
        (5 - equipment_df['criticality']) * 3 +  # Lower criticality number = higher score
        (11 - equipment_df['condition_score']) * 2  # Lower condition = higher score
    )
    
    # Add randomness
    equipment_df['score'] = equipment_df['score'] + np.random.uniform(-2, 2, len(equipment_df))
    
    # Select top N equipment items
    selected_equipment = equipment_df.nlargest(n_items, 'score')
    
    for idx, equip in selected_equipment.iterrows():
        # Determine work type based on condition and failures
        recent_failures = failures_df[
            (failures_df['equipment_id'] == equip['equipment_id']) &
            (pd.to_datetime(failures_df['failure_date']) > datetime.now() - timedelta(days=730))
        ]
        
        if equip['condition_score'] <= 3:
            work_type = 'replacement'
        elif len(recent_failures) > 0:
            work_type = 'repair'
        elif equip['condition_score'] <= 6:
            work_type = random.choice(['repair', 'inspection'])
        else:
            work_type = random.choice(['inspection', 'upgrade'])
        
        # Get cost and duration ranges
        equip_type = equip['equipment_type']
        if equip_type in work_types[work_type]:
            min_cost, max_cost, min_dur, max_dur = work_types[work_type][equip_type]
        else:
            min_cost, max_cost, min_dur, max_dur = work_types[work_type]['default']
        
        # Generate specific values
        cost_estimate = int(np.random.uniform(min_cost, max_cost))
        duration_days = round(np.random.uniform(min_dur, max_dur), 1)
        
        # Regulatory flag (10% of items are mandatory)
        is_regulatory = random.random() < 0.10
        
        # Work description
        descriptions = {
            'inspection': f"{equip['equipment_type'].replace('_', ' ').title()} internal inspection per API standards",
            'replacement': f"Replace {equip['equipment_type'].replace('_', ' ')} due to end-of-life condition",
            'repair': f"Repair {equip['equipment_type'].replace('_', ' ')} - address identified deficiencies",
            'upgrade': f"Upgrade {equip['equipment_type'].replace('_', ' ')} controls and instrumentation"
        }
        
        scope_items.append({
            'scope_item_id': f"WO-{len(scope_items)+1:04d}",
            'equipment_id': equip['equipment_id'],
            'work_type': work_type,
            'description': descriptions[work_type],
            'cost_estimate_usd': cost_estimate,
            'duration_days': duration_days,
            'is_regulatory': is_regulatory,
            'priority_score': round(equip['score'], 2)
        })
    
    df = pd.DataFrame(scope_items)
    df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
    
    print(f"\n✓ Generated {len(df)} scope items")
    print(f"  Total estimated cost: ${df['cost_estimate_usd'].sum():,.0f}")
    print(f"  Total estimated duration: {df['duration_days'].sum():.1f} days")
    print(f"  Work type distribution: {df['work_type'].value_counts().to_dict()}")
    print(f"  Regulatory items: {df['is_regulatory'].sum()}")
    
    return df

def create_sqlite_database():
    """Create SQLite database with all tables"""
    
    db_path = 'data/demo.db'
    
    # Remove existing database
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Read CSVs
    equipment_df = pd.read_csv('data/demo_equipment.csv')
    failures_df = pd.read_csv('data/demo_failures.csv')
    scope_df = pd.read_csv('data/demo_scope.csv')
    
    # Write to database
    equipment_df.to_sql('equipment', conn, index=False, if_exists='replace')
    failures_df.to_sql('failures', conn, index=False, if_exists='replace')
    scope_df.to_sql('scope_proposals', conn, index=False, if_exists='replace')
    
    # Create indexes for performance
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX idx_equipment_criticality ON equipment(criticality)')
    cursor.execute('CREATE INDEX idx_failures_equipment ON failures(equipment_id)')
    cursor.execute('CREATE INDEX idx_failures_date ON failures(failure_date)')
    cursor.execute('CREATE INDEX idx_scope_equipment ON scope_proposals(equipment_id)')
    
    conn.commit()
    
    # Verify
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"\n✓ Created SQLite database: {db_path}")
    print(f"  Tables: {[t[0] for t in tables]}")
    
    for table in ['equipment', 'failures', 'scope_proposals']:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  • {table}: {count} rows")
    
    conn.close()
    
    return db_path

if __name__ == "__main__":
    print("=" * 60)
    print("GENERATING SYNTHETIC REFINERY TURNAROUND DATA")
    print("=" * 60)
    
    # Generate equipment registry
    print("\n[1/4] Equipment Registry")
    equipment_df = generate_equipment_registry(50)
    equipment_df.to_csv('data/demo_equipment.csv', index=False)
    print(f"✓ Saved to data/demo_equipment.csv")
    
    # Generate failure history
    print("\n[2/4] Failure History")
    failures_df = generate_failure_history(equipment_df)
    failures_df.to_csv('data/demo_failures.csv', index=False)
    print(f"✓ Saved to data/demo_failures.csv")
    
    # Generate proposed scope
    print("\n[3/4] Proposed Scope")
    scope_df = generate_proposed_scope(equipment_df, failures_df, n_items=30)
    scope_df.to_csv('data/demo_scope.csv', index=False)
    print(f"✓ Saved to data/demo_scope.csv")
    
    # Create SQLite database
    print("\n[4/4] SQLite Database")
    db_path = create_sqlite_database()
    
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  • data/demo_equipment.csv ({len(equipment_df)} rows)")
    print(f"  • data/demo_failures.csv ({len(failures_df)} rows)")
    print(f"  • data/demo_scope.csv ({len(scope_df)} rows)")
    print(f"  • data/demo.db (SQLite database)")
    print(f"\n✓ Ready for Agent 1 development (Day 2)")