 
"""
Data validation utilities using Pydantic for schema enforcement
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Literal, Optional
from datetime import datetime
import pandas as pd


class EquipmentSchema(BaseModel):
    """Schema for equipment registry"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    equipment_id: str = Field(min_length=1, max_length=50)
    system: str = Field(min_length=1)
    equipment_type: str = Field(min_length=1)
    criticality: Literal[1, 2, 3, 4]
    install_date: str  # Will validate format
    last_inspection_date: str
    condition_score: int = Field(ge=1, le=10)
    design_life_years: int = Field(gt=0, le=100)
    current_age_years: int = Field(ge=0, le=100)
    
    @field_validator('install_date', 'last_inspection_date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")
    
    @field_validator('current_age_years')
    @classmethod
    def validate_age_vs_design_life(cls, v, info):
        if 'design_life_years' in info.data and v > info.data['design_life_years'] * 2:
            raise ValueError(f"Age {v} exceeds 2x design life - check data")
        return v


class FailureSchema(BaseModel):
    """Schema for failure history"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    failure_id: int = Field(gt=0)
    equipment_id: str = Field(min_length=1, max_length=50)
    failure_date: str
    failure_mode: str = Field(min_length=1)
    repair_cost_usd: int = Field(gt=0, le=10_000_000)
    downtime_days: float = Field(ge=0, le=365)
    root_cause: str = Field(min_length=1)
    
    @field_validator('failure_date')
    @classmethod
    def validate_date_format(cls, v):
        try:
            dt = datetime.strptime(v, '%Y-%m-%d')
            # Check not in future
            if dt > datetime.now():
                raise ValueError(f"Failure date cannot be in future: {v}")
            return v
        except ValueError as e:
            if "future" in str(e):
                raise e
            raise ValueError(f"Date must be in YYYY-MM-DD format, got: {v}")


class ScopeProposalSchema(BaseModel):
    """Schema for proposed scope items"""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    scope_item_id: str = Field(min_length=1, max_length=50)
    equipment_id: str = Field(min_length=1, max_length=50)
    work_type: str = Field(min_length=1)
    description: str = Field(min_length=10, max_length=500)
    cost_estimate_usd: int = Field(gt=0, le=50_000_000)
    duration_days: float = Field(gt=0, le=365)
    is_regulatory: bool
    
    @field_validator('work_type')
    @classmethod
    def validate_work_type(cls, v):
        valid_types = ['inspection', 'repair', 'replacement', 'upgrade', 'maintenance']
        if v not in valid_types:
            raise ValueError(f"work_type must be one of {valid_types}, got: {v}")
        return v


class ConstraintsSchema(BaseModel):
    """Schema for optimization constraints"""
    budget_cap: float = Field(gt=0, le=1_000_000_000)
    duration_cap: float = Field(gt=0, le=730)  # Max 2 years
    production_capacity_bbl_day: float = Field(gt=0, le=1_000_000)
    margin_per_bbl: float = Field(gt=0, le=200)
    
    @field_validator('budget_cap')
    @classmethod
    def validate_realistic_budget(cls, v):
        if v < 1_000_000:
            raise ValueError(f"Budget cap ${v:,.0f} seems too low for refinery TAR (min $1M)")
        return v


def validate_dataframe(df: pd.DataFrame, schema: type[BaseModel], name: str) -> dict:
    """
    Validate entire DataFrame against Pydantic schema.
    
    Returns:
        dict with 'valid': bool, 'errors': list, 'warnings': list
    """
    errors = []
    warnings = []
    valid_rows = 0
    
    for idx, row in df.iterrows():
        try:
            schema(**row.to_dict())
            valid_rows += 1
        except Exception as e:
            errors.append(f"Row {idx}: {str(e)}")
    
    # Check for duplicates
    # Check for duplicate PRIMARY keys only
    # Equipment: equipment_id should be unique
    # Failures: failure_id should be unique (equipment_id can repeat)
    # Scope: scope_item_id should be unique (equipment_id can repeat)
    
    if name == "Equipment Registry" and 'equipment_id' in df.columns:
        duplicates = df[df.duplicated('equipment_id', keep=False)]
        if len(duplicates) > 0:
            errors.append(f"Duplicate equipment_ids: {duplicates['equipment_id'].unique().tolist()}")
    
    if name == "Failure History" and 'failure_id' in df.columns:
        duplicates = df[df.duplicated('failure_id', keep=False)]
        if len(duplicates) > 0:
            errors.append(f"Duplicate failure_ids: {duplicates['failure_id'].tolist()}")
    
    if name == "Scope Proposals" and 'scope_item_id' in df.columns:
        duplicates = df[df.duplicated('scope_item_id', keep=False)]
        if len(duplicates) > 0:
            errors.append(f"Duplicate scope_item_ids: {duplicates['scope_item_id'].unique().tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        for col, count in missing_cols.items():
            warnings.append(f"Column '{col}' has {count} missing values")
    
    return {
        'dataset': name,
        'valid': len(errors) == 0,
        'total_rows': len(df),
        'valid_rows': valid_rows,
        'errors': errors,
        'warnings': warnings
    }


def validate_referential_integrity(
    equipment_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    scope_df: pd.DataFrame
) -> dict:
    """
    Validate foreign key relationships between datasets.
    """
    errors = []
    
    # Check failures reference valid equipment
    invalid_failures = ~failures_df['equipment_id'].isin(equipment_df['equipment_id'])
    if invalid_failures.any():
        invalid_ids = failures_df[invalid_failures]['equipment_id'].unique()
        errors.append(f"Failures reference non-existent equipment: {invalid_ids.tolist()}")
    
    # Check scope references valid equipment
    invalid_scope = ~scope_df['equipment_id'].isin(equipment_df['equipment_id'])
    if invalid_scope.any():
        invalid_ids = scope_df[invalid_scope]['equipment_id'].unique()
        errors.append(f"Scope items reference non-existent equipment: {invalid_ids.tolist()}")
    
    # Check for orphaned equipment (no scope items)
    orphaned = ~equipment_df['equipment_id'].isin(scope_df['equipment_id'])
    orphaned_count = orphaned.sum()
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'orphaned_equipment': orphaned_count,
        'failure_coverage': (failures_df['equipment_id'].nunique() / len(equipment_df)) * 100,
        'scope_coverage': (scope_df['equipment_id'].nunique() / len(equipment_df)) * 100
    }


def validate_business_rules(
    equipment_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    constraints: dict
) -> dict:
    """
    Validate business logic and realistic constraints.
    """
    warnings = []
    errors = []
    
    # Check if total scope cost is realistic
    total_cost = scope_df['cost_estimate_usd'].sum()
    if total_cost < 5_000_000:
        warnings.append(f"Total scope cost ${total_cost:,.0f} is low for refinery TAR (typical: $15-30M)")
    elif total_cost > 100_000_000:
        warnings.append(f"Total scope cost ${total_cost:,.0f} is very high (typical: $15-30M)")
    
    # Check regulatory item ratio
    regulatory_count = scope_df['is_regulatory'].sum()
    regulatory_pct = (regulatory_count / len(scope_df)) * 100
    if regulatory_pct > 30:
        warnings.append(f"{regulatory_pct:.0f}% regulatory items is high (typical: 5-15%)")
    
    # Check critical equipment in scope
    critical_equipment = equipment_df[equipment_df['criticality'] == 1]['equipment_id']
    critical_in_scope = scope_df['equipment_id'].isin(critical_equipment).sum()
    critical_coverage = (critical_in_scope / len(critical_equipment)) * 100 if len(critical_equipment) > 0 else 0
    
    if critical_coverage < 50:
        warnings.append(f"Only {critical_coverage:.0f}% of critical equipment in scope (consider increasing)")
    
    # Check if constraints are feasible
    if 'budget_cap' in constraints:
        regulatory_cost = scope_df[scope_df['is_regulatory'] == True]['cost_estimate_usd'].sum()
        if regulatory_cost > constraints['budget_cap']:
            errors.append(f"Regulatory work (${regulatory_cost:,.0f}) exceeds budget cap (${constraints['budget_cap']:,.0f})")
    
    # Check for equipment with poor condition but not in scope
    poor_condition = equipment_df[
        (equipment_df['condition_score'] <= 3) & 
        (equipment_df['criticality'] <= 2)
    ]
    poor_not_in_scope = ~poor_condition['equipment_id'].isin(scope_df['equipment_id'])
    
    if poor_not_in_scope.any():
        count = poor_not_in_scope.sum()
        warnings.append(f"{count} critical items in poor condition not in scope")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'metrics': {
            'total_scope_cost': total_cost,
            'regulatory_pct': regulatory_pct,
            'critical_coverage_pct': critical_coverage
        }
    }


def run_full_validation(
    equipment_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    scope_df: pd.DataFrame,
    constraints: dict
) -> dict:
    """
    Run all validation checks and return comprehensive report.
    """
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)
    
    # Schema validation
    print("\n[1] Schema Validation")
    equipment_validation = validate_dataframe(equipment_df, EquipmentSchema, "Equipment Registry")
    failures_validation = validate_dataframe(failures_df, FailureSchema, "Failure History")
    scope_validation = validate_dataframe(scope_df, ScopeProposalSchema, "Scope Proposals")
    
    for validation in [equipment_validation, failures_validation, scope_validation]:
        status = "✓" if validation['valid'] else "✗"
        print(f"  {status} {validation['dataset']}: {validation['valid_rows']}/{validation['total_rows']} valid")
        if validation['errors']:
            for error in validation['errors'][:3]:  # Show first 3 errors
                print(f"      ERROR: {error}")
        if validation['warnings']:
            for warning in validation['warnings'][:2]:
                print(f"      WARNING: {warning}")
    
    # Referential integrity
    print("\n[2] Referential Integrity")
    ref_validation = validate_referential_integrity(equipment_df, failures_df, scope_df)
    status = "✓" if ref_validation['valid'] else "✗"
    print(f"  {status} Foreign key relationships")
    if ref_validation['errors']:
        for error in ref_validation['errors']:
            print(f"      ERROR: {error}")
    print(f"  • Equipment with failures: {ref_validation['failure_coverage']:.0f}%")
    print(f"  • Equipment in scope: {ref_validation['scope_coverage']:.0f}%")
    print(f"  • Orphaned equipment: {ref_validation['orphaned_equipment']}")
    
    # Business rules
    print("\n[3] Business Rules")
    business_validation = validate_business_rules(equipment_df, failures_df, scope_df, constraints)
    status = "✓" if business_validation['valid'] else "✗"
    print(f"  {status} Business logic")
    if business_validation['errors']:
        for error in business_validation['errors']:
            print(f"      ERROR: {error}")
    if business_validation['warnings']:
        for warning in business_validation['warnings']:
            print(f"      WARNING: {warning}")
    
    # Overall status
    print("\n" + "=" * 60)
    all_valid = (
        equipment_validation['valid'] and
        failures_validation['valid'] and
        scope_validation['valid'] and
        ref_validation['valid'] and
        business_validation['valid']
    )
    
    if all_valid:
        print("✓ ALL VALIDATIONS PASSED")
    else:
        print("✗ VALIDATION FAILED - Review errors above")
    print("=" * 60)
    
    return {
        'overall_valid': all_valid,
        'schema_validation': {
            'equipment': equipment_validation,
            'failures': failures_validation,
            'scope': scope_validation
        },
        'referential_integrity': ref_validation,
        'business_rules': business_validation
    }


if __name__ == "__main__":
    """Test validation with demo data"""
    import sqlite3
    
    # Load data
    conn = sqlite3.connect('data/demo.db')
    equipment_df = pd.read_sql('SELECT * FROM equipment', conn)
    failures_df = pd.read_sql('SELECT * FROM failures', conn)
    scope_df = pd.read_sql('SELECT * FROM scope_proposals', conn)
    conn.close()
    
    # Define constraints
    constraints = {
        'budget_cap': 18_500_000,
        'duration_cap': 42,
        'production_capacity_bbl_day': 100000,
        'margin_per_bbl': 25.0
    }
    
    # Run validation
    report = run_full_validation(equipment_df, failures_df, scope_df, constraints)
    
    # Save report
    import json
    with open('data/validation_report.json', 'w') as f:
        # Convert to JSON-serializable format
        json_report = {
            'timestamp': datetime.now().isoformat(),
            'overall_valid': report['overall_valid'],
            'summary': {
                'equipment_valid': report['schema_validation']['equipment']['valid'],
                'failures_valid': report['schema_validation']['failures']['valid'],
                'scope_valid': report['schema_validation']['scope']['valid'],
                'referential_integrity_valid': report['referential_integrity']['valid'],
                'business_rules_valid': report['business_rules']['valid']
            }
        }
        json.dump(json_report, f, indent=2)
    
    print("\n✓ Validation report saved to data/validation_report.json")