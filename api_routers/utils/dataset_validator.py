"""
Dataset Validation Module

This module provides comprehensive validation for custom uploaded datasets,
including schema validation, data type checking, and relationship validation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json
import os
from sdv.metadata import Metadata
from fastapi import HTTPException


class ValidationSeverity(Enum):
    """Validation error severity levels."""

    ERROR = "error"  # Blocks synthesis
    WARNING = "warning"  # Allows synthesis with warnings
    INFO = "info"  # Informational only


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the dataset."""

    severity: ValidationSeverity
    table_name: str
    column_name: Optional[str]
    issue_type: str
    message: str
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of dataset validation."""

    is_valid: bool
    issues: List[ValidationIssue]
    warnings: List[ValidationIssue]
    info: List[ValidationIssue]
    summary: Dict[str, Any]


class DatasetValidator:
    """Validates custom uploaded datasets against domain schemas."""

    def __init__(self):
        self.domain_schemas = self._load_domain_schemas()

    def _load_domain_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Load expected schemas for each domain/usecase combination."""
        schemas = {}

        # Pharmacohort schema
        schemas["pharmacohort"] = {
            "required_tables": ["person", "condition_era", "care_site"],
            "table_schemas": {
                "person": {
                    "required_columns": [
                        "person_id",
                        "gender_concept_id",
                        "year_of_birth",
                    ],
                    "column_types": {
                        "person_id": "int64",
                        "gender_concept_id": "int64",
                        "year_of_birth": "int64",
                        "month_of_birth": "int64",
                        "day_of_birth": "int64",
                        "birth_datetime": "float64",  # Can be NaN
                        "race_concept_id": "int64",
                        "ethnicity_concept_id": "int64",
                        "location_id": "int64",
                        "provider_id": "float64",  # Can be NaN
                        "care_site_id": "float64",  # Can be NaN
                        "person_source_value": "object",
                        "gender_source_value": "int64",
                        "gender_source_concept_id": "float64",  # Can be NaN
                        "race_source_value": "int64",
                        "race_source_concept_id": "float64",  # Can be NaN
                        "ethnicity_source_value": "int64",
                        "ethnicity_source_concept_id": "float64",  # Can be NaN
                    },
                    "primary_key": "person_id",
                    "foreign_keys": {},
                },
                "condition_era": {
                    "required_columns": [
                        "condition_era_id",
                        "person_id",
                        "condition_concept_id",
                    ],
                    "column_types": {
                        "condition_era_id": "int64",
                        "person_id": "int64",
                        "condition_concept_id": "int64",
                        "condition_era_start_date": "object",  # Date string
                        "condition_era_end_date": "object",  # Date string
                        "condition_occurrence_count": "int64",
                    },
                    "primary_key": "condition_era_id",
                    "foreign_keys": {"person_id": "person.person_id"},
                },
                "care_site": {
                    "required_columns": ["care_site_id", "care_site_name"],
                    "column_types": {
                        "care_site_id": "int64",
                        "care_site_name": "float64",
                        "place_of_service_concept_id": "int64",
                        "location_id": "float64",
                        "care_site_source_value": "object",
                        "place_of_service_source_value": "object",
                    },
                    "primary_key": "care_site_id",
                    "foreign_keys": {},
                },
            },
        }

        # Pharmacovigilance schema
        schemas["pharmacv"] = {
            "required_tables": [
                "DEMO_cleaned",
                "DRUG_cleaned",
                "INDI_cleaned",
                "OUTC_cleaned",
                "REAC_cleaned",
                "RPSR_cleaned",
                "THER_cleaned",
            ],
            "table_schemas": {
                "DEMO_cleaned": {
                    "required_columns": ["primaryid", "caseid"],
                    "column_types": {
                        "primaryid": "int64",
                        "caseid": "int64",
                        "caseversion": "int64",
                        "fda_dt": "int64",
                        "age": "float64",  # Can be NaN
                        "sex": "object",
                        "reporter_country": "object",
                        "rept_cod": "object",
                    },
                    "primary_key": "primaryid",
                    "foreign_keys": {},
                },
                "DRUG_cleaned": {
                    "required_columns": ["primaryid", "drug_seq"],
                    "column_types": {
                        "primaryid": "int64",
                        "drug_seq": "int64",
                        "role_cod": "object",
                        "drugname": "object",
                        "val_vbm": "object",
                        "route": "object",
                        "dose_vbm": "object",
                        "cum_dose_chr": "object",
                        "cum_dose_unit": "object",
                        "dechal": "object",
                        "rechal": "object",
                        "lot_num": "object",
                        "exp_dt": "object",
                        "nda_num": "object",
                    },
                    "primary_key": None,  # Composite key
                    "foreign_keys": {"primaryid": "DEMO_cleaned.primaryid"},
                },
                "INDI_cleaned": {
                    "required_columns": ["primaryid", "indi_drug_seq", "indi_pt"],
                    "column_types": {
                        "primaryid": "int64",
                        "indi_drug_seq": "int64",
                        "indi_pt": "object",
                    },
                    "primary_key": None,
                    "foreign_keys": {"primaryid": "DEMO_cleaned.primaryid"},
                },
                "OUTC_cleaned": {
                    "required_columns": ["primaryid", "outc_cod"],
                    "column_types": {"primaryid": "int64", "outc_cod": "object"},
                    "primary_key": None,
                    "foreign_keys": {"primaryid": "DEMO_cleaned.primaryid"},
                },
                "REAC_cleaned": {
                    "required_columns": ["primaryid", "pt"],
                    "column_types": {"primaryid": "int64", "pt": "object"},
                    "primary_key": None,
                    "foreign_keys": {"primaryid": "DEMO_cleaned.primaryid"},
                },
                "RPSR_cleaned": {
                    "required_columns": ["primaryid", "rpsr_cod"],
                    "column_types": {"primaryid": "int64", "rpsr_cod": "object"},
                    "primary_key": None,
                    "foreign_keys": {"primaryid": "DEMO_cleaned.primaryid"},
                },
                "THER_cleaned": {
                    "required_columns": [
                        "primaryid",
                        "dsg_drug_seq",
                        "start_dt",
                        "end_dt",
                    ],
                    "column_types": {
                        "primaryid": "int64",
                        "dsg_drug_seq": "int64",
                        "start_dt": "object",
                        "end_dt": "object",
                        "dur": "object",
                        "dur_cod": "object",
                    },
                    "primary_key": None,
                    "foreign_keys": {"primaryid": "DEMO_cleaned.primaryid"},
                },
            },
        }

        return schemas

    def validate_dataset(
        self, dataset: Dict[str, pd.DataFrame], domain: str, usecase: str
    ) -> ValidationResult:
        """
        Validate a custom uploaded dataset against domain schema.

        Args:
            dataset: Dictionary of table_name -> DataFrame
            domain: Domain name (e.g., "Pharma")
            usecase: Usecase name (e.g., "Patient Cohort Builder")

        Returns:
            ValidationResult with validation issues and summary
        """
        issues = []
        warnings = []
        info = []

        # Map domain/usecase to schema key
        schema_key = self._get_schema_key(domain, usecase)
        if schema_key not in self.domain_schemas:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    table_name="",
                    column_name=None,
                    issue_type="unsupported_domain",
                    message=f"Domain '{domain}' with usecase '{usecase}' is not supported for validation",
                    suggested_fix="Use a supported domain/usecase combination",
                )
            )
            return ValidationResult(
                is_valid=False,
                issues=issues,
                warnings=warnings,
                info=info,
                summary={"total_issues": len(issues)},
            )

        schema = self.domain_schemas[schema_key]

        # Check for missing required tables
        missing_tables = self._check_missing_tables(dataset, schema)
        issues.extend(missing_tables)

        # Validate each table
        for table_name, df in dataset.items():
            if table_name in schema["table_schemas"]:
                table_issues = self._validate_table(
                    df, table_name, schema["table_schemas"][table_name]
                )
                issues.extend(
                    [i for i in table_issues if i.severity == ValidationSeverity.ERROR]
                )
                warnings.extend(
                    [
                        i
                        for i in table_issues
                        if i.severity == ValidationSeverity.WARNING
                    ]
                )
                info.extend(
                    [i for i in table_issues if i.severity == ValidationSeverity.INFO]
                )
            else:
                warnings.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        table_name=table_name,
                        column_name=None,
                        issue_type="unexpected_table",
                        message=f"Table '{table_name}' is not expected for this domain/usecase",
                        suggested_fix="Consider removing this table or verify the correct domain/usecase",
                    )
                )

        # Validate relationships
        relationship_issues = self._validate_relationships(dataset, schema)
        issues.extend(
            [i for i in relationship_issues if i.severity == ValidationSeverity.ERROR]
        )
        warnings.extend(
            [i for i in relationship_issues if i.severity == ValidationSeverity.WARNING]
        )

        # Generate summary
        summary = self._generate_summary(dataset, issues, warnings, info)

        return ValidationResult(
            is_valid=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            info=info,
            summary=summary,
        )

    def _get_schema_key(self, domain: str, usecase: str) -> str:
        """Map domain/usecase to schema key."""
        if domain == "Pharma":
            if usecase == "Patient Cohort Builder":
                return "pharmacohort"
            elif usecase == "Pharmacovigilance":
                return "pharmacv"
        return ""

    def _check_missing_tables(
        self, dataset: Dict[str, pd.DataFrame], schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check for missing required tables."""
        issues = []
        required_tables = schema["required_tables"]
        present_tables = set(dataset.keys())

        for required_table in required_tables:
            if required_table not in present_tables:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        table_name=required_table,
                        column_name=None,
                        issue_type="missing_table",
                        message=f"Required table '{required_table}' is missing",
                        suggested_fix=f"Upload a CSV file named '{required_table}.csv'",
                    )
                )

        return issues

    def _validate_table(
        self, df: pd.DataFrame, table_name: str, table_schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate a single table against its schema."""
        issues = []

        # Check for missing required columns
        missing_columns = self._check_missing_columns(df, table_name, table_schema)
        issues.extend(missing_columns)

        # Validate column data types
        type_issues = self._validate_column_types(df, table_name, table_schema)
        issues.extend(type_issues)

        # Validate primary key
        pk_issues = self._validate_primary_key(df, table_name, table_schema)
        issues.extend(pk_issues)

        # Check for empty tables
        if len(df) == 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    table_name=table_name,
                    column_name=None,
                    issue_type="empty_table",
                    message=f"Table '{table_name}' is empty",
                    suggested_fix="Ensure the table contains at least one row of data",
                )
            )

        # Check for duplicate rows
        if len(df) != len(df.drop_duplicates()):
            duplicate_count = len(df) - len(df.drop_duplicates())
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    table_name=table_name,
                    column_name=None,
                    issue_type="duplicate_rows",
                    message=f"Table '{table_name}' contains {duplicate_count} duplicate rows",
                    suggested_fix="Consider removing duplicate rows for better data quality",
                )
            )

        return issues

    def _check_missing_columns(
        self, df: pd.DataFrame, table_name: str, table_schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Check for missing required columns."""
        issues = []
        required_columns = table_schema.get("required_columns", [])
        present_columns = set(df.columns)

        for required_col in required_columns:
            if required_col not in present_columns:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        table_name=table_name,
                        column_name=required_col,
                        issue_type="missing_column",
                        message=f"Required column '{required_col}' is missing from table '{table_name}'",
                        suggested_fix=f"Add column '{required_col}' to the table",
                    )
                )

        return issues

    def _validate_column_types(
        self, df: pd.DataFrame, table_name: str, table_schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate column data types."""
        issues = []
        expected_types = table_schema.get("column_types", {})

        for col_name, expected_type in expected_types.items():
            if col_name not in df.columns:
                continue  # Missing column already reported

            actual_type = str(df[col_name].dtype)

            # Handle flexible type matching
            if not self._types_compatible(actual_type, expected_type):
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        table_name=table_name,
                        column_name=col_name,
                        issue_type="invalid_data_type",
                        message=f"Column '{col_name}' has type '{actual_type}' but expected '{expected_type}'",
                        suggested_fix=f"Convert column '{col_name}' to the correct data type",
                    )
                )

        return issues

    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual and expected types are compatible."""
        # Handle common type variations
        type_mappings = {
            "int64": ["int64", "int32", "int16", "int8"],
            "float64": ["float64", "float32", "int64", "int32", "int16", "int8"],
            "object": ["string", "object"],
            "bool": ["bool", "int64", "int32", "int16", "int8"],
        }

        if actual_type == expected_type:
            return True

        if expected_type in type_mappings:
            return actual_type in type_mappings[expected_type]

        return False

    def _validate_primary_key(
        self, df: pd.DataFrame, table_name: str, table_schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate primary key constraints."""
        issues = []
        primary_key = table_schema.get("primary_key")

        if not primary_key:
            return issues  # No primary key defined

        if primary_key not in df.columns:
            return issues  # Missing column already reported

        # Check for null values in primary key
        null_count = df[primary_key].isnull().sum()
        if null_count > 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    table_name=table_name,
                    column_name=primary_key,
                    issue_type="null_primary_key",
                    message=f"Primary key column '{primary_key}' contains {null_count} null values",
                    suggested_fix="Remove rows with null primary key values or provide valid primary key values",
                )
            )

        # Check for duplicate primary key values
        duplicate_count = df[primary_key].duplicated().sum()
        if duplicate_count > 0:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    table_name=table_name,
                    column_name=primary_key,
                    issue_type="duplicate_primary_key",
                    message=f"Primary key column '{primary_key}' contains {duplicate_count} duplicate values",
                    suggested_fix="Ensure primary key values are unique",
                )
            )

        return issues

    def _validate_relationships(
        self, dataset: Dict[str, pd.DataFrame], schema: Dict[str, Any]
    ) -> List[ValidationIssue]:
        """Validate foreign key relationships."""
        issues = []

        for table_name, table_schema in schema["table_schemas"].items():
            if table_name not in dataset:
                continue

            foreign_keys = table_schema.get("foreign_keys", {})
            df = dataset[table_name]

            for fk_column, reference in foreign_keys.items():
                if fk_column not in df.columns:
                    continue  # Missing column already reported

                # Parse reference (e.g., "person.person_id")
                ref_table, ref_column = reference.split(".")

                if ref_table not in dataset:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            table_name=table_name,
                            column_name=fk_column,
                            issue_type="missing_referenced_table",
                            message=f"Foreign key '{fk_column}' references table '{ref_table}' which is not present",
                            suggested_fix=f"Upload table '{ref_table}' or remove the foreign key relationship",
                        )
                    )
                    continue

                ref_df = dataset[ref_table]
                if ref_column not in ref_df.columns:
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            table_name=table_name,
                            column_name=fk_column,
                            issue_type="missing_referenced_column",
                            message=f"Foreign key '{fk_column}' references column '{ref_column}' which is not present in table '{ref_table}'",
                            suggested_fix=f"Add column '{ref_column}' to table '{ref_table}'",
                        )
                    )
                    continue

                # Check for orphaned foreign key values
                fk_values = set(df[fk_column].dropna().unique())
                ref_values = set(ref_df[ref_column].dropna().unique())
                orphaned_values = fk_values - ref_values

                if orphaned_values:
                    orphaned_count = len(orphaned_values)
                    issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.ERROR,
                            table_name=table_name,
                            column_name=fk_column,
                            issue_type="orphaned_foreign_key",
                            message=f"Foreign key '{fk_column}' contains {orphaned_count} values that don't exist in referenced table '{ref_table}'",
                            suggested_fix=f"Remove orphaned values or add corresponding records to table '{ref_table}'",
                        )
                    )

        return issues

    def _generate_summary(
        self,
        dataset: Dict[str, pd.DataFrame],
        issues: List[ValidationIssue],
        warnings: List[ValidationIssue],
        info: List[ValidationIssue],
    ) -> Dict[str, Any]:
        """Generate validation summary."""
        return {
            "total_tables": len(dataset),
            "total_rows": sum(len(df) for df in dataset.values()),
            "total_columns": sum(len(df.columns) for df in dataset.values()),
            "total_issues": len(issues),
            "total_warnings": len(warnings),
            "total_info": len(info),
            "validation_passed": len(issues) == 0,
            "table_summary": {
                name: {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                }
                for name, df in dataset.items()
            },
        }


def validate_custom_dataset(
    dataset: Dict[str, pd.DataFrame], domain: str, usecase: str
) -> ValidationResult:
    """
    Convenience function to validate a custom dataset.

    Args:
        dataset: Dictionary of table_name -> DataFrame
        domain: Domain name (e.g., "Pharma")
        usecase: Usecase name (e.g., "Patient Cohort Builder")

    Returns:
        ValidationResult with validation issues and summary
    """
    validator = DatasetValidator()
    return validator.validate_dataset(dataset, domain, usecase)
