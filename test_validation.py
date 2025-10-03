#!/usr/bin/env python3
"""
Test script for dataset validation functionality.

This script demonstrates how to use the dataset validation system
with sample data to test various validation scenarios.
"""

import pandas as pd
import numpy as np
from api_routers.dataset_validator import validate_custom_dataset


def create_sample_pharmacohort_data():
    """Create sample data for pharmacohort validation testing."""

    # Valid person data
    person_data = {
        "person_id": [1, 2, 3, 4, 5],
        "gender_concept_id": [8532, 8507, 8532, 8507, 8532],
        "year_of_birth": [1990, 1985, 1992, 1988, 1995],
        "month_of_birth": [1, 6, 3, 12, 8],
        "day_of_birth": [15, 22, 10, 5, 30],
        "birth_datetime": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "race_concept_id": [8527, 8527, 8527, 8527, 8527],
        "ethnicity_concept_id": [38003564, 38003564, 38003564, 38003564, 38003564],
        "location_id": [1, 2, 3, 4, 5],
        "provider_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "care_site_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "person_source_value": ["P001", "P002", "P003", "P004", "P005"],
        "gender_source_value": ["F", "M", "F", "M", "F"],
        "gender_source_concept_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "race_source_value": ["White", "White", "White", "White", "White"],
        "race_source_concept_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "ethnicity_source_value": [
            "Non-Hispanic",
            "Non-Hispanic",
            "Non-Hispanic",
            "Non-Hispanic",
            "Non-Hispanic",
        ],
        "ethnicity_source_concept_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
    }

    # Valid condition_era data
    condition_era_data = {
        "condition_era_id": [1, 2, 3, 4, 5],
        "person_id": [1, 2, 3, 4, 5],  # Valid foreign keys
        "condition_concept_id": [72926, 80502, 138384, 72926, 80502],
        "condition_era_start_date": [
            "2020-01-15",
            "2020-02-20",
            "2020-03-10",
            "2020-04-05",
            "2020-05-12",
        ],
        "condition_era_end_date": [
            "2020-01-15",
            "2020-02-20",
            "2020-03-10",
            "2020-04-05",
            "2020-05-12",
        ],
        "condition_occurrence_count": [1, 1, 1, 1, 1],
    }

    # Valid care_site data
    care_site_data = {
        "care_site_id": [1, 2, 3, 4, 5],
        "care_site_name": [
            "Hospital A",
            "Clinic B",
            "Hospital C",
            "Clinic D",
            "Hospital E",
        ],
        "place_of_service_concept_id": [9201, 9202, 9201, 9202, 9201],
        "location_id": [1, 2, 3, 4, 5],
        "care_site_source_value": ["HOSP_A", "CLIN_B", "HOSP_C", "CLIN_D", "HOSP_E"],
        "place_of_service_source_value": [
            "Inpatient",
            "Outpatient",
            "Inpatient",
            "Outpatient",
            "Inpatient",
        ],
    }

    return {
        "person": pd.DataFrame(person_data),
        "condition_era": pd.DataFrame(condition_era_data),
        "care_site": pd.DataFrame(care_site_data),
    }


def create_invalid_pharmacohort_data():
    """Create sample data with validation issues."""

    # Person data with issues
    person_data = {
        "person_id": [1, 2, 3, 4, 1],  # Duplicate primary key
        "gender_concept_id": [8532, 8507, 8532, 8507, 8532],
        "year_of_birth": [1990, 1985, 1992, 1988, 1995],
        # Missing required columns: month_of_birth, day_of_birth
        "birth_datetime": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "race_concept_id": [8527, 8527, 8527, 8527, 8527],
        "ethnicity_concept_id": [38003564, 38003564, 38003564, 38003564, 38003564],
        "location_id": [1, 2, 3, 4, 5],
        "provider_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "care_site_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "person_source_value": ["P001", "P002", "P003", "P004", "P005"],
        "gender_source_value": ["F", "M", "F", "M", "F"],
        "gender_source_concept_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "race_source_value": ["White", "White", "White", "White", "White"],
        "race_source_concept_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "ethnicity_source_value": [
            "Non-Hispanic",
            "Non-Hispanic",
            "Non-Hispanic",
            "Non-Hispanic",
            "Non-Hispanic",
        ],
        "ethnicity_source_concept_id": [np.nan, np.nan, np.nan, np.nan, np.nan],
    }

    # Condition era data with orphaned foreign key
    condition_era_data = {
        "condition_era_id": [1, 2, 3, 4, 5],
        "person_id": [1, 2, 3, 4, 999],  # 999 doesn't exist in person table
        "condition_concept_id": [72926, 80502, 138384, 72926, 80502],
        "condition_era_start_date": [
            "2020-01-15",
            "2020-02-20",
            "2020-03-10",
            "2020-04-05",
            "2020-05-12",
        ],
        "condition_era_end_date": [
            "2020-01-15",
            "2020-02-20",
            "2020-03-10",
            "2020-04-05",
            "2020-05-12",
        ],
        "condition_occurrence_count": [1, 1, 1, 1, 1],
    }

    return {
        "person": pd.DataFrame(person_data),
        "condition_era": pd.DataFrame(condition_era_data),
        # Missing care_site table
    }


def test_valid_data():
    """Test validation with valid data."""
    print("=" * 60)
    print("TESTING VALID DATA")
    print("=" * 60)

    dataset = create_sample_pharmacohort_data()
    result = validate_custom_dataset(dataset, "Pharma", "Patient Cohort Builder")

    print(f"Validation passed: {result.is_valid}")
    print(f"Total issues: {len(result.issues)}")
    print(f"Total warnings: {len(result.warnings)}")
    print(f"Total info: {len(result.info)}")

    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(f"  - {issue.table_name}: {issue.message}")

    if result.warnings:
        print("\nWarnings found:")
        for warning in result.warnings:
            print(f"  - {warning.table_name}: {warning.message}")

    print(f"\nSummary: {result.summary}")


def test_invalid_data():
    """Test validation with invalid data."""
    print("\n" + "=" * 60)
    print("TESTING INVALID DATA")
    print("=" * 60)

    dataset = create_invalid_pharmacohort_data()
    result = validate_custom_dataset(dataset, "Pharma", "Patient Cohort Builder")

    print(f"Validation passed: {result.is_valid}")
    print(f"Total issues: {len(result.issues)}")
    print(f"Total warnings: {len(result.warnings)}")
    print(f"Total info: {len(result.info)}")

    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(
                f"  - {issue.severity.value.upper()}: {issue.table_name}: {issue.message}"
            )
            if issue.suggested_fix:
                print(f"    Suggested fix: {issue.suggested_fix}")

    if result.warnings:
        print("\nWarnings found:")
        for warning in result.warnings:
            print(
                f"  - {warning.severity.value.upper()}: {warning.table_name}: {warning.message}"
            )
            if warning.suggested_fix:
                print(f"    Suggested fix: {warning.suggested_fix}")

    print(f"\nSummary: {result.summary}")


def test_unsupported_domain():
    """Test validation with unsupported domain."""
    print("\n" + "=" * 60)
    print("TESTING UNSUPPORTED DOMAIN")
    print("=" * 60)

    dataset = create_sample_pharmacohort_data()
    result = validate_custom_dataset(dataset, "Unsupported", "Invalid Usecase")

    print(f"Validation passed: {result.is_valid}")
    print(f"Total issues: {len(result.issues)}")

    if result.issues:
        print("\nIssues found:")
        for issue in result.issues:
            print(f"  - {issue.severity.value.upper()}: {issue.message}")


if __name__ == "__main__":
    print("Dataset Validation Test Suite")
    print(
        "This script tests the dataset validation functionality with various scenarios."
    )

    try:
        test_valid_data()
        test_invalid_data()
        test_unsupported_domain()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED")
        print("=" * 60)

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
