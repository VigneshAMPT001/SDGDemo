# Dataset Validation Guide

This document explains how the dataset validation system works and how to resolve common validation issues.

## Overview

The dataset validation system ensures that custom uploaded datasets meet the requirements for synthetic data generation. It validates:

1. **Schema Compliance** - Required tables and columns are present
2. **Data Types** - Columns have the correct data types
3. **Relationships** - Foreign key relationships are valid
4. **Data Quality** - No duplicate primary keys, no orphaned references

## Supported Domains and Usecases

### Pharma Domain

#### Patient Cohort Builder
**Required Tables:**
- `person.csv` - Patient demographic information
- `condition_era.csv` - Patient condition records  
- `care_site.csv` - Healthcare facility information

**Schema Requirements:**

**person.csv:**
- `person_id` (int64) - Primary key, must be unique
- `gender_concept_id` (int64) - Required
- `year_of_birth` (int64) - Required
- `month_of_birth` (int64) - Optional
- `day_of_birth` (int64) - Optional
- `birth_datetime` (float64) - Optional, can be NaN
- `race_concept_id` (int64) - Optional
- `ethnicity_concept_id` (int64) - Optional
- `location_id` (int64) - Optional
- `provider_id` (float64) - Optional, can be NaN
- `care_site_id` (float64) - Optional, can be NaN
- `person_source_value` (object) - Optional
- `gender_source_value` (object) - Optional
- `gender_source_concept_id` (float64) - Optional, can be NaN
- `race_source_value` (object) - Optional
- `race_source_concept_id` (float64) - Optional, can be NaN
- `ethnicity_source_value` (object) - Optional
- `ethnicity_source_concept_id` (float64) - Optional, can be NaN

**condition_era.csv:**
- `condition_era_id` (int64) - Primary key, must be unique
- `person_id` (int64) - Foreign key to person.person_id
- `condition_concept_id` (int64) - Required
- `condition_era_start_date` (object) - Date string in YYYY-MM-DD format
- `condition_era_end_date` (object) - Date string in YYYY-MM-DD format
- `condition_occurrence_count` (int64) - Required

**care_site.csv:**
- `care_site_id` (int64) - Primary key, must be unique
- `care_site_name` (object) - Required
- `place_of_service_concept_id` (int64) - Optional
- `location_id` (int64) - Optional
- `care_site_source_value` (object) - Optional
- `place_of_service_source_value` (object) - Optional

#### Pharmacovigilance
**Required Tables:**
- `DEMO_cleaned.csv` - Patient demographics
- `DRUG_cleaned.csv` - Drug information
- `INDI_cleaned.csv` - Indication data
- `OUTC_cleaned.csv` - Outcome data
- `REAC_cleaned.csv` - Reaction data
- `RPSR_cleaned.csv` - Reporter data
- `THER_cleaned.csv` - Therapy data

**Schema Requirements:**

**DEMO_cleaned.csv:**
- `primaryid` (int64) - Primary key, must be unique
- `caseid` (int64) - Required
- `caseversion` (int64) - Optional
- `fda_dt` (int64) - Optional
- `age` (float64) - Optional, can be NaN
- `sex` (object) - Optional
- `reporter_country` (object) - Optional
- `rept_cod` (object) - Optional

**DRUG_cleaned.csv:**
- `primaryid` (int64) - Foreign key to DEMO_cleaned.primaryid
- `drug_seq` (int64) - Required
- `role_cod` (object) - Optional
- `drugname` (object) - Optional
- `val_vbm` (object) - Optional
- `route` (object) - Optional
- `dose_vbm` (object) - Optional
- `cum_dose_chr` (object) - Optional
- `cum_dose_unit` (object) - Optional
- `dechal` (object) - Optional
- `rechal` (object) - Optional
- `lot_num` (object) - Optional
- `exp_dt` (object) - Optional
- `nda_num` (object) - Optional

**INDI_cleaned.csv:**
- `primaryid` (int64) - Foreign key to DEMO_cleaned.primaryid
- `indi_drug_seq` (int64) - Required
- `indi_pt` (object) - Required

**OUTC_cleaned.csv:**
- `primaryid` (int64) - Foreign key to DEMO_cleaned.primaryid
- `outc_cod` (object) - Required

**REAC_cleaned.csv:**
- `primaryid` (int64) - Foreign key to DEMO_cleaned.primaryid
- `pt` (object) - Required

**RPSR_cleaned.csv:**
- `primaryid` (int64) - Foreign key to DEMO_cleaned.primaryid
- `rpsr_cod` (object) - Required

**THER_cleaned.csv:**
- `primaryid` (int64) - Foreign key to DEMO_cleaned.primaryid
- `dsg_drug_seq` (int64) - Required
- `start_dt` (object) - Required
- `end_dt` (object) - Required
- `dur` (object) - Optional
- `dur_cod` (object) - Optional

## Validation Error Types

### Critical Issues (Block Synthesis)

1. **Missing Table** - Required table is not present
   - **Fix:** Upload the missing CSV file with the correct name

2. **Missing Column** - Required column is missing from a table
   - **Fix:** Add the missing column to your CSV file

3. **Invalid Data Type** - Column has wrong data type
   - **Fix:** Convert the column to the correct data type (e.g., convert strings to numbers)

4. **Null Primary Key** - Primary key column contains null values
   - **Fix:** Remove rows with null primary key values or provide valid values

5. **Duplicate Primary Key** - Primary key column contains duplicate values
   - **Fix:** Ensure all primary key values are unique

6. **Missing Referenced Table** - Foreign key references a table that doesn't exist
   - **Fix:** Upload the referenced table or remove the foreign key relationship

7. **Missing Referenced Column** - Foreign key references a column that doesn't exist
   - **Fix:** Add the referenced column to the target table

8. **Orphaned Foreign Key** - Foreign key values don't exist in the referenced table
   - **Fix:** Remove orphaned values or add corresponding records to the referenced table

9. **Empty Table** - Table contains no data
   - **Fix:** Ensure the table contains at least one row of data

### Warnings (Allow Synthesis)

1. **Unexpected Table** - Table is not expected for this domain/usecase
   - **Fix:** Remove the table or verify you're using the correct domain/usecase

2. **Duplicate Rows** - Table contains duplicate rows
   - **Fix:** Remove duplicate rows for better data quality

## Using the Validation System

### In Streamlit App

1. Upload your CSV files in the "Custom Dataset Upload" tab
2. Select the appropriate domain and usecase for validation
3. Click "🔍 Validate Dataset" to run validation
4. Review the validation results:
   - **Green checkmark** - Validation passed, ready for synthesis
   - **Red X** - Critical issues found, must be fixed
   - **Yellow warning** - Warnings found, can proceed but should be addressed
5. Fix any critical issues and re-validate
6. Proceed to synthesis once validation passes

### Via API

#### Validate Dataset
```bash
curl -X POST "http://localhost:8000/validate/dataset" \
  -F "files=@person.csv" \
  -F "files=@condition_era.csv" \
  -F "files=@care_site.csv" \
  -F "domain=Pharma" \
  -F "usecase=Patient Cohort Builder"
```

#### Get Expected Schema
```bash
curl "http://localhost:8000/validate/schema/Pharma/Patient%20Cohort%20Builder"
```

#### Quick Validation Check
```bash
curl -X POST "http://localhost:8000/validate/quick-check" \
  -F "files=@person.csv" \
  -F "files=@condition_era.csv" \
  -F "domain=Pharma" \
  -F "usecase=Patient Cohort Builder"
```

## Common Data Preparation Tips

### Data Type Conversion

**Convert strings to numbers:**
```python
import pandas as pd
df['person_id'] = pd.to_numeric(df['person_id'], errors='coerce')
```

**Handle missing values:**
```python
# For optional columns that can be NaN
df['provider_id'] = df['provider_id'].replace('', np.nan)
df['provider_id'] = pd.to_numeric(df['provider_id'], errors='coerce')
```

**Convert dates:**
```python
df['condition_era_start_date'] = pd.to_datetime(df['condition_era_start_date'], errors='coerce')
df['condition_era_start_date'] = df['condition_era_start_date'].dt.strftime('%Y-%m-%d')
```

### File Naming

- Use exact table names as specified in the schema
- File extensions should be `.csv`
- Case-sensitive: `person.csv` not `Person.csv`

### Data Quality

- Remove duplicate rows
- Ensure primary keys are unique
- Check that foreign key values exist in referenced tables
- Handle missing values appropriately (use NaN for optional columns)

## Troubleshooting

### "No schema available" Error
- Ensure you're using a supported domain/usecase combination
- Check that the domain and usecase names match exactly

### "Validation failed" Error
- Check that all CSV files can be parsed correctly
- Ensure files are not corrupted
- Verify file encoding (should be UTF-8)

### Foreign Key Issues
- Ensure all referenced tables are uploaded
- Check that foreign key column names match exactly
- Verify that foreign key values exist in the referenced table

### Data Type Issues
- Check for mixed data types in columns
- Look for non-numeric values in numeric columns
- Ensure date columns are in the correct format

## Best Practices

1. **Validate Early** - Run validation as soon as you upload your data
2. **Fix Critical Issues First** - Address all critical issues before proceeding
3. **Review Warnings** - While warnings don't block synthesis, they may affect data quality
4. **Test with Sample Data** - Start with a small sample to verify your data format
5. **Document Your Schema** - Keep track of your data structure and relationships
6. **Backup Your Data** - Keep original copies of your data files
