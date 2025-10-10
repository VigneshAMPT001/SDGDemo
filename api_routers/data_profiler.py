from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any, List, Optional, Union
import io
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import warnings


router = APIRouter(prefix="/data_profiler", tags=["data_profiler"])


def _infer_schema(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def _int_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]


async def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    try:
        content = await file.read()
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read {file.filename}: {str(e)}"
        )


def _get_numeric_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Calculate comprehensive numeric statistics for a column"""
    try:
        numeric_series = pd.to_numeric(df[col], errors="coerce")
        numeric_series = numeric_series.dropna()

        if len(numeric_series) == 0:
            return {"error": "No valid numeric values"}

        stats_dict = {
            "count": len(numeric_series),
            "mean": float(numeric_series.mean()),
            "median": float(numeric_series.median()),
            "mode": (
                float(numeric_series.mode().iloc[0])
                if len(numeric_series.mode()) > 0
                else None
            ),
            "std": float(numeric_series.std()),
            "var": float(numeric_series.var()),
            "min": float(numeric_series.min()),
            "max": float(numeric_series.max()),
            "range": float(numeric_series.max() - numeric_series.min()),
            "q25": float(numeric_series.quantile(0.25)),
            "q50": float(numeric_series.quantile(0.50)),
            "q75": float(numeric_series.quantile(0.75)),
            "iqr": float(numeric_series.quantile(0.75) - numeric_series.quantile(0.25)),
            "sum": float(numeric_series.sum()),
            "product": (
                float(numeric_series.prod()) if len(numeric_series) < 100 else None
            ),
            "skewness": float(numeric_series.skew()),
            "kurtosis": float(numeric_series.kurtosis()),
            "mad": float(
                (numeric_series - numeric_series.mean()).abs().mean()
            ),  # Mean Absolute Deviation
            "sem": float(numeric_series.sem()),  # Standard Error of Mean
            "cv": (
                float(numeric_series.std() / numeric_series.mean())
                if numeric_series.mean() != 0
                else None
            ),  # Coefficient of Variation
        }

        # Percentiles
        percentiles = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
        for p in percentiles:
            stats_dict[f"p{p}"] = float(numeric_series.quantile(p / 100))

        # Outlier detection using IQR method
        q1, q3 = numeric_series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = numeric_series[
            (numeric_series < lower_bound) | (numeric_series > upper_bound)
        ]
        stats_dict["outlier_count"] = len(outliers)
        stats_dict["outlier_percentage"] = float(
            len(outliers) / len(numeric_series) * 100
        )

        # Distribution analysis
        stats_dict["is_uniform"] = len(numeric_series.unique()) == len(numeric_series)
        stats_dict["zero_count"] = int((numeric_series == 0).sum())
        stats_dict["negative_count"] = int((numeric_series < 0).sum())
        stats_dict["positive_count"] = int((numeric_series > 0).sum())

        return stats_dict
    except Exception as e:
        return {"error": str(e)}


def _get_categorical_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Calculate comprehensive categorical statistics for a column"""
    try:
        series = df[col].dropna()
        if len(series) == 0:
            return {"error": "No valid values"}

        value_counts = series.value_counts()
        unique_values = series.unique()

        stats_dict = {
            "count": len(series),
            "unique_count": len(unique_values),
            "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
            "most_frequent_count": (
                int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
            ),
            "most_frequent_percentage": (
                float(value_counts.iloc[0] / len(series) * 100)
                if len(value_counts) > 0
                else 0
            ),
            "least_frequent": value_counts.index[-1] if len(value_counts) > 0 else None,
            "least_frequent_count": (
                int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0
            ),
            "entropy": float(
                stats.entropy(value_counts.values + 1e-10)
            ),  # Add small epsilon to avoid log(0)
            "gini_impurity": float(
                1 - sum((count / len(series)) ** 2 for count in value_counts.values)
            ),
        }

        # Top 10 most frequent values
        top_10 = value_counts.head(10)
        stats_dict["top_values"] = {
            str(val): int(count) for val, count in top_10.items()
        }

        # Distribution analysis
        stats_dict["is_uniform"] = len(unique_values) == len(series)
        stats_dict["has_duplicates"] = len(unique_values) < len(series)
        stats_dict["duplicate_percentage"] = float(
            (len(series) - len(unique_values)) / len(series) * 100
        )

        # String-specific analysis
        if series.dtype == "object":
            str_series = series.astype(str)
            stats_dict["avg_length"] = float(str_series.str.len().mean())
            stats_dict["min_length"] = int(str_series.str.len().min())
            stats_dict["max_length"] = int(str_series.str.len().max())
            stats_dict["empty_string_count"] = int((str_series == "").sum())
            stats_dict["whitespace_only_count"] = int(
                str_series.str.strip().eq("").sum()
            )
            stats_dict["contains_numbers"] = int(
                str_series.str.contains(r"\d", na=False).sum()
            )
            stats_dict["contains_special_chars"] = int(
                str_series.str.contains(r"[^a-zA-Z0-9\s]", na=False).sum()
            )

        return stats_dict
    except Exception as e:
        return {"error": str(e)}


def _get_datetime_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Calculate comprehensive datetime statistics for a column"""
    try:
        datetime_series = pd.to_datetime(df[col], errors="coerce").dropna()
        if len(datetime_series) == 0:
            return {"error": "No valid datetime values"}

        stats_dict = {
            "count": len(datetime_series),
            "min_date": datetime_series.min().isoformat(),
            "max_date": datetime_series.max().isoformat(),
            "date_range_days": int(
                (datetime_series.max() - datetime_series.min()).days
            ),
            "unique_dates": len(datetime_series.dt.date.unique()),
            "has_time_component": bool(
                (datetime_series.dt.time != pd.Timestamp("00:00:00").time()).any()
            ),
        }

        # Time-based analysis
        if stats_dict["has_time_component"]:
            stats_dict["min_time"] = datetime_series.dt.time.min().isoformat()
            stats_dict["max_time"] = datetime_series.dt.time.max().isoformat()

        # Date component analysis
        stats_dict["year_range"] = [
            int(datetime_series.dt.year.min()),
            int(datetime_series.dt.year.max()),
        ]
        stats_dict["month_distribution"] = (
            datetime_series.dt.month.value_counts().to_dict()
        )
        stats_dict["weekday_distribution"] = (
            datetime_series.dt.day_name().value_counts().to_dict()
        )

        return stats_dict
    except Exception as e:
        return {"error": str(e)}


def _get_boolean_stats(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Calculate boolean statistics for a column"""
    try:
        bool_series = df[col].dropna()
        if len(bool_series) == 0:
            return {"error": "No valid boolean values"}

        true_count = int(bool_series.sum())
        false_count = int(len(bool_series) - true_count)

        return {
            "count": len(bool_series),
            "true_count": true_count,
            "false_count": false_count,
            "true_percentage": float(true_count / len(bool_series) * 100),
            "false_percentage": float(false_count / len(bool_series) * 100),
        }
    except Exception as e:
        return {"error": str(e)}


def _profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive dataframe profiling with extensive statistical analysis"""
    rows, cols = df.shape

    # Basic info
    null_counts = df.isnull().sum().to_dict()
    unique_counts = df.nunique(dropna=True).to_dict()
    schema = _infer_schema(df)

    # Data quality metrics
    completeness = {
        col: float((rows - null_counts[col]) / rows * 100) for col in df.columns
    }
    uniqueness = {col: float(unique_counts[col] / rows * 100) for col in df.columns}

    # Column analysis
    column_analysis = {}

    for col in df.columns:
        col_analysis = {
            "dtype": str(df[col].dtype),
            "null_count": int(null_counts[col]),
            "null_percentage": float(null_counts[col] / rows * 100),
            "unique_count": int(unique_counts[col]),
            "unique_percentage": float(unique_counts[col] / rows * 100),
            "completeness": completeness[col],
            "memory_usage": int(df[col].memory_usage(deep=True)),
        }

        # Type-specific analysis
        if pd.api.types.is_numeric_dtype(df[col]):
            col_analysis["numeric_stats"] = _get_numeric_stats(df, col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            col_analysis["datetime_stats"] = _get_datetime_stats(df, col)
        elif (
            pd.api.types.is_bool_dtype(df[col])
            or df[col].dtype == "object"
            and df[col].isin([True, False, "True", "False", "true", "false"]).all()
        ):
            col_analysis["boolean_stats"] = _get_boolean_stats(df, col)
        else:
            col_analysis["categorical_stats"] = _get_categorical_stats(df, col)

        column_analysis[col] = col_analysis

    # Cross-column analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    correlation_matrix = {}
    if len(numeric_cols) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_matrix = df[numeric_cols].corr()
            correlation_matrix = {
                col: {
                    other_col: float(corr_matrix.loc[col, other_col])
                    for other_col in numeric_cols
                    if col != other_col
                }
                for col in numeric_cols
            }

    # Data quality summary
    data_quality = {
        "total_rows": rows,
        "total_columns": cols,
        "complete_columns": len([col for col in df.columns if null_counts[col] == 0]),
        "columns_with_nulls": len([col for col in df.columns if null_counts[col] > 0]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percentage": float(df.duplicated().sum() / rows * 100),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "avg_completeness": float(np.mean(list(completeness.values()))),
        "avg_uniqueness": float(np.mean(list(uniqueness.values()))),
    }

    return {
        "basic_info": {
            "rows": rows,
            "columns": cols,
            "schema": schema,
        },
        "data_quality": data_quality,
        "column_analysis": column_analysis,
        "correlation_matrix": correlation_matrix,
        "summary_stats": {
            "total_null_values": sum(null_counts.values()),
            "total_unique_values": sum(unique_counts.values()),
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(df.columns) - len(numeric_cols),
            "high_cardinality_columns": len(
                [col for col in df.columns if unique_counts[col] > rows * 0.8]
            ),
            "low_cardinality_columns": len(
                [col for col in df.columns if unique_counts[col] < 10]
            ),
        },
    }


@router.post("/profile")
async def profile_csvs(
    files: List[UploadFile] = File(..., description="One or more CSV files")
) -> Dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    result: Dict[str, Any] = {"tables": {}, "summary": {}}
    total_rows = 0
    total_columns = 0
    total_memory = 0
    all_numeric_cols = set()

    for f in files:
        df = await _read_csv_upload(f)
        table_profile = _profile_dataframe(df)
        name = (f.filename or "table").rsplit(".", 1)[0]
        result["tables"][name] = table_profile

        # Aggregate summary statistics
        total_rows += table_profile["basic_info"]["rows"]
        total_columns += table_profile["basic_info"]["columns"]
        total_memory += table_profile["data_quality"]["memory_usage_bytes"]

        # Collect numeric columns for cross-table analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_numeric_cols.update(numeric_cols)

    # Comprehensive summary statistics
    result["summary"] = {
        "total_tables": len(result["tables"]),
        "total_rows": total_rows,
        "total_columns": total_columns,
        "total_memory_bytes": total_memory,
        "total_memory_mb": round(total_memory / (1024 * 1024), 2),
        "avg_rows_per_table": round(total_rows / len(result["tables"]), 2),
        "avg_columns_per_table": round(total_columns / len(result["tables"]), 2),
        "unique_numeric_columns": len(all_numeric_cols),
        "table_names": list(result["tables"].keys()),
        "data_quality_overview": {
            "tables_with_complete_data": len(
                [
                    t
                    for t in result["tables"].values()
                    if t["data_quality"]["complete_columns"]
                    == t["data_quality"]["total_columns"]
                ]
            ),
            "tables_with_duplicates": len(
                [
                    t
                    for t in result["tables"].values()
                    if t["data_quality"]["duplicate_rows"] > 0
                ]
            ),
            "avg_completeness": round(
                np.mean(
                    [
                        t["data_quality"]["avg_completeness"]
                        for t in result["tables"].values()
                    ]
                ),
                2,
            ),
            "avg_uniqueness": round(
                np.mean(
                    [
                        t["data_quality"]["avg_uniqueness"]
                        for t in result["tables"].values()
                    ]
                ),
                2,
            ),
        },
    }
    return result
