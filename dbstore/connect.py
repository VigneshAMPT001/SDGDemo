import os
import psycopg2
from psycopg2 import Error, extras
from dotenv import load_dotenv
import json

# Load variables from .env file
load_dotenv()


def create_connection(isCloud: bool = False):
    connection = None
    try:
        if isCloud:
            connection = psycopg2.connect(os.getenv("DB_CONN_CLOUD"))
        else:
            connection = psycopg2.connect(
                host=os.getenv("DB_HOST", "localhost"),
                database=os.getenv("DB_NAME"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                port=os.getenv("DB_PORT", "5432"),
            )
        # print("Connection to PostgreSQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")
    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query executed successfully")
    except Error as e:
        print(f"The error '{e}' occurred")


def fetch_query(connection, query, params=None):
    """For SELECT queries that return rows"""
    cursor = connection.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        records = cursor.fetchall()
        return records
    except Error as e:
        print(f"The error '{e}' occurred")
        return []
    finally:
        cursor.close()


# ---------------------------
# Synthesis jobs helpers (UUID job_id, bytes)
# ---------------------------


def insert_job_record(
    connection,
    job_id,
    file_name,
    file_bytes,
    num_files,
    status,
    model_used,
    version=1,
    file_size=None,
    file_type=None,
    domain=None,
    use_case=None,
):
    """Insert a synthesis job row into jobs table using provided schema, including file_type TEXT."""
    if file_size is None and file_bytes is not None:
        file_size = len(file_bytes)

    sql = (
        "INSERT INTO synthetic_jobs (job_id, file_name, file_size, num_files, status, model_used, version, file_type, domain, use_case, csv_bytes) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    )
    params = (
        job_id,
        file_name,
        file_size,
        num_files,
        status,
        model_used,
        version,
        file_type,
        domain,
        use_case,
        psycopg2.Binary(file_bytes) if file_bytes is not None else None,
    )

    cursor = connection.cursor()
    try:
        cursor.execute(sql, params)
        connection.commit()
        return True
    except Error as e:
        connection.rollback()
        print(f"The error '{e}' occurred")
        return False
    finally:
        cursor.close()


def get_job_by_id(connection, job_id):
    """Fetch a job by job_id. Returns dict or None."""
    sql = (
        "SELECT job_id, created_at, file_name, file_size, num_files, status, model_used, version, file_type, domain, use_case, csv_bytes "
        "FROM synthetic_jobs WHERE job_id = %s"
    )
    cursor = connection.cursor()
    try:
        cursor.execute(sql, (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "job_id": row[0],
            "created_at": row[1],
            "file_name": row[2],
            "file_size": row[3],
            "num_files": row[4],
            "status": row[5],
            "model_used": row[6],
            "version": row[7],
            "file_type": row[8],
            "domain": row[9],
            "use_case": row[10],
            "csv_bytes": row[11],
        }
    except Error as e:
        print(f"The error '{e}' occurred")
        return None
    finally:
        cursor.close()


# ---------------------------
# Quality jobs helpers
# ---------------------------


def insert_quality_job(
    connection,
    job_id,
    domain,
    synthesis_job_id,
    status,
    property_scores,
    report_data,
    score=None,
    error=None,
    message=None,
):
    """Insert a quality job record into a quality_jobs table."""
    sql = """
        INSERT INTO quality_runs (job_id, domain, synthesis_job_id, status, score, property_scores, error, message, report_data, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
    """
    property_scores_string = json.dumps(property_scores)
    params = (
        job_id,
        domain,
        synthesis_job_id,
        status,
        score,
        property_scores_string,
        error,
        message,
        report_data,
    )

    cursor = connection.cursor()
    try:
        cursor.execute(sql, params)
        connection.commit()
        return True
    except Error as e:
        connection.rollback()
        print(f"The error '{e}' occurred")
        return False
    finally:
        cursor.close()


def get_quality_job_by_id(connection, job_id):
    """Fetch a quality job by job_id. Returns dict or None."""
    sql = """
        SELECT job_id, domain, synthesis_job_id, status, score, error, message, report_data, created_at, property_scores
	    FROM quality_runs WHERE synthesis_job_id = %s
    """
    cursor = connection.cursor()
    try:
        cursor.execute(sql, (job_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "job_id": row[0],
            "domain": row[1],
            "synthesis_job_id": row[2],
            "status": row[3],
            "score": row[4],
            "error": row[5],
            "message": row[6],
            "report_data": row[7],
            "created_at": row[8],
        }
    except Error as e:
        print(f"The error '{e}' occurred")
        return None
    finally:
        cursor.close()


if __name__ == "__main__":
    connection = create_connection()
    if connection:
        query = """
            SELECT * FROM synthetic_jobs;
        """
        rows = fetch_query(connection, query)
        for row in rows:
            print(row)
        connection.close()
