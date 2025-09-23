import os
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


def create_connection():
    connection = None
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            port=os.getenv("DB_PORT", "5432"),
        )
        print("Connection to PostgreSQL DB successful")
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


def fetch_query(connection, query):
    """For SELECT queries that return rows"""
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        records = cursor.fetchall()
        return records
    except Error as e:
        print(f"The error '{e}' occurred")
        return []


def insert_job(connection, job_data):
    """
    Insert a job record into the jobs table.

    Parameters:
        connection: psycopg2 connection object
        job_data: dict mapping column name -> value, e.g. {"title": "...", "company": "..."}

    Returns:
        The newly inserted job id if the table has an 'id' serial/identity column and RETURNING works; otherwise None.
    """
    if not job_data or not isinstance(job_data, dict):
        raise ValueError("job_data must be a non-empty dict of column -> value")

    columns = ", ".join(job_data.keys())
    placeholders = ", ".join(["%s"] * len(job_data))
    values = list(job_data.values())

    query = f"INSERT INTO jobs ({columns}) VALUES ({placeholders}) RETURNING id;"

    cursor = connection.cursor()
    try:
        cursor.execute(query, values)
        inserted_id_row = cursor.fetchone()
        connection.commit()
        return inserted_id_row[0] if inserted_id_row else None
    except Error as e:
        connection.rollback()
        print(f"The error '{e}' occurred")
        return None
    finally:
        cursor.close()


def select_jobs(
    connection, where_clause=None, params=None, order_by=None, limit=None, offset=None
):
    """
    Select jobs from the jobs table.

    Parameters:
        connection: psycopg2 connection object
        where_clause: optional SQL string for WHERE without the 'WHERE' keyword. Use placeholders (%s) for params.
        params: tuple/list of parameters for the where_clause placeholders
        order_by: optional SQL fragment for ORDER BY without the 'ORDER BY' keyword
        limit: optional integer to limit results
        offset: optional integer to offset results

    Returns:
        List of rows (tuples) as returned by cursor.fetchall().
    """
    sql_parts = ["SELECT * FROM jobs"]

    if where_clause:
        sql_parts.append(f"WHERE {where_clause}")

    if order_by:
        sql_parts.append(f"ORDER BY {order_by}")

    if limit is not None:
        sql_parts.append("LIMIT %s")

    if offset is not None:
        sql_parts.append("OFFSET %s")

    sql = " ".join(sql_parts)

    exec_params = []
    if params:
        exec_params.extend(list(params))
    if limit is not None:
        exec_params.append(limit)
    if offset is not None:
        exec_params.append(offset)

    cursor = connection.cursor()
    try:
        cursor.execute(sql, tuple(exec_params) if exec_params else None)
        rows = cursor.fetchall()
        return rows
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
):
    """Insert a synthesis job row into jobs table using provided schema, including file_type TEXT."""
    if file_size is None and file_bytes is not None:
        file_size = len(file_bytes)

    sql = (
        "INSERT INTO jobs (job_id, file_name, file_size, num_files, status, model_used, version, file_type, csv_bytes) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
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
        "SELECT job_id, created_at, file_name, file_size, num_files, status, model_used, version, file_type, csv_bytes "
        "FROM jobs WHERE job_id = %s"
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
            "csv_bytes": row[9],
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
            SELECT * FROM jobs;
        """
        rows = fetch_query(connection, query)
        for row in rows:
            print(row)
        connection.close()
