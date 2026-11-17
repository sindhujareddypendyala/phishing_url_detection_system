import sqlite3

DB_NAME = "database/scans.db"

def create_table():

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS scans(

            id INTEGER PRIMARY KEY,

            url TEXT,

            prediction TEXT,

            confidence REAL,

            timestamp DATETIME
            DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    conn.commit()

    conn.close()

def save_scan(
    url,
    prediction,
    confidence
):

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO scans(
            url,
            prediction,
            confidence
        )

        VALUES(
            ?,
            ?,
            ?
        )
        """,
        (
            url,
            prediction,
            confidence
        )
    )

    conn.commit()

    conn.close()

def get_recent_scans():

    conn = sqlite3.connect(
        DB_NAME
    )

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM scans

        ORDER BY id DESC

        LIMIT 20
        """
    )

    rows = cursor.fetchall()

    conn.close()

    return rows