import psycopg2
import psycopg2.extensions
from project import ProjectRecord


class PostgresConnector:
    def __init__(self, host, port, database, user, password):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password

    def connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            database=self.database,
            user=self.user,
            password=self.password,
        )
        return conn

    def create_tables(self):
        self.create_table_campfire_list()

    def create_table_campfire_list(self):
        """
        Creates a table named 'campfire_list' if it does not already exist in the database.
        The table has the following columns:
        - id: SERIAL PRIMARY KEY
        - img_url: TEXT
        - detail_url: TEXT
        - area: TEXT
        - title: TEXT
        - meter: INTEGER
        - category: TEXT
        - owner: TEXT
        - current_funding: INTEGER
        - supporters: INTEGER
        - remaining_days: INTEGER
        - status: TEXT
        """

        cnn = cur = None
        try:
            cnn = self.connect()
            cur = cnn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS campfire_list (
                    id SERIAL PRIMARY KEY,
                    img_url TEXT,
                    detail_url TEXT,
                    area TEXT,
                    title TEXT,
                    meter INTEGER,
                    category TEXT,
                    owner TEXT,
                    current_funding INTEGER,
                    supporters INTEGER,
                    remaining_days INTEGER,
                    status TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_campfire_list_created_at ON campfire_list (created_at)
                """
            )
            cnn.commit()
        finally:
            if cur is not None:
                cur.close()
            if cnn is not None:
                cnn.rollback()
                cnn.close()

    def insert_project_records(self, records: list[ProjectRecord]):
        """
        Inserts the given records into the campfire_list table.

        Args:
            records (list): A list of records to be inserted into the table.

        Returns:
            None
        """

        cnn = cur = None
        try:
            cnn = self.connect()
            cur = cnn.cursor()
            cur.executemany(
                """
                INSERT INTO campfire_list (
                    img_url, detail_url, area, title, meter, category, owner, current_funding, supporters, remaining_days, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                records,
            )
            cnn.commit()
        finally:
            if cur is not None:
                cur.close()
            if cnn is not None:
                cnn.rollback()
                cnn.close()


if __name__ == "__main__":
    project_records: ProjectRecord = ()

    # # Connect to the Postgres database
    pgc = PostgresConnector(
        host="postgresql",
        port="5432",  # default port of Postgres
        database="campfire_db",
        user="postgres",
        password="postgres",
    )

    # # Create the table
    pgc.create_tables()

    # # Insert the records into the table
    # pgc.insert_project_records(project_records)
