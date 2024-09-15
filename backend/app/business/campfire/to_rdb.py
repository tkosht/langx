import datetime
import os
from inspect import signature

import pandas as pd
import typer
from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase, Record

# from neo4j.graph import Node
from omegaconf import DictConfig
from sqlalchemy import Engine, create_engine, text
from typing_extensions import Self

load_dotenv()


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Neo4jProvider(object):
    def __init__(self, database: str = "neo4j"):
        self.database: str = database
        self.driver: Driver = self.neo4j_driver()

    def neo4j_driver(self) -> Driver:
        neo4j_user = os.environ["neo4j_user"]
        neo4j_pswd = os.environ["neo4j_pswd"]
        driver: Driver = GraphDatabase.driver(uri="bolt://neo4j", auth=(neo4j_user, neo4j_pswd), database=self.database)
        return driver

    def select_query(self, query: str, **props):
        def _select(tx, query: str, **props):
            results = tx.run(query, **props)
            return list(results)

        results = []
        with self.driver.session() as session:
            results = session.execute_read(_select, query, **props)
        return results


class PostgresProvider(object):
    def __init__(self, host: str = "postgresql", port: int = 5432, dbname: str = "campfire_db") -> None:
        self.host: str = host
        self.port: int = port
        self.dbname: str = dbname
        self.engine: Engine = self.create_engine()

    def create_engine(self) -> Engine:
        postgres_user = os.environ["postgres_user"]
        postgres_pswd = os.environ["postgres_pswd"]
        engine: Engine = create_engine(
            f"postgresql://{postgres_user}:{postgres_pswd}@{self.host}:{self.port}/{self.dbname}"
        )
        return engine

    def select_query(self, query: str, **props) -> pd.DataFrame:
        with self.engine.connect() as cnn:
            df = pd.read_sql(sql=text(query), con=cnn, **props)
        return df

    def execute(self, sql: str, **props):
        with self.engine.connect() as cnn:
            # results = cnn.execution_options(autocommit=True).execute(text(sql), **props)
            results = cnn.exec_driver_sql(sql, **props)
            cnn.commit()
        return results

    def do_import(self, table_name: str, index_keys: list[str], records: list[dict]):
        print(f"{now()} start importing...", table_name, len(records))
        df = pd.DataFrame(records)
        df.index = [df[k] for k in index_keys]
        df.drop(columns=index_keys, inplace=True)

        from sqlalchemy import types  # import Date, Float, Integer, String

        dtypes = {col: types.Date() for col in df.select_dtypes(include=["datetime"]).columns}
        dtypes.update({col: types.Integer() for col in df.select_dtypes(include=["int", "integer"]).columns})
        dtypes.update(
            {col: types.Float() for col in set(df.select_dtypes(include=["number"]).columns) - set(dtypes.keys())}
        )
        dtypes.update({col: types.String() for col in set(df.columns) - set(dtypes.keys())})

        df.to_sql(
            table_name,  # like "projects"
            self.engine,
            if_exists="replace",
            index=True,
            index_label=index_keys,
            dtype=dtypes,
        )
        print(f"{now()} end importing...", table_name, len(records))
        return self

    def create_views(self, sql_dir: str = "conf/sql") -> Self:
        for sql_file in os.listdir(sql_dir):
            if sql_file.endswith(".sql"):
                with open(os.path.join(sql_dir, sql_file)) as f:
                    sql = f.read()
                self.execute(sql)


def nvl(val, default):
    return val if val is not None else default


def _main(params: DictConfig):
    neo4j_provider = Neo4jProvider()
    postgres_provider = PostgresProvider(port=5431)

    # drop tables
    tables = ["projects", "project_details"]
    for tbl in tables:
        postgres_provider.execute(f"drop table if exists {tbl} cascade")

    # 1. projects
    query = """
    match (pr:ProjectRoot)--(dr:ProjectDataRoot) 
    with pr, count(dr.execution_id) as cnt
    match (pr)-[*]->(pj:Project)-->(pjd:ProjectDetails)
    return cnt, pr, pj, pjd
    order by cnt desc, pj.project_id, pj.sortby
    """

    print(f"{now()} start selecting projects...")
    results: list[Record] = neo4j_provider.select_query(query)
    print(f"{now()} end selecting projects")

    records = []
    for rec in results:
        # return の順番で取得 ('*' の場合は、アルファベット順)
        n_dates, pr, pj, pjd = rec.values()
        created_at = datetime.datetime.strptime(pj["created_at"], "%Y-%m-%d")
        page: int = int(pj["page"]) if pj["page"] is not None else None
        data_position: int = int(pj["data_position"])
        ranking: int = data_position + page * 20 if page is not None else None
        record = dict(
            execution_id=pj["execution_id"],
            project_id=pj["project_id"],
            project_url=pj["project_url"],
            title=pj["title"],
            data_brand=pj["data_brand"],
            data_category=pj["data_category"],
            data_dimension=pj["data_dimension"],
            sortby=pj["sortby"],
            page=page,
            data_position=data_position,
            ranking=ranking,
            current_funding=int(nvl(pj["current_funding"], 0)),
            current_supporters=int(nvl(pj["current_supporters"], 0)),
            success_rate=int(nvl(pj["success_rate"], 0)),  # %
            remaining_days=int(nvl(pj["remaining_days"], -1)),
            created_at=created_at,  # already `datetime` type
        )
        records.append(record)

    postgres_provider.do_import(
        table_name="projects",
        index_keys=["project_id", "execution_id", "sortby"],
        records=records,
    )

    # 2. project_details
    query = """
    match (pr: ProjectRoot) --> (dr: ProjectDataRoot) --> (pj:Project) --> (pjd:ProjectDetails)
    return pr, dr, pj, pjd
    """
    print(f"{now()} start selecting project_details...")
    results: list[Record] = neo4j_provider.select_query(query)
    print(f"{now()} end selecting project_details")

    records = []
    for idx, rec in enumerate(results):
        record = {}
        [record.update(d) for d in rec.data("pr", "dr", "pj", "pjd").values()]
        record["data_position"] = int(pj["data_position"])
        record["current_funding"] = int(nvl(pj["current_funding"], 0))
        record["current_supporters"] = int(nvl(pj["current_supporters"], 0))
        record["success_rate"] = int(nvl(pj["success_rate"], 0))  # %
        record["remaining_days"] = int(nvl(pj["remaining_days"], -1))
        record["created_at"] = datetime.datetime.strptime(record["created_at"], "%Y-%m-%d")
        records.append(record)

    postgres_provider.do_import(
        table_name="project_details",
        index_keys=["project_id", "execution_id", "sortby", "project_data_id"],
        records=records,
    )

    # 3. return_boxes
    records = []
    query = """
    match(r:ReturnBox)
    with r.project_id as project_id, max(r.created_at) as latest
    match (rbx:ReturnBox {project_id: project_id, created_at: latest})
    return rbx
    """

    print(f"{now()} start selecting return_boxes ...")
    boxes: list[Record] = neo4j_provider.select_query(query)
    print(f"{now()} end selecting return_boxes ...")

    for bx in boxes:
        record = {}
        [record.update(bx.data()["rbx"])]
        record["return_idx"] = int(nvl(record["return_idx"], -1))
        record["created_at"] = datetime.datetime.strptime(record["created_at"], "%Y-%m-%d")
        records.append(record)

    postgres_provider.do_import(
        table_name="return_boxes",
        index_keys=["project_id", "execution_id", "sortby", "project_data_id", "return_idx"],
        records=records,
    )

    # last. create views
    print(f"{now()} start creating views...")
    postgres_provider.create_views()
    print(f"{now()} end creating views")

    print("done.")


def config():
    cfg = DictConfig(dict(is_experiment=True, do_share=False))
    return cfg


def main(
    execution_id: str = "20240301000000",
):
    s = signature(main)
    kwargs = {}
    for k in list(s.parameters):
        v = locals()[k]
        if v is not None:
            kwargs[k] = v

    params = config()  # use as default
    params.update(kwargs)
    return _main(params)


if __name__ == "__main__":
    typer.run(main)
