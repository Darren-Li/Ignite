import pandas as pd
from sqlalchemy import create_engine

def load_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def load_db(db_type, conn_str, table):
    engine = create_engine(conn_str)
    return pd.read_sql(f"SELECT * FROM {table}", engine)
