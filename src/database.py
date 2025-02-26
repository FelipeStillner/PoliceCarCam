from datetime import datetime
import pandas as pd
from sqlalchemy import create_engine, inspect
import asyncio
import sqlite3


def backup_to_sqlite():
    HOST = "0.tcp.sa.ngrok.io:12484"
    DATABASE = "postgres"
    USER = "postgres"
    PASSWORD = "postgres"
    SQLITE_DB = "../Database/local.db"
    try:
        # Connect to the SQLite database and get the date of the last update
        sqlite_conn = sqlite3.connect(SQLITE_DB)
        cursor = sqlite_conn.cursor()
        cursor.execute("SELECT MAX(updated_at) FROM vehicle_log")
        last_update_date = cursor.fetchone()[0]
        cursor.close()

        # Connect to the PostgreSQL database using SQLAlchemy
        engine = create_engine(f"postgresql://{USER}:{PASSWORD}@{HOST}/{DATABASE}")
        print("Conexão com o banco de dados estabelecida com sucesso.")

        # Read tables from PostgreSQL database into Pandas DataFrames
        inspector = inspect(engine)
        table_names = inspector.get_table_names()
        dataframes = {}
        for table in table_names:
            if last_update_date:
                query = f"SELECT * FROM {table} WHERE updated_at > '{last_update_date}'"
            else:
                query = f"SELECT * FROM {table}"
            dataframes[table] = pd.read_sql_query(query, engine)

        # Write DataFrames to SQLite database
        for table, df in dataframes.items():
            # Insert new rows
            for index, row in df.iterrows():
                if not last_update_date or (
                    last_update_date
                    and row["created_at"]
                    >= datetime.strptime(last_update_date, "%Y-%m-%d").date()
                ):
                    print(f"Inserting row {index} into table {table}")
                    insert_query = f"""
                    INSERT INTO {table} ({', '.join(df.columns)})
                    VALUES ({', '.join(['?' for _ in df.columns])})
                    """
                    cursor = sqlite_conn.cursor()
                    cursor.execute(insert_query, tuple(row))
                    sqlite_conn.commit()
                    cursor.close()
            # Update existing rows
            for index, row in df.iterrows():
                update_query = f"""
                UPDATE {table}
                SET {', '.join([f"{col} = ?" for col in df.columns if col != 'id'])}
                WHERE id = ?
                """
                cursor = sqlite_conn.cursor()
                cursor.execute(
                    update_query,
                    tuple(row[col] for col in df.columns if col != "id") + (row["id"],),
                )
                sqlite_conn.commit()
                cursor.close()

        print("Backup concluído com sucesso.")

    except Exception as e:
        print(f"Erro ao fazer o backup: {e}")

    finally:
        engine.dispose()
        sqlite_conn.close()
        print("Conexão com o banco de dados encerrada.")


def get_plate_from_database(plate: str) -> tuple[bool, str, str]:
    SQLITE_DB = "../Database/local.db"
    sqlite_conn = sqlite3.connect(SQLITE_DB)
    cursor = sqlite_conn.cursor()
    query = f'SELECT vehicle_log.type, vehicle_log.severity  FROM vehicle_log join vehicle on vehicle_log.vehicle_id = vehicle.id where vehicle.plate = "{plate}"'
    cursor.execute(query)
    car_info = cursor.fetchone()
    cursor.close()

    if car_info is None:
        return (False, None, None)

    type = car_info[0]
    severity = car_info[1]
    return (True, type, severity)


async def backup_if_wifi(wifi_event):
    while True:
        await wifi_event.wait()  # Espera até que o Wi-Fi esteja conectado
        try:
            backup_to_sqlite()  # Tenta fazer o backup
            print("Backup realizado com sucesso!")
        except Exception as e:
            print(f"Erro ao realizar backup: {e}")
        await asyncio.sleep(
            60
        )  # Espera 1 minuto antes de tentar fazer o backup novamente
