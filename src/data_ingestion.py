import pandas as pd
import sqlite3
import os

raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

raw_dir = os.path.join(raiz, "data", "raw")
train_path = os.path.join(raw_dir, 'train_ver2.csv')
test_path = os.path.join(raw_dir, 'test_ver2.csv')
sample_path = os.path.join(raw_dir, 'sample_submission.csv')

database_path = os.path.join(raiz, "database.db")

table_train = 'santander_train_data'
table_test = 'santander_test_data'
table_sample = 'santander_sample_submission'

def load_csv_to_sqlite(csv_path: str, table_name: str, db_path: str, if_exists: str = 'replace') -> bool:
    """
    Carga un archivo CSV específico en una tabla de una base de datos SQLite.

    Args:
        csv_path (str): Ruta completa al archivo CSV.
        table_name (str): Nombre de la tabla en la base de datos donde se cargarán los datos.
        db_path (str): Ruta completa al archivo de la base de datos SQLite.
        if_exists (str): Comportamiento si la tabla ya existe ('fail', 'replace', 'append').
                         Por defecto, 'replace'.

    Returns:
        bool: True si la carga fue exitosa, False en caso contrario.
    """

    df = None
    conn = None
    try:
        df = pd.read_csv(csv_path)

        conn = sqlite3.connect(db_path)

        df.to_sql(table_name, conn, if_exists=if_exists, index=False)

        df_from_db = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 3", conn)
        print(df_from_db)
        return True

    except pd.errors.EmptyDataError:
        print(f"Error: El archivo CSV '{csv_path}' está vacío.")
        return False
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la carga a la base de datos: {e}")
        return False
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    print("--- Iniciando proceso de carga de CSVs a SQLite ---")

    # Cargar train_ver2.csv
    if load_csv_to_sqlite(train_path, table_train, database_path, if_exists='replace'):
        print(f"\nDatos de '{os.path.basename(train_path)}' cargados a la DB.")
    else:
        print(f"\nFallo al cargar '{os.path.basename(train_path)}' a la DB.")

    # Cargar test_ver2.csv
    if load_csv_to_sqlite(test_path, table_test, database_path, if_exists='replace'):
        print(f"\nDatos de '{os.path.basename(test_path)}' cargados a la DB.")
    else:
        print(f"\nFallo al cargar '{os.path.basename(test_path)}' a la DB.")

    # Cargar sample_submission.csv
    if load_csv_to_sqlite(sample_path, table_sample, database_path, if_exists='replace'):
        print(f"\nDatos de '{os.path.basename(sample_path)}' cargados a la DB.")
    else:
        print(f"\nFallo al cargar '{os.path.basename(sample_path)}' a la DB.")

    print("\n--- Proceso de carga de CSVs a SQLite completado ---")
