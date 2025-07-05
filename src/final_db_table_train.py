import sqlite3
import os
import pandas as pd
from datetime import datetime, timedelta

raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_path = os.path.join(raiz,"database.db")
table_train = "santander_train_data"

processed_data_path = os.path.join(raiz, "data", "processed")

# Función 1

def load_last_n_months_data(db_path: str, table_name: str, num_months: int = 3) -> pd.DataFrame:
    """
    Carga los datos de n meses a la base de datos
    """

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(f"SELECT MAX(fecha_dato) FROM {table_name}")
        max_date_str = cursor.fetchone()[0]

    
        if not max_date_str:
            print(f"Error: No se encontró la fecha máxima en la tabla '{table_name}'. La tabla podría estar vacía o la columna 'fecha_dato' no existe/no tiene datos.")
            return pd.DataFrame()
        
        max_date = datetime.strptime(max_date_str, '%Y-%m-%d')
        print(f"Fecha más reciente encontrada: {max_date.strftime('%Y-%m-%d')}")

        start_date = max_date - timedelta(days=30 * num_months)
        start_date_str = start_date.strftime('%Y-%m-%d')
        print(f"Calculando inicio del período de {num_months} meses desde: {start_date_str}")

        query = f"SELECT * FROM {table_name} WHERE fecha_dato >= '{start_date_str}'"
        df = pd.read_sql_query(query, conn)


        print(f"Datos de los últimos {num_months} meses cargados. Dimensiones: {df.shape}")
        return df
    except sqlite3.Error as e:
        print(f"Error de SQLite al cargar datos: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Ocurrió un error inesperado al cargar datos: {e}")
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()
            print("Conexión a la base de datos cerrada.")


# Función 2

def create_product_targets_and_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una variable objetivo binaria por cada producto (1 si el cliente adquirió
    ese producto en el mes, 0 si no) y también crea características de tenencia
    de productos del mes anterior.
    """


    df['fecha_dato'] = pd.to_datetime(df['fecha_dato'], format='%Y-%m-%d')
    df['ncodpers'] = df['ncodpers'].astype(int)

    df = df.sort_values(by=['ncodpers', 'fecha_dato'])

    product_cols = [col for col in df.columns if col.startswith('ind_') and col.endswith('_ult1')]
    product_cols.extend(['ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'])
    product_cols = list(set(product_cols))
    product_cols = [col for col in product_cols if col in df.columns]

    print(f"Columnas de producto identificadas ({len(product_cols)}): {product_cols[:5]}...")

    df[product_cols] = df[product_cols].fillna(0.0).astype(int)

    print("Creando características de tenencia de productos del mes anterior...")
    df_lagged_products = df.groupby('ncodpers')[product_cols].shift(1).fillna(0).astype(int)
    df_lagged_products.columns = [f"{col}_prev" for col in product_cols]
    df = pd.concat([df, df_lagged_products], axis=1)

    print("Creando variables objetivo de adquisición de nuevos productos...")
    df_new_products = df.groupby('ncodpers')[product_cols].diff()
    for col in product_cols:
        df[f'target_{col}_acquired'] = (df_new_products[col] == 1).astype(int)
    
    df['target_any_new_product'] = (df_new_products > 0).any(axis=1).astype(int)
    print(f"Variable objetivo general 'target_any_new_product' creada:")
    print(df['target_any_new_product'].value_counts())

    initial_customer_records = df.groupby('ncodpers')['fecha_dato'].transform('min') == df['fecha_dato']
    print(f"Filas iniciales de clientes (primer registro de cada cliente): {initial_customer_records.sum()}")
    df_filtered = df[~initial_customer_records].copy()
    print(f"DataFrame después de excluir el primer registro de cada cliente: {df_filtered.shape}")

    return df_filtered



# Flujo del archivo

if __name__ == "__main__":
    print("--- Iniciando Proceso de Creación de Tabla Final en DB ---")

    # 1. Cargar el DataFrame del último semestre (6 meses)
    df_raw_data = load_last_n_months_data(database_path, table_train, num_months=3)

    if not df_raw_data.empty:
        # 2. Filtrar por actividad del cliente (ind_actividad_cliente = 1)
        initial_rows = df_raw_data.shape[0]
        df_active_customers = df_raw_data[df_raw_data['ind_actividad_cliente'] == 1].copy()
        print(f"Filas antes del filtro: {initial_rows}, Filas después del filtro: {df_active_customers.shape[0]}")

        if not df_active_customers.empty:
            # 3. Crear las variables objetivo y las características rezagadas de productos
            df_processed_features = create_product_targets_and_lagged_features(df_active_customers.copy())

            if not df_processed_features.empty:
                print(f"DataFrame procesado. Tamaño inicial: {df_processed_features.shape[0]} filas, {df_processed_features.shape[1]} columnas.")

                # --- 4. Seleccionar solo las columnas deseadas para la tabla final ---
                print("\n--- Seleccionando columnas finales para el dataset reducido en DB ---")

                id_date_cols = ['ncodpers', 'fecha_dato']

                core_predictor_cols = [
                    "ind_empleado", "pais_residencia", "sexo", "age", "fecha_alta",
                    "ind_nuevo", "antiguedad", "indrel1", "indrel_1mes", "tiprel_1mes",
                    "indresi", "indext", "conyuemp", "canal_entrada", "indfall",
                    "cod_prov", "nomprov", "ind_actividad_cliente", "renta", "segmento"
                ]
                core_predictor_cols = [col for col in core_predictor_cols if col in df_processed_features.columns]
                print(f"Columnas predictoras demográficas/relacionales seleccionadas ({len(core_predictor_cols)}): {core_predictor_cols[:5]}...")

                all_target_acquired_cols = [col for col in df_processed_features.columns if col.startswith('target_ind_') and col.endswith('_acquired')]
                print(f"Columnas de adquisición de productos (targets) seleccionadas ({len(all_target_acquired_cols)}): {all_target_acquired_cols[:5]}...")

                all_prev_product_cols = [col for col in df_processed_features.columns if col.endswith('_prev')]
                print(f"Columnas de tenencia de productos anteriores seleccionadas ({len(all_prev_product_cols)}): {all_prev_product_cols[:5]}...")

                any_new_product_target = ['target_any_new_product']
                print(f"Columna 'target_any_new_product' incluida.")

                final_columns_to_keep = id_date_cols + core_predictor_cols + all_target_acquired_cols + all_prev_product_cols + any_new_product_target
                
                df_final_db_table = df_processed_features[final_columns_to_keep].copy()
                
                print(f"DataFrame filtrado. Tamaño final para DB: {df_final_db_table.shape[0]} filas, {df_final_db_table.shape[1]} columnas.")


                # --- 5. y 6. Eliminar tabla existente y guardar el nuevo DataFrame en DB ---
                print(f"\n--- Actualizando la tabla '{table_train}' en la base de datos ---")
                conn = None
                try:
                    conn = sqlite3.connect(database_path)
                    cursor = conn.cursor()

                    print(f"Eliminando la tabla existente '{table_train}'...")
                    cursor.execute(f"DROP TABLE IF EXISTS {table_train}")
                    print(f"Tabla '{table_train}' eliminada exitosamente.")

                    print(f"Guardando el DataFrame reducido y procesado ({df_final_db_table.shape[0]} filas) como nueva tabla '{table_train}'...")
                    df_final_db_table.to_sql(table_train, conn, if_exists='replace', index=False)
                    print(f"Tabla '{table_train}' actualizada en la DB con el nuevo conjunto de datos reducido y filtrado.")

                    cursor.execute(f"PRAGMA table_info({table_train})")
                    cols_in_db = cursor.fetchall()
                    print(f"Verificación: La tabla '{table_train}' ahora tiene {len(cols_in_db)} columnas en la DB.")
                    cursor.execute(f"SELECT COUNT(*) FROM {table_train}")
                    final_db_row_count = cursor.fetchone()[0]
                    print(f"Verificación: La tabla '{table_train}' ahora tiene {final_db_row_count} filas en la DB.")

                except sqlite3.Error as e:
                    print(f"Error de SQLite durante la actualización de la tabla: {e}")
                except Exception as e:
                    print(f"Ocurrió un error inesperado durante la actualización de la tabla: {e}")
                finally:
                    if conn:
                        conn.close()
                        print("Conexión a la base de datos cerrada después de la actualización.")

            else:
                print("Fallo en el procesamiento de características. No se generó un DataFrame para guardar.")
        else:
            print("No hay clientes activos en el período seleccionado. Abortando creación de tabla.")
    else:
        print("No se pudo cargar el DataFrame de entrenamiento del último semestre. Abortando creación de tabla.")

    print("\n--- Fin del Proceso de Creación de Tabla Final en DB ---")
