import pandas as pd
import sqlite3
import os
import zipfile
from dotenv import load_dotenv

# 0. Definiendo las rutas

raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_raw_dir = os.path.join(raiz, 'data', 'raw')

main_zip_file_name = 'santander-product-recommendation.zip'
main_zip_file_path = os.path.join(data_raw_dir, main_zip_file_name)

extracted_data_dir = os.path.join(raiz, 'data', 'raw')

nested_zip_names = [
    'sample_submission.csv.zip',
    'test_ver2.csv.zip',
    'train_ver2.csv.zip'
]


if not os.path.exists(main_zip_file_path):
    print(f"Error: El archivo ZIP principal '{main_zip_file_path}' no fue encontrado.")
    print("Asegúrate de haberlo descargado de Kaggle y que esté en la raíz de tu proyecto.")
else:
    # 2. Crear el directorio final de destino si no existe
    os.makedirs(extracted_data_dir, exist_ok=True)

    # 3. Directorio temporal para los ZIPs anidados
    temp_extract_dir = os.path.join(raiz, 'temp_zip_extract')
    os.makedirs(temp_extract_dir, exist_ok=True)
    print(f"Creando directorio temporal para ZIPs anidados: {temp_extract_dir}")

    try:
        # Extraer los ZIPs anidados del ZIP principal
        print(f"\nExtrayendo ZIPs anidados de '{main_zip_file_name}'...")
        with zipfile.ZipFile(main_zip_file_path, 'r') as main_zip_ref:

            for nested_zip_name in nested_zip_names:
                if nested_zip_name in main_zip_ref.namelist():
                    print(f"  - Extrayendo '{nested_zip_name}' a '{temp_extract_dir}'...")
                    main_zip_ref.extract(nested_zip_name, path=temp_extract_dir)
                else:
                    print(f"  - Advertencia: '{nested_zip_name}' no se encontró dentro de '{main_zip_file_name}'.")

        # Descomprimir cada ZIP anidado para obtener los CSVs
        print(f"\nDescomprimiendo cada ZIP anidado a '{extracted_data_dir}'...")
        for nested_zip_name in nested_zip_names:
            nested_zip_path = os.path.join(temp_extract_dir, nested_zip_name)

            if os.path.exists(nested_zip_path):
                csv_name = nested_zip_name.replace('.zip', '')
                print(f"  - Descomprimiendo '{nested_zip_name}' para obtener '{csv_name}'...")
                with zipfile.ZipFile(nested_zip_path, 'r') as nested_zip_ref:
                    nested_zip_ref.extract(csv_name, path=extracted_data_dir)
                    print(f"    '{csv_name}' extraído con éxito.")
            else:
                print(f"  - Error: El ZIP anidado '{nested_zip_path}' no se encontró. ¿Falló la primera etapa?")

    except zipfile.BadZipFile:
        print(f"Error: El archivo '{main_zip_file_path}' no es un ZIP válido o está corrupto.")
    except Exception as e:
        print(f"Ocurrió un error inesperado durante la descompresión: {e}")
    finally:
        if os.path.exists(temp_extract_dir):
            print(f"\nLimpiando directorio temporal: {temp_extract_dir}")
            for item in os.listdir(temp_extract_dir):
                item_path = os.path.join(temp_extract_dir, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(temp_extract_dir)
            print("Directorio temporal limpiado.")

print("\nProceso de descompresión de múltiples niveles finalizado.")

# Para verificar los archivos descomprimidos en data/raw
print("\nArchivos resultantes en 'data/raw':")
if os.path.exists(extracted_data_dir):
    for f in os.listdir(extracted_data_dir):
        if f.endswith('.csv'):
            print(f"- {f}")