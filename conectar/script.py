# gym/conectar/script.py

import gspread
import os
from dotenv import load_dotenv
import json

# Construye la ruta al archivo .env en el mismo directorio que este script
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

# Carga el contenido JSON de las credenciales desde la variable de entorno
google_creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')

if not google_creds_json_str:
    raise ValueError("La variable GOOGLE_APPLICATION_CREDENTIALS_JSON no se encontró. Asegúrate de que tu archivo .env está configurado correctamente en gym/conectar/.env")

try:
    # Convierte la cadena JSON a un diccionario
    google_creds_dict = json.loads(google_creds_json_str)
except json.JSONDecodeError as e:
    raise ValueError(f"Error al decodificar GOOGLE_APPLICATION_CREDENTIALS_JSON: {e}. Verifica el formato del JSON en tu archivo .env.")

# Autentica usando el diccionario de credenciales
gc = gspread.service_account_from_dict(google_creds_dict)

# Abre tu hoja de cálculo (asegúrate de que el nombre es correcto)
# Considera también guardar el nombre de la hoja en el .env si cambia entre entornos
sheet_name = 'hoja_holamundo314159'
try:
    sh = gc.open(sheet_name).sheet1
except gspread.exceptions.SpreadsheetNotFound:
    print(f"Error: No se pudo encontrar la hoja de cálculo con el nombre '{sheet_name}'. Verifica el nombre y tus permisos.")
    exit()


sh.update([['Hola Mundo desde .env!']], 'A1') # Actualizado para ver el cambio
sh.update([[18]], 'D1')

# Usa 'SUM' o 'SUMA' según el idioma de tu hoja
# Podrías poner la fórmula en el .env si necesitas que sea configurable
sh.update([['=SUM(A3:B3)']], 'C3')  # O usa '=SUMA(A3:B3)' si tu hoja está en español

edad_cell_value = sh.acell('D1').value
if edad_cell_value is not None: # Verifica que la celda no esté vacía
    try:
        edad = int(edad_cell_value)
        sh.update([[edad + 1]], 'E1')
    except ValueError:
        print(f"Advertencia: El valor en D1 ('{edad_cell_value}') no es un entero válido. No se pudo actualizar E1.")
else:
    print("Advertencia: La celda D1 está vacía. No se pudo actualizar E1.")

print("Operaciones realizadas correctamente.")