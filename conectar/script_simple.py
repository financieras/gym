# gym/conectar/script_simple.py
# Versión simplificada y probada

import gspread
import os
from dotenv import load_dotenv
import json
from google.oauth2.service_account import Credentials

# Cargar configuración
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)

google_creds_json_str = os.getenv('GOOGLE_APPLICATION_CREDENTIALS_JSON')
if not google_creds_json_str:
    raise ValueError("Variable GOOGLE_APPLICATION_CREDENTIALS_JSON no encontrada")

google_creds_dict = json.loads(google_creds_json_str)

# Conectar
scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scopes)
gc = gspread.authorize(credentials)
sh = gc.open('hoja_holamundo314159').sheet1

print("✓ Conectado exitosamente")

# Operaciones básicas
sh.update([['Operaciones desde Python']], 'A1')
sh.update([[18]], 'D1')

# Datos para la suma
sh.update([[5]], 'A3')
sh.update([[10]], 'B3')

# FÓRMULAS - Método compatible con todas las versiones
# Opción 1: Usando update() con rango específico
sh.update('C3:C3', [['=SUM(A3:B3)']], value_input_option='USER_ENTERED')
sh.update('D3:D3', [['=AVERAGE(A3:B3)']], value_input_option='USER_ENTERED')

print("✓ Fórmulas insertadas")

# Verificar resultados
import time
time.sleep(1)

print(f"A3: {sh.acell('A3').value}")
print(f"B3: {sh.acell('B3').value}")
print(f"C3 (suma): {sh.acell('C3').value}")
print(f"D3 (promedio): {sh.acell('D3').value}")

# Actualizar edad
edad = int(sh.acell('D1').value)
sh.update([[edad + 1]], 'E1')

print(f"✓ Edad: {edad} → {edad + 1}")
print("✓ Operaciones completadas")