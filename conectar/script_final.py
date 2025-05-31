# gym/conectar/script_final.py
# Versión final sin warnings

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

# Operaciones básicas usando la sintaxis correcta
sh.update(values=[['Hola Mundo desde .env!']], range_name='A1')
sh.update(values=[[18]], range_name='D1')

# Datos para las fórmulas
sh.update(values=[[5]], range_name='A3')
sh.update(values=[[10]], range_name='B3')

print("✓ Datos básicos insertados")

# FÓRMULAS - Sintaxis correcta sin warnings
sh.update(values=[['=SUM(A3:B3)']], range_name='C3', value_input_option='USER_ENTERED')
sh.update(values=[['=AVERAGE(A3:B3)']], range_name='C4', value_input_option='USER_ENTERED')
sh.update(values=[['=MAX(A3:B3)']], range_name='C5', value_input_option='USER_ENTERED')
sh.update(values=[['=MIN(A3:B3)']], range_name='C6', value_input_option='USER_ENTERED')

print("✓ Fórmulas insertadas correctamente")

# Esperar un momento para que se calculen las fórmulas
import time
time.sleep(1)

# Verificar resultados
print("\n=== RESULTADOS ===")
print(f"A3 (primer número): {sh.acell('A3').value}")
print(f"B3 (segundo número): {sh.acell('B3').value}")
print(f"C3 (suma): {sh.acell('C3').value}")
print(f"C4 (promedio): {sh.acell('C4').value}")
print(f"C5 (máximo): {sh.acell('C5').value}")
print(f"C6 (mínimo): {sh.acell('C6').value}")

# Actualizar edad
edad = int(sh.acell('D1').value)
sh.update(values=[[edad + 1]], range_name='E1')

print(f"\n✓ Edad actualizada: {edad} → {edad + 1}")

# Agregar algunos ejemplos adicionales de fórmulas útiles
print("\n=== FÓRMULAS ADICIONALES ===")

# Fórmula con fecha
sh.update(values=[['=TODAY()']], range_name='F1', value_input_option='USER_ENTERED')

# Fórmula condicional
sh.update(values=[['=IF(D1>=18,"Adulto","Menor")']], range_name='F2', value_input_option='USER_ENTERED')

# Fórmula de concatenación
sh.update(values=[['=CONCATENATE("La suma es: ",C3)']], range_name='F3', value_input_option='USER_ENTERED')

# Verificar las nuevas fórmulas
time.sleep(1)
print(f"F1 (fecha actual): {sh.acell('F1').value}")
print(f"F2 (verificación edad): {sh.acell('F2').value}")
print(f"F3 (texto + suma): {sh.acell('F3').value}")

print("\n✅ Script completado exitosamente")
print("🎉 ¡Todas las fórmulas funcionan correctamente!")