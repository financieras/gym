# gym/conectar/script.py

import gspread
import os
from dotenv import load_dotenv
import json
from google.oauth2.service_account import Credentials

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

# Autentica usando las credenciales
scopes = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]

credentials = Credentials.from_service_account_info(google_creds_dict, scopes=scopes)
gc = gspread.authorize(credentials)

# Abre tu hoja de cálculo
sheet_name = 'hoja_holamundo314159'
try:
    sh = gc.open(sheet_name).sheet1
    print(f"✓ Conectado exitosamente a la hoja: {sheet_name}")
except gspread.exceptions.SpreadsheetNotFound:
    print(f"Error: No se pudo encontrar la hoja de cálculo con el nombre '{sheet_name}'. Verifica el nombre y tus permisos.")
    exit()

# Realizar las operaciones en la hoja
try:
    # Actualizar texto y números (estos funcionan bien)
    sh.update([['Hola Mundo desde .env!']], 'A1')
    sh.update([[18]], 'D1')
    print("✓ Texto y números actualizados")
    
    # Agregar algunos números en A3 y B3 para que la suma tenga sentido
    sh.update([[5]], 'A3')
    sh.update([[10]], 'B3')
    print("✓ Números base agregados en A3 y B3")
    
    # MÉTODO CORRECTO para insertar fórmulas
    # Opción 1: Usando update() con range y values como parámetros separados
    sh.update(
        range_name='C3',
        values=[['=SUM(A3:B3)']],
        value_input_option='USER_ENTERED'
    )
    print("✓ Fórmula SUM insertada en C3")
    
    # Opción 2: Usando update_cell() que es más directo para una celda
    sh.update_cell(4, 3, '=AVERAGE(A3:B3)', value_input_option='USER_ENTERED')
    print("✓ Fórmula AVERAGE insertada en C4")
    
    # Opción 3: Múltiples fórmulas usando batch_update()
    batch_data = [
        {
            'range': 'C5',
            'values': [['=MAX(A3:B3)']]
        },
        {
            'range': 'C6', 
            'values': [['=IF(D1>10,"Mayor que 10","Menor o igual a 10")']]
        }
    ]
    
    sh.batch_update(batch_data, value_input_option='USER_ENTERED')
    print("✓ Fórmulas adicionales insertadas con batch_update")
    
    # Esperar un momento para que las fórmulas se calculen
    import time
    time.sleep(2)
    
    # Verificar que las fórmulas funcionan
    print("\n=== VERIFICACIÓN DE RESULTADOS ===")
    cells_to_check = ['A3', 'B3', 'C3', 'C4', 'C5', 'C6', 'D1']
    
    for cell in cells_to_check:
        try:
            value = sh.acell(cell).value
            print(f"Celda {cell}: {value}")
        except Exception as e:
            print(f"Error al leer {cell}: {e}")
    
    # Actualizar edad como antes
    edad_cell_value = sh.acell('D1').value
    if edad_cell_value is not None:
        try:
            edad = int(edad_cell_value)
            sh.update([[edad + 1]], 'E1')
            print(f"✓ Edad actualizada: {edad} → {edad + 1}")
        except ValueError:
            print(f"Advertencia: El valor en D1 ('{edad_cell_value}') no es un entero válido. No se pudo actualizar E1.")
    else:
        print("Advertencia: La celda D1 está vacía. No se pudo actualizar E1.")
    
    print("\n✓ Todas las operaciones realizadas correctamente.")
    
except Exception as e:
    print(f"Error al realizar operaciones en la hoja: {e}")
    import traceback
    traceback.print_exc()  # Esto nos dará más detalles del error