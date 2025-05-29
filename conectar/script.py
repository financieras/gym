import gspread

gc = gspread.service_account(filename='credentials.json')
sh = gc.open('hoja_holamundo314159').sheet1

sh.update([['Hola Mundo']], 'A1')
sh.update([[18]], 'D1')

# Usa 'SUM' o 'SUMA' según el idioma de tu hoja
sh.update([['=SUM(A3:B3)']], 'C3')  # O usa '=SUMA(A3:B3)' si tu hoja está en español

edad = int(sh.acell('D1').value)
sh.update([[edad + 1]], 'E1')

print("Operaciones realizadas correctamente.")
