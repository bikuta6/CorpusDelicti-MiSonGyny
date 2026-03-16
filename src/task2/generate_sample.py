import pandas as pd

# Datos de prueba para Task 1 (Clasificación de Misoginia en Canciones)
data = {
    "text": [
        # --- CLASE 1: MISÓGINA (Letras agresivas o degradantes) ---
        "Ella es mi propiedad y lo tiene que saber\nsi no hace lo que digo la voy a hacer caer.\n"
        * 15,  # Repetido para probar longitud
        "Tú no vales nada sin mi dinero,\neres solo un objeto, un cuero.\n"
        + "Estribillo:\nTe trato como quiero, te tiro en el suelo,\nsi me engañas nena, te quito hasta el pelo.\n"
        * 5,
        "Maldita mujer, te voy a castigar,\npor haber nacido solo para cocinar.\n" * 10,
        # --- CLASE 0: NO MISÓGINA (Letras de amor, desamor o sociales) ---
        "Caminando por la ciudad bajo la lluvia,\nrecordando tus besos y tu alegría.\n"
        + "Estribillo:\nAmor eterno, luz de mi vida,\nsin ti mis días son una herida.\n"
        * 8,
        "La justicia social es nuestro estandarte,\nluchamos unidos en cualquier parte.\n"
        + "Verso 2:\nNo permitiremos más opresión,\nqueremos paz para nuestra nación.\n"
        * 6,
        "Ayer te vi en la plaza con un libro en la mano,\nte saludé de lejos como un hermano.\n"
        + "Fue lindo verte sonreír otra vez,\ndespués de tantos meses de timidez.\n"
        * 12,
    ],
    "label": [1, 1, 1, 0, 0, 0],
}

# Crear DataFrame
df_test = pd.DataFrame(data)

# Guardar en la ruta que espera tu script
import os

os.makedirs("../../data/task2", exist_ok=True)
df_test.to_csv("../../data/task2/train.csv", index=False)

print("Archivo train.csv de prueba creado exitosamente.")
print(f"Total de muestras: {len(df_test)}")
print(f"Distribución de clases:\n{df_test['label'].value_counts()}")
