from datetime import datetime, timedelta
import pandas as pd

def generar_tabla_vacaciones(fecha_inicio, fecha_fin, dias_vacaciones, nombre_archivo):
    # Convertir fechas
    inicio = datetime.strptime(fecha_inicio, "%Y-%m-%d")
    fin = datetime.strptime(fecha_fin, "%Y-%m-%d")

    # Días totales del periodo
    dias_periodo = (fin - inicio).days + 1

    # Días necesarios para generar 1 día de vacaciones
    dias_por_vacacion = dias_periodo / dias_vacaciones

    resultados = []

    for i in range(1, dias_vacaciones + 1):
        fecha_generacion = inicio + timedelta(days=dias_por_vacacion * i)
        resultados.append({
            "Día de vacaciones": i,
            "Fecha de generación": fecha_generacion.date()
        })

    # Crear DataFrame
    df = pd.DataFrame(resultados)

    # Exportar a Excel
    df.to_excel(nombre_archivo, index=False)

    print(f"Archivo generado correctamente: {nombre_archivo}")


# =========================
# EJEMPLO DE USO
# =========================
if __name__ == "__main__":
    fecha_inicio = "2025-12-11"
    fecha_fin = "2026-12-10"
    dias_vacaciones = 16
    nombre_archivo = "generacion_vacaciones.xlsx"

    generar_tabla_vacaciones(
        fecha_inicio,
        fecha_fin,
        dias_vacaciones,
        nombre_archivo
    )
