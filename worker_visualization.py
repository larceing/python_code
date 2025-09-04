###EL CODIGO SE DIVIDE EN 2 PARTES, GRÁFICO Y LEYENDA###

# El código siguiente, que crea un dataframe y quita las filas duplicadas, siempre se ejecuta y actúa como un preámbulo del script: 
# dataset = pandas.DataFrame(Alias, Año, Fecha, Tipo)
# dataset = dataset.drop_duplicates()

# Pegue o escriba aquí el código de script:

import pandas as pd
import matplotlib.pyplot as plt
import datetime



# Mostrar mensaje si no hay datos
if dataset.empty:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, 'No hay datos para mostrar', ha='center', va='center', fontsize=12)
    ax.axis('off')
    plt.show()
else:
    # Convertir Fecha a datetime
    dataset['Fecha'] = pd.to_datetime(dataset['Fecha'])

    # Clasificar fines de semana si están marcados como trabajo
    dataset['Tipo'] = dataset.apply(
        lambda row: 'Fin de Semana' if row['Fecha'].weekday() >= 5 and row['Tipo'] == 'Trabajo' else row['Tipo'],
        axis=1
    )

    # Ordenar por trabajador y fecha
    dataset = dataset.sort_values(by=["Alias", "Fecha"])

    # Colores para cada tipo
    colores = {
        "Trabajo": "#B3CFFF",
        "Vacaciones": "#FFB3B3",
        "Festivo": "#D9D9D9",
        "Fin de Semana": "#C9E2B3",
        "Fuera oficina": "#FFF2B3" 
    }

    # Crear gráfico
    fig, ax = plt.subplots(figsize=(14, len(dataset['Alias'].unique()) * 0.4))

    for i, trabajador in enumerate(dataset['Alias'].unique()):
        datos_trabajador = dataset[dataset['Alias'] == trabajador]
        for _, fila in datos_trabajador.iterrows():
            ax.barh(
                y=i,
                width=1,
                left=fila['Fecha'],
                color=colores.get(fila['Tipo'], "#FFFFFF"),
                edgecolor='black'
            )

    # Dibujar línea de hoy SOLO si está dentro del rango visible
    hoy = pd.to_datetime(datetime.date.today())
    if dataset['Fecha'].min() <= hoy <= dataset['Fecha'].max():
        ax.axvline(x=hoy, color='red', linestyle='--', linewidth=2)

    # Ajustes visuales
    ax.set_yticks(range(len(dataset['Alias'].unique())))
    ax.set_yticklabels(dataset['Alias'].unique())
    ax.invert_yaxis()
    ax.set_xlabel("Fecha")
    ax.set_title("Seguimiento diario por trabajador")

    plt.tight_layout()
    plt.show()



### Leyenda


# El código siguiente, que crea un dataframe y quita las filas duplicadas, siempre se ejecuta y actúa como un preámbulo del script: 
# dataset = pandas.DataFrame(Tipo)
# dataset = dataset.drop_duplicates()

# Pegue o escriba aquí el código de script:
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Definimos los colores (igual que en el gráfico principal)
colores = {
    "Trabajo": "#B3CFFF",
    "Vacaciones": "#FFB3B3",
    "Festivo": "#D9D9D9",
    "Fin de Semana": "#C9E2B3",
    "Fuera oficina": "#FFF2B3",
    "Hoy": "#FF0000"
}

# Crear figura y leyenda
fig, ax = plt.subplots(figsize=(3, 2))
leyenda = [mpatches.Patch(color=c, label=t) for t, c in colores.items()]
ax.legend(handles=leyenda, loc="center", frameon=False, fontsize=14)
ax.axis("off")
plt.tight_layout()
plt.show()
