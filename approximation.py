import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
import pandas as pd
import os

# Dateiname der CSV-Datei
csv_dateiname = 'zst5651_2021.csv'

# Dateipfad zur CSV-Datei erstellen
csv_dateipfad = os.path.join(os.getcwd(), csv_dateiname)

def read_lkw_data(csv_dateipfad=csv_dateipfad):

    # DataFrame aus der CSV-Datei erstellen (Semikolon als Trennzeichen angeben)
    df = pd.read_csv(csv_dateipfad, delimiter=';')

    # Nur die gewünschten Spalten behalten
    gewuenschte_spalten = ['Datum', 'Stunde', 'LoA_R1', 'Lzg_R1', 'LoA_R2', 'Lzg_R2', 'Sat_R1', 'Sat_R2']
    df = df[gewuenschte_spalten]

    # Spalten 'LoA_R1' und 'Lzg_R1' addieren und Ergebnisse in einer neuen Spalte 'gesamt_LKW_R1' speichern
    df['gesamt_LKW_R1'] = df['LoA_R1'] + df['Lzg_R1'] + df['Sat_R1']

    # Spalten 'LoA_R2' und 'Lzg_R2' addieren und Ergebnisse in einer neuen Spalte 'gesamt_LKW_R2' speichern
    df['gesamt_LKW_R2'] = df['LoA_R2'] + df['Lzg_R2'] + df['Sat_R2']

    x_values = np.arange(8760)
    y_values = df['gesamt_LKW_R1'].to_numpy()

    # Erstellen Sie eine Kubische-Spline-Interpolation
    spline = CubicSpline(x_values, y_values)

    # Generieren Sie Werte für die stetige Funktion
    x_continuous = np.linspace(0, 8759, 105120)  # Mehr Punkte für eine glattere Kurve
    y_continuous = spline(x_continuous)

    # 1. Summe aller y_values
    sum_y_values = np.sum(y_values)
    print(f"Summe aller y_values: {sum_y_values}")

    lkws_in_timestep = []
    summe_in_timestep = []
    summe = 0
    summe_lkws = 0
    for i in y_continuous:
        summe += i*1/12
        wert = math.floor(summe)
        lkws_in_timestep.append(wert)
        summe_lkws += wert
        summe = summe - wert
        summe_in_timestep.append(summe)
    print(f"Summe aller LKWs: {summe_lkws}")
    # Plot der Original-Stufenfunktion und der approximierten stetigen Funktion
    plt.step(x_values, y_values, where='mid', label='Stufenfunktion')
    plt.plot(x_continuous, y_continuous, label='Approximierte Funktion')
    plt.legend()
    plt.show()
    return lkws_in_timestep

df = read_lkw_data()
dummy = 0



