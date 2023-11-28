import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from main import *

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

    y_values_angepasst = anpassen_liste(y_values)
    # Erstellen Sie eine Kubische-Spline-Interpolation
    x_values_angepasst = x_values + 0.5
    spline = CubicSpline(x_values_angepasst, y_values_angepasst)

    # Generieren Sie Werte für die stetige Funktion
    x_continuous = np.linspace(0, 8760, int(8760*60/timedelta))
    y_continuous = spline(x_continuous)


    # 1. Summe aller y_values
    sum_y_values = np.sum(y_values)
    print(f"Summe aller y_values: {sum_y_values}")

    lkws_in_timestep = []
    summe_in_timestep = []
    summe = 0
    summe_lkws = 0
    for index, i in enumerate(y_continuous):
        summe += i * timedelta/60
        wert = math.floor(summe)
        lkws_in_timestep.append(wert)
        summe_lkws += wert
        summe = summe - wert
        summe_in_timestep.append(summe)

    summen_liste = []
    for i in range(0, len(lkws_in_timestep), int(60/timedelta)):
        # Summiere die nächsten 12 Werte und füge die Summe zur neuen Liste hinzu
        summen_liste.append(sum(lkws_in_timestep[i:i + int(60/timedelta)]))

    differenzen = [abs(a - b) for a, b in zip(y_values, summen_liste)]

    anzahl_um_3 = sum(diff > 10 for diff in differenzen)
    anzahl_um_5 = sum(diff > 50 for diff in differenzen)
    anzahl_um_10 = sum(diff > 100 for diff in differenzen)
    anzahl_groesser_10 = sum(diff > 200 for diff in differenzen)

    print(f"Differenzen größer als 10: {anzahl_um_3} Mal")
    print(f"Differenzen größer als 50: {anzahl_um_5} Mal")
    print(f"Differenzen größer als 100: {anzahl_um_10} Mal")
    print(f"Differenzen größer als 200: {anzahl_groesser_10} Mal")


    print(f"Summe aller LKWs: {summe_lkws}")
    # Plot der Original-Stufenfunktion und der approximierten stetigen Funktion
    plt.step(x_values, y_values, label='Stufenfunktion', where='post')
    plt.plot(x_continuous, y_continuous, label='Approximierte Funktion')
    plt.step(x_continuous, y_continuous, label='Approximierte Stufenfunktion', where='post')


    # for i, diff in enumerate(differenzen):
    #     if diff > 20:
    #         plt.scatter(x_values[i], y_values[i], color='red')

    plt.legend()
    plt.show()

    # df_ankommende_lkws_anzahl = create_dataframe_with_dimensions(num_rows=len(lkws_in_timestep), num_columns=1,
    #                                                              anzahl_ncs=1, anzahl_hpc=0)
    # df_ankommende_lkws_anzahl.rename(columns={'Ladesäule 1 NCS': 'Ankommende LKWs'}, inplace=True)
    #
    # for index, value in enumerate(df_ankommende_lkws_anzahl['Ankommende LKWs']):
    #     df_ankommende_lkws_anzahl['Ankommende LKWs'].iloc[index] = lkws_in_timestep[index]

    return lkws_in_timestep

def generate_new_lkw(ankommenszeit):
    return 0

def kumulierte_werte(liste):
    kumulierte_liste = []
    kumulierte_summe = 0

    for wert in liste:
        kumulierte_summe += wert
        kumulierte_liste.append(kumulierte_summe)

    return kumulierte_liste

def anpassen_liste(lst):
    if len(lst) < 3:
        return lst  # Die Liste sollte mindestens drei Elemente enthalten, um Vor- und Nachgänger zu überprüfen

    angepasste_liste = [lst[0]]  # Das erste Element bleibt unverändert

    for i in range(1, len(lst)-1):
        if lst[i] > lst[i-1] and lst[i] > lst[i+1]:
            mittlere_abweichung = (lst[i-1] - 2 * lst[i] + lst[i+1]) / 2
            angepasste_liste.append(lst[i] - 0.15 * mittlere_abweichung)
        elif lst[i] < lst[i-1] and lst[i] < lst[i+1]:
            mittlere_abweichung = (lst[i-1] - 2 * lst[i] + lst[i+1]) / 2
            angepasste_liste.append(lst[i] - 0.15 * mittlere_abweichung)
        else:
            angepasste_liste.append(lst[i])

    angepasste_liste.append(lst[-1])  # Das letzte Element bleibt unverändert

    return angepasste_liste

read_lkw_data(csv_dateipfad=csv_dateipfad)
#df = read_lkw_data()
#print(df)
dummy = 0



