import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from main import *
import random

# Dateiname der CSV-Datei
csv_dateiname = 'zst5651_2021.csv'

# Dateipfad zur CSV-Datei erstellen
csv_dateipfad = os.path.join(os.getcwd(), csv_dateiname)

def read_lkw_data(csv_dateipfad=csv_dateipfad):
    # LKW-Daten aus csv einlesen und relevante Spalten summieren
    df = pd.read_csv(csv_dateipfad, delimiter=';')
    gewuenschte_spalten = ['Datum', 'Stunde', 'LoA_R1', 'Lzg_R1', 'LoA_R2', 'Lzg_R2', 'Sat_R1', 'Sat_R2']
    df = df[gewuenschte_spalten]
    df['gesamt_LKW_R1'] = df['LoA_R1'] + df['Lzg_R1'] + df['Sat_R1']
    df['gesamt_LKW_R2'] = df['LoA_R2'] + df['Lzg_R2'] + df['Sat_R2']

    # Daten für Säulendiagramm generieren (Breite: 1h, Höhe: stundenscharfer durchschnittlicher LKW-Verkehr in LKW/h)
    x_values = np.arange(8760)
    y_values = df['gesamt_LKW_R1'].to_numpy()

    # Polynomapproximation
    # lokale Extrema finden um Polynom-Approximation zu verbessern
    y_values_angepasst = anpassen_liste(y_values)
    # approximierten Funktion soll durch mitte der Säulen verlaufen
    x_values_angepasst = x_values + 0.5
    # Durchführen der Approximation
    spline = CubicSpline(x_values_angepasst, y_values_angepasst)
    x_continuous = np.linspace(0, 8760, int(8760*60/timedelta))
    y_continuous = spline(x_continuous)

    # Zuordnen der LKW zu jedem timestep
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

    # Auslesen der Differenzen zwischen Ausgangsdaten und approximierten Daten (stündlich)
    summen_liste = []
    for i in range(0, len(lkws_in_timestep), int(60/timedelta)):
        summen_liste.append(sum(lkws_in_timestep[i:i + int(60/timedelta)]))
    differenzen = [abs(a - b) for a, b in zip(y_values, summen_liste)]
    anzahl_um_10 = sum(diff > 10 for diff in differenzen)
    anzahl_um_20 = sum(diff > 20 for diff in differenzen)
    anzahl_um_50 = sum(diff > 50 for diff in differenzen)
    print(f"Differenzen größer als 10: {anzahl_um_10} Mal")
    print(f"Differenzen größer als 20: {anzahl_um_20} Mal")
    print(f"Differenzen größer als 50: {anzahl_um_50} Mal")

    # Vergleich der gesamten LKWs in einem Jahr zwischen Ausgangsdaten und approximierten Daten
    sum_y_values = np.sum(y_values)
    print(f"Summe aller LKWs in einem Jahr aus den Ausgangsdaten: {sum_y_values}")
    print(f"Summe aller LKWs in einem Jahr aus den approximierten Daten: {summe_lkws}")

    # Plot der Original-Stufenfunktion und der approximierten stetigen Funktion
    # plt.step(x_values, y_values, label='Stufenfunktion', where='post')
    # plt.plot(x_continuous, y_continuous, label='Approximierte Funktion')
    # plt.step(x_continuous, y_continuous, label='Approximierte Stufenfunktion', where='post')

    # Markieren der Stunden wo sich die Ausgangsdaten und die approximierten Daten um mehr als 20 LKWs unterscheiden
    # for i, diff in enumerate(differenzen):
    #     if diff > 20:
    #         plt.scatter(x_values[i], y_values[i], color='red')

    #plt.legend()
    #plt.show()

    dummy=0

    return lkws_in_timestep


def generate_bev_lkw_data(lkws_in_timestep, probability):
    df_ankommende_bev_lkws_anzahl = create_dataframe_with_dimensions(num_rows=len(lkws_in_timestep), num_columns=1,
                                                                 anzahl_ncs=1, anzahl_hpc=0)
    df_ankommende_bev_lkws_anzahl.rename(columns={'Ladesäule 1 NCS': 'Ankommende LKWs'}, inplace=True)

    summe_bev_gesamt = 0
    for index, value in enumerate(df_ankommende_bev_lkws_anzahl['Ankommende LKWs']):
        summe_bev_in_timestep = 0.0
        for lkw in range(lkws_in_timestep[index]):
            random_number = random.random()  # Generiere eine Zufallszahl zwischen 0 und 1
            if random_number <= probability:
                summe_bev_in_timestep += 1
        df_ankommende_bev_lkws_anzahl['Ankommende LKWs'].iloc[index] = summe_bev_in_timestep
        summe_bev_gesamt += summe_bev_in_timestep

    dummy = 0
    return df_ankommende_bev_lkws_anzahl, summe_bev_gesamt

def generate_new_lkw(ankommenszeit):
    return 0


# Hilfsfunktionen
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

lkws_in_timestep = read_lkw_data(csv_dateipfad=csv_dateipfad)
#bev_lkws_in_timestep, summe_bev_gesamt = generate_bev_lkw_data(lkws_in_timestep=lkws_in_timestep, probability=0.02)

gesamt_df = pd.DataFrame()

# Wiederhole die Funktion 100 Mal und füge die Ergebnisse dem Gesamtdatenrahmen hinzu
for i in range(500):
    bev_lkws_in_timestep, summe_bev_gesamt = generate_bev_lkw_data(lkws_in_timestep=lkws_in_timestep, probability=0.02)
    gesamt_df[f'Run_{i}'] = bev_lkws_in_timestep['Ankommende LKWs']
    gesamt_df[f'Run_{i}'] = pd.to_numeric(gesamt_df[f'Run_{i}'], errors='coerce')
    print(f'Berechne Iteration {i + 1}/100')


min_values = gesamt_df.min(axis=1, skipna=True)
max_values = gesamt_df.max(axis=1, skipna=True)
avg_values = gesamt_df.mean(axis=1, skipna=True)
median_values = gesamt_df.median(axis=1)



# Plotte die Ergebnisse als Band
plt.plot(gesamt_df.index, avg_values, label='Durchschnitt')
plt.plot(gesamt_df.index, median_values, label='Median', color='orange')

plt.fill_between(gesamt_df.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

plt.xlabel('Index des Datenrahmens')
plt.ylabel('Werte')
plt.legend()
plt.show()
dummy =0
#print(f"Summe aller BEV LKWs in einem Jahr aus den approximierten Daten: {summe_bev_gesamt}")
#print(f"BEV Quote: {summe_bev_gesamt/sum(lkws_in_timestep)}")


dummy = 0



