import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from main import *
from config import *
import random

# Dateiname der CSV-Datei
csv_dateiname = 'zst5651_2021.csv'

# Dateipfad zur CSV-Datei erstellen
csv_dateipfad = os.path.join(os.getcwd(), csv_dateiname)


def read_lkw_data(csv_dateipfad=csv_dateipfad):
    # LKW-Daten aus csv einlesen und relevante Spalten summieren
    df = pd.read_csv(csv_dateipfad, delimiter=';')
    df = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]
    df = df.reset_index(drop=True)

    hours_difference = len(df)

    gewuenschte_spalten = ['Datum', 'Stunde', 'LoA_R1', 'Lzg_R1', 'LoA_R2', 'Lzg_R2', 'Sat_R1', 'Sat_R2']
    df = df[gewuenschte_spalten]
    df['gesamt_LKW_R1'] = df['LoA_R1'] + df['Lzg_R1'] + df['Sat_R1']
    df['gesamt_LKW_R2'] = df['LoA_R2'] + df['Lzg_R2'] + df['Sat_R2']

    # Daten für Säulendiagramm generieren (Breite: 1h, Höhe: stundenscharfer durchschnittlicher LKW-Verkehr in LKW/h)
    x_values = np.arange(hours_difference)
    y_values = df['gesamt_LKW_R1'].to_numpy() * verkehrssteigerung

    # Polynomapproximation
    # lokale Extrema anpassen um Polynom-Approximation zu verbessern
    y_values_angepasst = anpassen_liste(y_values)
    # approximierten Funktion soll durch Mitte der Säulen verlaufen
    x_values_angepasst = x_values + 0.5
    # Durchführen der Approximation
    spline = CubicSpline(x_values_angepasst, y_values_angepasst)
    x_continuous = np.linspace(0, hours_difference, int(hours_difference*60/timedelta))
    y_continuous = spline(x_continuous)

    # Zuordnen der LKW zu jedem timestep
    lkws_in_timesteps = []
    summe_in_timesteps = []
    summe = 0
    summe_lkws = 0
    for index, i in enumerate(y_continuous):
        minute = (index * timedelta) % 1440

        summe += i * timedelta/60
        wert = math.floor(summe)

        lkw_in_timestep = [wert, minute]
        lkws_in_timesteps.append(lkw_in_timestep)
        summe_lkws += wert
        summe = summe - wert
        summe_in_timesteps.append(summe)

    # Auslesen der Differenzen zwischen Ausgangsdaten und approximierten Daten (stündlich)
    summen_liste = []
    for i in range(0, len(lkws_in_timesteps), int(60 / timedelta)):
        # Summiere die ersten Werte aller Arrays im ausgewählten Bereich
        sum_of_values = sum(array[0] for array in lkws_in_timesteps[i:i + int(60 / timedelta)])
        summen_liste.append(sum_of_values)
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

    verkehrsdaten = {'x_values': x_values, 'y_values': y_values, 'x_continuous': x_continuous, 'y_continuous': y_continuous, 'differenzen': differenzen}

    return lkws_in_timesteps, verkehrsdaten

def read_LKW_probability_data():
    working_directory = os.getcwd()
    file_path = os.path.join(working_directory, 'Abbiegewahrscheinlichkeiten.xlsx')
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Fehler beim Lesen der Abbiegewahrscheinlichkeiten.xlsx Datei: {e}")
        return None

def generate_bev_lkw_data(lkws_in_timesteps, probability_df):
    df_ankommende_bev_lkws_anzahl = create_dataframe_with_dimensions(num_rows=len(lkws_in_timesteps), anzahl_ladesäulen_typ={'NCS': 1, 'HPC': 0})
    df_ankommende_bev_lkws_anzahl.rename(columns={'Ladesäule 1 NCS': 'Ankommende LKWs'}, inplace=True)

    df_ankommende_bev_lkws = df_ankommende_bev_lkws_anzahl.copy()

    summe_bev_gesamt = 0
    for index, value in enumerate(df_ankommende_bev_lkws_anzahl['Ankommende LKWs']):
        tageszeit = index*timedelta % 1440
        abbiegewahrscheinlichkeiten_tageszeit = probability_df.loc[probability_df['Tageszeit'] == tageszeit]
        abbiegewahrscheinlichkeiten_tageszeit = abbiegewahrscheinlichkeiten_tageszeit.drop('Tageszeit', axis=1)

        spalten = abbiegewahrscheinlichkeiten_tageszeit.columns

        # Intervallgrenzen berechnen
        intervall_grenzen_1 = [0] + [abbiegewahrscheinlichkeiten_tageszeit[col].values[0] for col in spalten]
        intervall_grenzen_1 = [sum(intervall_grenzen_1[:i + 1]) for i in range(len(intervall_grenzen_1))]
        intervall_grenzen = [x * anteil_bev for x in intervall_grenzen_1]

        summe_bev_in_timestep = 0.0
        bev_lkws_in_timestep = []
        for lkw in range(lkws_in_timesteps[index][0]):
            zufallszahl = random.random()
            for i in range(len(intervall_grenzen) - 1):
                if intervall_grenzen[i] < zufallszahl <= intervall_grenzen[i + 1]:
                    l = spalten[i]
                    summe_bev_in_timestep += 1
                    bev_lkw = generate_new_lkw(ankommenszeit=lkws_in_timesteps[index][1], l_type=l)
                    bev_lkws_in_timestep.append(bev_lkw)
                    break
        df_ankommende_bev_lkws_anzahl['Ankommende LKWs'].iloc[index] = summe_bev_in_timestep
        df_ankommende_bev_lkws['Ankommende LKWs'].iloc[index] = bev_lkws_in_timestep
        summe_bev_gesamt += summe_bev_in_timestep

    return df_ankommende_bev_lkws_anzahl, df_ankommende_bev_lkws, summe_bev_gesamt

def generate_new_lkw(ankommenszeit, l_type):
    akkustand = random.randint(5, 30)
    intervall_dict = {}
    kum_summe = 0
    kapazität = 20
    for name, wert in verteilung_kapazitäten.items():
        intervall_dict[name] = [kum_summe]
        kum_summe += wert
        intervall_dict[name].append(kum_summe)
    random_number = random.random()
    for name in verteilung_kapazitäten.keys():
        if intervall_dict[name][0] <= random_number <= intervall_dict[name][1]:
            kapazität = name

    return [akkustand, kapazität, l_type, 0, 0, 0]


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

def run_simulation():
    lkws_in_timesteps, verkehrsdaten = read_lkw_data(csv_dateipfad=csv_dateipfad)
    probability_df = read_LKW_probability_data()
    gesamt_df = pd.DataFrame()
    df_gesamt_anzahl = pd.DataFrame()
    for i in range(anzahl_simulationen):
        bev_lkws_in_timesteps_anzahl, bev_lkws_in_timesteps ,summe_bev_gesamt = generate_bev_lkw_data(lkws_in_timesteps=lkws_in_timesteps, probability_df=probability_df)
        gesamt_df[f'Run_{i}'] = bev_lkws_in_timesteps['Ankommende LKWs']
        df_gesamt_anzahl[f'Run_{i}'] = bev_lkws_in_timesteps_anzahl['Ankommende LKWs']
        df_gesamt_anzahl[f'Run_{i}'] = pd.to_numeric(df_gesamt_anzahl[f'Run_{i}'], errors='coerce')
        print(f'Generiere Inputdaten {i + 1}/{anzahl_simulationen}')
        #print(f'Summe BEV gesamt: {summe_bev_gesamt}')
    return gesamt_df, df_gesamt_anzahl ,verkehrsdaten




