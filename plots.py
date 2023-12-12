import pandas as pd
import matplotlib.pyplot as plt
import os
from config import *

show_plots = True
save_plots = False
folder = 'OUTPUT'

def get_scenario_name():
    scenario_name = f"{start_date}_{end_date}_{anzahl_simulationen}_{int(anteil_bev * 100)}_{int(tankwahrscheinlichkeit * 100)}_{netzanschlussleistung}"
    return scenario_name


def plot_lastgang(gesamt_df_ladeleistung):
    fig, ax = plt.subplots()
    min_values = gesamt_df_ladeleistung.min(axis=1, skipna=True)
    max_values = gesamt_df_ladeleistung.max(axis=1, skipna=True)
    avg_values = gesamt_df_ladeleistung.mean(axis=1, skipna=True)
    plt.plot(gesamt_df_ladeleistung.index, avg_values, label='Durchschnitt')
    plt.fill_between(gesamt_df_ladeleistung.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

    plt.xlabel('Zeit [min]')
    plt.ylabel('Leistung [kW]')
    plt.legend()

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_lastprofil.pdf"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)

    if show_plots:
        plt.show()


def plot_nicht_ladende_LKWs(gesamt_df_nicht_ladende_lkws):
    fig, ax = plt.subplots()
    min_values = gesamt_df_nicht_ladende_lkws.min(axis=1, skipna=True)
    max_values = gesamt_df_nicht_ladende_lkws.max(axis=1, skipna=True)
    avg_values = gesamt_df_nicht_ladende_lkws.mean(axis=1, skipna=True)
    plt.plot(gesamt_df_nicht_ladende_lkws.index, avg_values, label='Durchschnitt')
    plt.fill_between(gesamt_df_nicht_ladende_lkws.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

    plt.xlabel('Zeit [min]')
    plt.ylabel('Anzahl nicht ladender LKWs')
    plt.legend()

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_nicht_ladende_lkws.pdf"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)

    if show_plots:
        plt.show()


def plot_energiemenge(energien_dict):
    average_data = sum([df.values for df in energien_dict.values()]) / len(energien_dict)
    average_df = pd.DataFrame(average_data, index=energien_dict['Run_0'].index, columns=energien_dict['Run_0'].columns)

    fig, ax = plt.subplots()
    average_df.plot(kind='bar', stacked=True, ax=ax, width=0.8, figsize=(10, 6))
    plt.xlabel('Tag')
    plt.ylabel('Energiemenge [kWh]')
    plt.title('durch. Energiemenge über alle Simulationen pro Tag ')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_energien.pdf"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)

    if show_plots:
        plt.show()

def plot_verkehr(verkehrsdaten):
    x_values = verkehrsdaten['x_values']
    y_values = verkehrsdaten['y_values']
    x_continuous = verkehrsdaten['x_continuous']
    y_continuous = verkehrsdaten['y_continuous']
    differenzen = verkehrsdaten['differenzen']

    fig, ax = plt.subplots()
    # Plot der Original-Stufenfunktion und der approximierten stetigen Funktion
    plt.step(x_values, y_values, label='Stufenfunktion', where='post')
    plt.plot(x_continuous, y_continuous, label='Approximierte Funktion')
    plt.step(x_continuous, y_continuous, label='Approximierte Stufenfunktion', where='post')

    # Markieren der Stunden wo sich die Ausgangsdaten und die approximierten Daten um mehr als 20 LKWs unterscheiden
    # for i, diff in enumerate(differenzen):
    #     if diff > 20:
    #         plt.scatter(x_values[i], y_values[i], color='red')
    plt.xlabel('Zeit [min]')
    plt.ylabel('Anzahl der LKWs pro timestep')
    plt.legend()

    scenario_name = get_scenario_name()
    file_name = f"{scenario_name}_verkehr.pdf"
    file_path = os.path.join('INPUT', file_name)
    plt.savefig(file_path)

    if show_plots:
        plt.show()

def plot_bev_anzahl(df_gesamt_anzahl):
    fig, ax = plt.subplots()
    min_values = df_gesamt_anzahl.min(axis=1, skipna=True)
    max_values = df_gesamt_anzahl.max(axis=1, skipna=True)
    avg_values = df_gesamt_anzahl.mean(axis=1, skipna=True)
    median_values = df_gesamt_anzahl.median(axis=1)

    # Plotte die Ergebnisse der Anzahl der BEV-LKW in jedem timestep als Band
    plt.plot(df_gesamt_anzahl.index, avg_values, label='Durchschnitt')
    plt.plot(df_gesamt_anzahl.index, median_values, label='Median', color='orange')

    plt.fill_between(df_gesamt_anzahl.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

    plt.xlabel('Zeit [min]')
    plt.ylabel('Anzahl der ankommenden BEV-LKWs pro timestep')
    plt.legend()

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_bev_anzahl.pdf"
        file_path = os.path.join('INPUT', file_name)
        plt.savefig(file_path)

    if show_plots:
        plt.show()