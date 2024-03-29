import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from config import *

show_plots = False
save_plots = True
folder = 'PLOTS'

def get_scenario_name():
    scenario_name = f"{start_date}_{end_date}_{anzahl_simulationen}_{int(anteil_bev * 100)}_{int(tankwahrscheinlichkeit * 100)}_{netzanschlussleistung}"
    return scenario_name


def plot_lastgang(gesamt_df_ladeleistung):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots()
    min_values = gesamt_df_ladeleistung.min(axis=1, skipna=True)
    max_values = gesamt_df_ladeleistung.max(axis=1, skipna=True)
    avg_values = gesamt_df_ladeleistung.mean(axis=1, skipna=True)
    plt.plot(gesamt_df_ladeleistung.index, avg_values, label='Durchschnitt')
    plt.fill_between(gesamt_df_ladeleistung.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

    max_leistung = max_values.max()
    print(f"maximale Leistung: {max_leistung} kW")

    plt.xlabel('Zeit [min]')
    plt.ylabel('Leistung [kW]')
    plt.legend()

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_lastprofil.png"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)

    if show_plots:
        plt.show()


def plot_nicht_ladende_LKWs(gesamt_df_nicht_ladende_lkws):
    if not os.path.exists(folder):
        os.makedirs(folder)
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
    if not os.path.exists(folder):
        os.makedirs(folder)
    average_data = sum([df.values for df in energien_dict.values()]) / len(energien_dict)
    average_df = pd.DataFrame(average_data, index=energien_dict['Run_0'].index, columns=energien_dict['Run_0'].columns)

    gesamte_energiemenge = average_df.values.sum()
    print(f"gesamte Energiemenge: {gesamte_energiemenge} kWh")

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
    if not os.path.exists(folder):
        os.makedirs(folder)
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
    #plt.xlim(0,24)
    #plt.ylim(0,720)
    plt.xlabel('Zeit [h]')
    plt.ylabel('Anzahl der LKWs pro timestep')
    #plt.legend()

    scenario_name = get_scenario_name()
    file_name = f"{scenario_name}_verkehr.png"
    file_path = os.path.join(folder, file_name)
    plt.savefig(file_path)

    if show_plots:
        plt.show()

def plot_bev_anzahl(df_gesamt_anzahl):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots()
    min_values = df_gesamt_anzahl.min(axis=1, skipna=True)
    max_values = df_gesamt_anzahl.max(axis=1, skipna=True)
    avg_values = df_gesamt_anzahl.mean(axis=1, skipna=True)
    median_values = df_gesamt_anzahl.median(axis=1)

    # Plotte die Ergebnisse der Anzahl der BEV-LKW in jedem timestep als Band
    plt.plot(df_gesamt_anzahl.index, avg_values, label='Durchschnitt')
    #plt.plot(df_gesamt_anzahl.index, median_values, label='Median', color='orange')

    plt.fill_between(df_gesamt_anzahl.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

    plt.xlabel('Zeit [min]')
    plt.ylabel('Anzahl der ankommenden BEV-LKWs pro timestep')
    plt.legend()

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_bev_anzahl.png"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)

    if show_plots:
        plt.show()

def plot_energien_pro_lkw(gesamt_energiemengen_pro_lkw_dict, ladesäulentyp):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots()

    ladesäulentyp_data = [run[ladesäulentyp] for run in gesamt_energiemengen_pro_lkw_dict.values()]

    global_min = min(min(ladesäulentyp_list) for ladesäulentyp_list in ladesäulentyp_data)
    global_max = max(max(ladesäulentyp_list) for ladesäulentyp_list in ladesäulentyp_data)

    bin_width = 1
    bin_centers = np.arange(global_min - bin_width / 2, global_max + bin_width, bin_width)

    for i, ladesäulentyp_list in enumerate(ladesäulentyp_data):
        plt.hist(ladesäulentyp_list, bins=bin_centers, alpha=0.1, color='blue', density=True)

    plt.xlabel(f"geladene Strommenge pro LKW an {ladesäulentyp} [kWh]")
    plt.ylabel('rel. Häufigkeit')

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_{ladesäulentyp}_energien_pro_lkw.pdf"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)
    if show_plots:
        plt.show()
    dummy=0

def plot_ladezeiten_pro_lkw(gesamt_ladezeiten_pro_lkw_dict, ladesäulentyp):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots()

    ladesäulentyp_data = [run[ladesäulentyp] for run in gesamt_ladezeiten_pro_lkw_dict.values()]

    global_min = min(min(ladesäulentyp_list) for ladesäulentyp_list in ladesäulentyp_data)
    global_max = max(max(ladesäulentyp_list) for ladesäulentyp_list in ladesäulentyp_data)

    bin_width = 1
    bin_centers = np.arange(global_min - bin_width / 2, global_max + bin_width, bin_width)

    for i, ladesäulentyp_list in enumerate(ladesäulentyp_data):
        plt.hist(ladesäulentyp_list, bins=bin_centers, alpha=0.1, color='blue', density=True)

    plt.xlabel(f"Ladezeit pro LKW an {ladesäulentyp} [min]")
    plt.ylabel('rel. Häufigkeit')

    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"{scenario_name}_{ladesäulentyp}_ladezeiten_pro_lkw.pdf"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)
    if show_plots:
        plt.show()

def plot_ladekurve(ladekurve):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots()
    plt.grid(zorder=0)
    plt.step(ladekurve.index, ladekurve.values, label='Stufenfunktion', where='post')
    plt.ylabel(f"relative Ladeleistung")
    plt.xlabel('State of Charge (SoC) [%]')




    if save_plots:
        scenario_name = get_scenario_name()
        file_name = f"ladekurve.png"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)
    if show_plots:
        plt.show()