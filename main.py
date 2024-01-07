import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from approximation import *
from plots import plot_energiemenge, plot_lastgang, plot_nicht_ladende_LKWs, plot_verkehr, plot_bev_anzahl, plot_energien_pro_lkw, plot_ladezeiten_pro_lkw
from config import *


def get_scenario_name():
    scenario_name = f"{start_date}_{end_date}_{anzahl_simulationen}_{int(anteil_bev * 100)}_{int(tankwahrscheinlichkeit * 100)}_{netzanschlussleistung}"
    return scenario_name


def read_example_LKW_data():
    working_directory = os.getcwd()
    file_path = os.path.join(working_directory, 'INPUT_LKW.xlsx')
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Fehler beim Lesen der Datei: {e}")
        return None


def create_timestep_array(timedelta, timesteps):
    return [i * timedelta for i in range(timesteps)]


def sortiere_lkws_nach_timstep(lkws):
    data = {f'Ankommende LKWs': [[]] * timesteps}
    df = pd.DataFrame(data)
    neuer_index = range(0, len(df) * timedelta, timedelta)
    df.set_index(pd.Index(neuer_index), inplace=True)

    for index, row in lkws.iterrows():
        lkwelement = [row['Akkustand'], row['Kapazität'], row['Ladesäule'], row['Ladezeit'], row['Energie']]
        if not df.at[lkws.at[index, 'Ankunftszeit'], 'Ankommende LKWs']:
            df.at[lkws.at[index, 'Ankunftszeit'], 'Ankommende LKWs'] = [lkwelement]
        else:
            df.at[lkws.at[index, 'Ankunftszeit'], 'Ankommende LKWs'] += [lkwelement]
    return df


def erstelle_Ladekurve():
    # num_rows = 101
    # cell_value = 600
    # df = pd.DataFrame({'Ladeleistung': [cell_value] * num_rows})
    # daten = []
    # for i in range(0, 100, 10):
    #     for j in range(i, i + 10):
    #         if j <= 100:
    #             daten.append((j, 600 - (i // 10) * 10))
    # df = pd.DataFrame(daten, columns=['Index', 'Ladeleistung'])
    # Erstelle eine Liste mit den gewünschten Werten entsprechend den angegebenen Intervallen
    values = [1.0] * 40 + [0.95] * 10 + [0.9] * 10 + [0.85] * 10 + [0.8] * 10 + [0.6] * 10 + [0.4] * 5 + [0.2] * 6

    # Erstelle das DataFrame
    df = pd.DataFrame({'rel. Ladeleistung': values})

    # Setze die Indizes von 0 bis 100
    df.index = range(101)

    return df


def create_dataframe_with_dimensions(num_rows, anzahl_ladesäulen_typ):
    num_columns = 0
    for wert in anzahl_ladesäulen_typ.values():
        num_columns += wert

    data = {f'Ladesäule {i}': [[]] * num_rows for i in range(1, num_columns + 1)}
    df = pd.DataFrame(data)

    namen = anzahl_ladesäulen_typ.keys()
    index = 0
    for name in namen:
        for i in range(1, anzahl_ladesäulen_typ[name] + 1):
            df = df.rename(columns={f'Ladesäule {index + i}': f'Ladesäule {index + i} {name}'})
        index += anzahl_ladesäulen_typ[name]

    neuer_index = range(0, len(df) * timedelta, timedelta)
    df.set_index(pd.Index(neuer_index), inplace=True)
    return df


def getladeleistung(ladestand, ladekurve, ladesäule):
    namen = anzahl_ladesäulen_typ.keys()
    for name in namen:
        if name in ladesäule:
            return ladekurve.at[ladestand, 'rel. Ladeleistung'] * max_ladeleistung_ladesäulen_typ[name]
    print('Fehler bei Leistungsdefinition der Ladesäulen')


def laden(df_ladesäulen_t, lkws_in_timestep, timestep, df_ladeleistung, lkws_geladen_gesamt, geladen_an_ladesäule_dict,
          nicht_geladen_an_ladesäule_dict, energiemengen_pro_lkw_dict, ladezeiten_pro_lkw_dict):
    lkws_nicht_geladen_gesamt = 0
    df_t1 = df_ladesäulen_t.copy()
    df_t1_leistung = df_ladeleistung.copy()
    summe_ladender_lkws_dict = {}
    for name in anzahl_ladesäulen_typ.keys():
        summe_ladender_lkws_dict[name] = 0

    if timestep > 0:
        vergleichsleistung = 0
        for l, ladesäule in enumerate(df_t1.columns):
            lkws_t_minus_1 = df_t1.at[timestep - timedelta, ladesäule]
            for i, lkw in enumerate(lkws_t_minus_1):
                vergleichsleistung += getladeleistung(ladestand=round(lkw[0]), ladekurve=erstelle_Ladekurve(),
                                                      ladesäule=ladesäule)
        ladefaktor = 0
        if vergleichsleistung <= netzanschlussleistung:
            ladefaktor = 1
        else:
            ladefaktor = netzanschlussleistung / vergleichsleistung
        # laden der LKWs aus t-1
        for l, ladesäule in enumerate(df_t1.columns):
            lkws_t_minus_1 = df_t1.at[timestep - timedelta, ladesäule]
            ladeleistungen = []
            lkws_t_0 = []
            max_time_ladesäule = get_max_time(ladesäule=ladesäule)
            for i, lkw in enumerate(lkws_t_minus_1):
                lkw_copy = lkw[:]
                ladeleistungen.append(getladeleistung(ladestand=round(lkw[0]), ladekurve=erstelle_Ladekurve(),
                                                      ladesäule=ladesäule) * ladefaktor)
                if lkw_copy[0] < max_akkustand:
                    lkw_t_0 = lade_lkw(lkw=lkw_copy, ladeleistung=ladeleistungen[i])
                    if lkw_t_0[0][0] < max_akkustand and lkw_copy[3] < max_time_ladesäule:
                        lkws_t_0 += lkw_t_0
                        summe_ladender_lkws_dict[lkw[2]] += 1
                        df_t1_leistung.at[timestep - timedelta, ladesäule] = ladeleistungen
                    else:
                        df_t1_leistung.at[timestep - timedelta, ladesäule] = ladeleistungen
                        energiemengen_pro_lkw_dict[lkw[2]].append(lkw_t_0[0][4])
                        ladezeiten_pro_lkw_dict[lkw[2]].append(lkw_t_0[0][3])
            df_t1.at[timestep, ladesäule] = lkws_t_0
    if len(lkws_in_timestep) == 0:
        return df_t1, df_t1_leistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt
    else:
        for lkw in lkws_in_timestep:
            for ladesäule, wert in df_t1.loc[timestep].items():
                summe_ladender_lkws = summe_ladender_lkws_dict[lkw[2]]
                anzahl_ladesäulen = anzahl_ladesäulen_typ[lkw[2]]
                if len(wert) == 0 and (summe_ladender_lkws < anzahl_ladesäulen) and lkw[2] in ladesäule:
                    df_t1.at[timestep, ladesäule] = [lkw]
                    summe_ladender_lkws += 1
                    summe_ladender_lkws_dict[lkw[2]] = summe_ladender_lkws
                    lkws_geladen_gesamt += 1
                    geladen_an_ladesäule_dict[lkw[2]] += 1
                    break
                # elif len(wert) == 1 and (summe_ladender_lkws >= anzahl_ladesäulen_typ) and lkw[2] in ladesäule:
                #     df_t1.at[timestep, ladesäule] += [lkw]
                #     summe_ladender_lkws += 1
                #     summe_ladender_lkws_dict[lkw[2]] = summe_ladender_lkws
                #     lkws_geladen_gesamt += 1
                #    break
                elif (summe_ladender_lkws >= anzahl_ladesäulen):
                    lkws_nicht_geladen_gesamt += 1
                    nicht_geladen_an_ladesäule_dict[lkw[2]] += 1
                    break
                else:
                    dummy = 0
                continue
    return df_t1, df_t1_leistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt


def get_max_time(ladesäule):
    namen = anzahl_ladesäulen_typ.keys()
    for name in namen:
        if name in ladesäule:
            return pausenzeiten_ladesäulen_typ[name]
    print("Fehler bei maximaler Ladezeit")


def lade_lkw(lkw, ladeleistung):
    lkw_t1 = lkw
    lkw_t1[0] += round(((ladeleistung * timedelta / 60) / lkw_t1[1]) * 100, 2)
    lkw_t1[3] += timedelta
    lkw_t1[4] += ladeleistung * timedelta / 60
    return [lkw_t1]


def get_gesamte_ladeleistung(df_ladeleistung, timesteps):
    df_gesamtleistung = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ={'NCS': 1, 'HPC': 0})
    df_gesamtleistung.rename(columns={'Ladesäule 1 NCS': 'Gesamtleistung'}, inplace=True)
    for index, row in df_ladeleistung.iterrows():
        summe = 0
        for column, leistungen in row.items():
            for leistung in leistungen:
                summe += leistung
        df_gesamtleistung.at[index, 'Gesamtleistung'] = summe
    return df_gesamtleistung


def tägliche_energiemenge(df_ladeleistung):
    anzahl_tage = int(len(df_ladeleistung) / (1440 / timedelta))
    result_df = pd.DataFrame(index=range(anzahl_tage), columns=anzahl_ladesäulen_typ.keys())

    # Iteriere über die Elemente im übergebenen Array
    for name in anzahl_ladesäulen_typ.keys():
        subset = df_ladeleistung.filter(like=name)
        modified_data = {key: [sum(inner) if inner else 0.0 for inner in value] for key, value in subset.items()}
        df = pd.DataFrame(modified_data)
        df_energien = df * timedelta / 60
        index = 0
        for i in range(0, len(df_energien), int(1440 / timedelta)):
            subset_tag = df_energien.iloc[i:i + int(1440 / timedelta), :]

            sum_values = subset_tag.filter(like=name).sum().sum()
            result_df[name][index] = sum_values
            index += 1
    return result_df


def save_input_data_to_pickle(data, df_gesamt_anzahl, verkehrsdaten, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    scenario_name = f"{start_date}_{end_date}_{anzahl_simulationen}_{int(anteil_bev * 100)}_{int(tankwahrscheinlichkeit * 100)}"
    file_name = f"{scenario_name}.pkl"
    file_path = os.path.join(folder, file_name)
    input_data = {
        'data': data,
        'df_gesamt_anzahl': df_gesamt_anzahl,
        'verkehrsdaten': verkehrsdaten
    }
    pd.to_pickle(input_data, file_path)


def load_input_data_from_pickle(folder):
    scenario_name = f"{start_date}_{end_date}_{anzahl_simulationen}_{int(anteil_bev * 100)}_{int(tankwahrscheinlichkeit * 100)}"
    file_name = f"{scenario_name}.pkl"
    file_path = os.path.join(folder, file_name)
    if os.path.exists(file_path):
        input_data = pd.read_pickle(file_path)

        data = input_data['data']
        df_gesamt_anzahl = input_data['df_gesamt_anzahl']
        verkehrsdaten = input_data['verkehrsdaten']

        return data, df_gesamt_anzahl, verkehrsdaten
    else:
        raise FileNotFoundError(f"Die Datei {file_path} existiert nicht.")


def save_output_data_to_pickle(gesamt_df_ladeleistung, gesamt_df_nicht_ladende_lkws, ladequoten, energien_dict,
                               df_lkws_dict, df_ladeleistung_dict,
                               gesamt_nicht_geladen_an_ladesöule_dict, gesamt_geladen_an_ladesöule_dict, folder,
                               gesamt_energiemengen_pro_lkw_dict, gesamt_ladezeiten_pro_lkw_dict):
    if not os.path.exists(folder):
        os.makedirs(folder)

    scenario_name = get_scenario_name()
    file_name = f"{scenario_name}_output.pkl"
    file_path = os.path.join(folder, file_name)

    output_data = {
        'gesamt_df_ladeleistung': gesamt_df_ladeleistung,
        'gesamt_df_nicht_ladende_lkws': gesamt_df_nicht_ladende_lkws,
        'ladequoten': ladequoten,
        'energien_dict': energien_dict,
        'df_lkws_dict': df_lkws_dict,
        'df_ladeleistung_dict': df_ladeleistung_dict,
        'gesamt_nicht_geladen_an_ladesöule_dict': gesamt_nicht_geladen_an_ladesöule_dict,
        'gesamt_geladen_an_ladesöule_dict': gesamt_geladen_an_ladesöule_dict,
        'gesamt_energiemengen_pro_lkw_dict': gesamt_energiemengen_pro_lkw_dict,
        'gesamt_ladezeiten_pro_lkw_dict': gesamt_ladezeiten_pro_lkw_dict
    }

    pd.to_pickle(output_data, file_path)
    print(f"Die Ausgabedaten wurden erfolgreich in '{file_path}' gespeichert.")


def load_output_data_from_pickle(folder):
    scenario_name = get_scenario_name()
    file_name = f"{scenario_name}_output.pkl"
    file_path = os.path.join(folder, file_name)

    if os.path.exists(file_path):
        output_data = pd.read_pickle(file_path)

        gesamt_df_ladeleistung = output_data['gesamt_df_ladeleistung']
        gesamt_df_nicht_ladende_lkws = output_data['gesamt_df_nicht_ladende_lkws']
        ladequoten = output_data['ladequoten']
        energien_dict = output_data['energien_dict']
        df_lkws_dict = output_data['df_lkws_dict']
        df_ladeleistung_dict = output_data['df_ladeleistung_dict']
        gesamt_geladen_an_ladesöule_dict = output_data['gesamt_geladen_an_ladesöule_dict']
        gesamt_nicht_geladen_an_ladesöule_dict = output_data['gesamt_nicht_geladen_an_ladesöule_dict']
        gesamt_energiemengen_pro_lkw_dict = output_data['gesamt_energiemengen_pro_lkw_dict']
        gesamt_ladezeiten_pro_lkw_dict = output_data['gesamt_ladezeiten_pro_lkw_dict']

        return gesamt_df_ladeleistung, gesamt_df_nicht_ladende_lkws, ladequoten, energien_dict, df_lkws_dict, \
            df_ladeleistung_dict, gesamt_nicht_geladen_an_ladesöule_dict, gesamt_geladen_an_ladesöule_dict, \
            gesamt_energiemengen_pro_lkw_dict, gesamt_ladezeiten_pro_lkw_dict

    else:
        raise FileNotFoundError(f"Die Datei {file_path} existiert nicht.")


def run_beispieldaten():
    lkws = read_example_LKW_data()
    df_lkws = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
    df_ladeleistung = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
    sortierte_lkw_liste = sortiere_lkws_nach_timstep(lkws=lkws)
    timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)

    lkws_geladen_gesamt = 0
    lkws_nicht_geladen_gesamt = 0
    geladen_an_ladesäule_dict = {}
    nicht_geladen_an_ladesäule_dict = {}
    energiemengen_pro_lkw_dict = {}
    ladezeiten_pro_lkw_dict = {}

    for name in anzahl_ladesäulen_typ.keys():
        geladen_an_ladesäule_dict[name] = 0
        nicht_geladen_an_ladesäule_dict[name] = 0
        energiemengen_pro_lkw_dict[name] = []
        ladezeiten_pro_lkw_dict[name] = []

    for t in timestepsarray:
        lkws_in_timestep = sortierte_lkw_liste.at[t, 'Ankommende LKWs']
        df_lkws, df_ladeleistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt = laden(df_ladesäulen_t=df_lkws,
                                                                                         lkws_in_timestep=lkws_in_timestep,
                                                                                         timestep=t,
                                                                                         df_ladeleistung=df_ladeleistung,
                                                                                         lkws_geladen_gesamt=lkws_geladen_gesamt,
                                                                                         geladen_an_ladesäule_dict=geladen_an_ladesäule_dict,
                                                                                         nicht_geladen_an_ladesäule_dict=nicht_geladen_an_ladesäule_dict,
                                                                                         energiemengen_pro_lkw_dict=energiemengen_pro_lkw_dict,
                                                                                         ladezeiten_pro_lkw_dict=ladezeiten_pro_lkw_dict)
    gesamte_ladeleistung = get_gesamte_ladeleistung(df_ladeleistung=df_ladeleistung, timesteps=timesteps)

    plt.plot(gesamte_ladeleistung.index, gesamte_ladeleistung['Gesamtleistung'], marker='o', linestyle='-')
    plt.xlabel('Zeit [min]')
    plt.ylabel('Gesamtleistung [kW]')
    plt.title('Lastgang')
    plt.show()


def run_reale_daten(data):

    timesteps = len(data)
    timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)

    lkws_geladen_gesamt = 0
    lkws_nicht_geladen_gesamt = 0
    gesamt_df_ladeleistung = pd.DataFrame()  # Speichert die Gesamtleistungen des Ladehubs in jedem timestep aus jedem run
    gesamt_df_nicht_ladende_lkws = pd.DataFrame()  # Speichert die nicht ladenden LKWs in jedem timestep aus jedem run
    ladequoten = {}  # Speichert die Ladequoten des Ladehubs in jedem timestep aus jedem run
    energien_dict = {}  # Speichert alle täglichen Strommengen aus jedem run
    df_lkws_dict = {}  # Speichert alle LKW-Daten aus jedem timestep und run
    df_ladeleistung_dict = {}  # Speichert alle Ladeleistungen jeder Ladesäule aus jedem timestep und run
    gesamt_geladen_an_ladesöule_dict = {}  # Speichert die Anzahl der geladenen LKWs an jedem Ladesäulentyp (auch unvollständiger Ladevorgang z.B. am Ende des betrachteten Zeitraums)
    gesamt_nicht_geladen_an_ladesöule_dict = {}  # Speichert die Anzahl der nicht geladenen LKWs aufgeschlüsselt nach Ladesäulentyp
    gesamt_energiemengen_pro_lkw_dict = {}  # Speichert alle geladenen Energiemengen pro LKW für abgeschlossenen Ladevorgänge
    gesamt_ladezeiten_pro_lkw_dict = {}  # # Speichert alle Ladezeiten pro LKW für abgeschlossenen Ladevorgänge

    for run in data.columns:
        # Dataframes für lkws, ladeleistungen und nicht ladende lkws deklarieren
        df_lkws = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
        df_ladeleistung = create_dataframe_with_dimensions(num_rows=timesteps,
                                                           anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
        df_nicht_ladende_lkws = create_dataframe_with_dimensions(num_rows=timesteps,
                                                                 anzahl_ladesäulen_typ={'NCS': 1, 'HPC': 0})
        df_nicht_ladende_lkws.rename(columns={'Ladesäule 1 NCS': run}, inplace=True)

        summe_nicht_ladende_lkws = 0
        lkws_geladen_gesamt = 0

        geladen_an_ladesäule_dict = {}
        nicht_geladen_an_ladesäule_dict = {}
        energiemengen_pro_lkw_dict = {}
        ladezeiten_pro_lkw_dict = {}
        for name in anzahl_ladesäulen_typ.keys():
            geladen_an_ladesäule_dict[name] = 0
            nicht_geladen_an_ladesäule_dict[name] = 0
            energiemengen_pro_lkw_dict[name] = []
            ladezeiten_pro_lkw_dict[name] = []

        for t in timestepsarray:
            lkws_in_timestep = data.at[t, run]
            df_lkws, df_ladeleistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt = laden(df_ladesäulen_t=df_lkws,
                                                                                             lkws_in_timestep=lkws_in_timestep,
                                                                                             timestep=t,
                                                                                             df_ladeleistung=df_ladeleistung,
                                                                                             lkws_geladen_gesamt=lkws_geladen_gesamt,
                                                                                             geladen_an_ladesäule_dict=geladen_an_ladesäule_dict,
                                                                                             nicht_geladen_an_ladesäule_dict=nicht_geladen_an_ladesäule_dict,
                                                                                             energiemengen_pro_lkw_dict=energiemengen_pro_lkw_dict,
                                                                                             ladezeiten_pro_lkw_dict=ladezeiten_pro_lkw_dict)
            df_nicht_ladende_lkws.at[t, run] = lkws_nicht_geladen_gesamt
            summe_nicht_ladende_lkws += lkws_nicht_geladen_gesamt

        df_ladeleistung_dict[run] = df_ladeleistung
        df_lkws_dict[run] = df_lkws

        gesamt_geladen_an_ladesöule_dict[run] = geladen_an_ladesäule_dict
        gesamt_nicht_geladen_an_ladesöule_dict[run] = nicht_geladen_an_ladesäule_dict
        gesamt_ladezeiten_pro_lkw_dict[run] = ladezeiten_pro_lkw_dict
        gesamt_energiemengen_pro_lkw_dict[run] = energiemengen_pro_lkw_dict

        gesamte_ladeleistung_df = get_gesamte_ladeleistung(df_ladeleistung=df_ladeleistung, timesteps=timesteps)
        gesamt_df_ladeleistung[run] = gesamte_ladeleistung_df['Gesamtleistung']
        gesamt_df_ladeleistung[run] = pd.to_numeric(gesamt_df_ladeleistung[run], errors='coerce')
        gesamt_df_nicht_ladende_lkws[run] = df_nicht_ladende_lkws[run]
        gesamt_df_nicht_ladende_lkws[run] = pd.to_numeric(gesamt_df_nicht_ladende_lkws[run], errors='coerce')

        ladequote = lkws_geladen_gesamt / (summe_nicht_ladende_lkws + lkws_geladen_gesamt)
        ladequoten[run] = ladequote

        energien = tägliche_energiemenge(df_ladeleistung=df_ladeleistung)
        energien_dict[run] = energien
        print("########################")
        print(run)
        print("------------------------")
        print(f"Ladequote: {ladequote}")
        print(f"nicht geladene LKWs: {summe_nicht_ladende_lkws}")
        print(f"geladene LKWs: {lkws_geladen_gesamt}")
        print(geladen_an_ladesäule_dict)
        print("übertragene Energiemenge in [kWh]")
        print(energien)

    save_output_data_to_pickle(gesamt_df_ladeleistung=gesamt_df_ladeleistung,
                               gesamt_df_nicht_ladende_lkws=gesamt_df_nicht_ladende_lkws,
                               df_lkws_dict=df_lkws_dict, df_ladeleistung_dict=df_ladeleistung_dict,
                               energien_dict=energien_dict, ladequoten=ladequoten, folder='OUTPUT',
                               gesamt_geladen_an_ladesöule_dict=gesamt_geladen_an_ladesöule_dict,
                               gesamt_nicht_geladen_an_ladesöule_dict=gesamt_nicht_geladen_an_ladesöule_dict,
                               gesamt_ladezeiten_pro_lkw_dict=gesamt_ladezeiten_pro_lkw_dict,
                               gesamt_energiemengen_pro_lkw_dict=gesamt_energiemengen_pro_lkw_dict)


if __name__ == '__main__':
    ladesäulentypen = list(anzahl_ladesäulen_typ.keys())
    if Beispieldaten:
        run_beispieldaten()
    else:
        if load_existing_input:
            try:
                data, df_gesamt_anzahl, verkehrsdaten = load_input_data_from_pickle('INPUT')
                timesteps = len(data)
                timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)
                data.index = timestepsarray

                print("Daten erfolgreich aus Pickle-Datei geladen.")
            except FileNotFoundError:
                print("Die INPUT-Datei existiert nicht. Führe Simulation aus.")

        if not load_existing_input:
            data, df_gesamt_anzahl, verkehrsdaten = run_simulation()
            save_input_data_to_pickle(data=data, df_gesamt_anzahl=df_gesamt_anzahl, verkehrsdaten=verkehrsdaten,
                                      folder='INPUT')
            print("Daten erfolgreich simuliert und in Pickle-Datei gespeichert.")
        if run_modell:
            run_reale_daten(data=data)
        gesamt_df_ladeleistung, gesamt_df_nicht_ladende_lkws, ladequoten, energien_dict, df_lkws_dict, df_ladeleistung_dict, \
            gesamt_nicht_geladen_an_ladesöule_dict, gesamt_geladen_an_ladesöule_dict, \
            gesamt_energiemengen_pro_lkw_dict, gesamt_ladezeiten_pro_lkw_dict = load_output_data_from_pickle('OUTPUT')
        plot_lastgang(gesamt_df_ladeleistung)
        plot_nicht_ladende_LKWs(gesamt_df_nicht_ladende_lkws)
        plot_energiemenge(energien_dict)
        plot_verkehr(verkehrsdaten)
        plot_bev_anzahl(df_gesamt_anzahl)
        for ladesäulentyp in ladesäulentypen:
            plot_energien_pro_lkw(gesamt_energiemengen_pro_lkw_dict, ladesäulentyp=ladesäulentyp)
            plot_ladezeiten_pro_lkw(gesamt_ladezeiten_pro_lkw_dict, ladesäulentyp=ladesäulentyp)



