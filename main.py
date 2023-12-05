import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from approximation import *

# config
############
timedelta = 5  # time resolution in min
max_akkustand = 80  # relative capacity when leaving the charging station
# Definition der Raststätte
anzahl_ladesäulen_typ = {'HPC': 18, 'NCS': 15, 'LPC': 10, 'MWC': 2}
max_ladeleistung_ladesäulen_typ = {'HPC': 350, 'NCS': 150, 'LPC': 150,'MWC': 1000} # in kW
pausenzeiten_ladesäulen_typ = {'HPC': 60, 'NCS': 480, 'LPC': 480,'MWC': 60} # in min
verteilung_ladesäulen_typ = {'HPC': 0.8, 'LPC': 0.15,'MWC': 0.05} # Verteilung tagsüber (Summe muss 1 sein)

netzanschlussleistung = 7000 #(anzahl_ncs * leistung_ncs + anzahl_hpc * leistung_hpc) * 0.8

Beispieldaten = False
# only if Beispieldaten = False :
plot = 'Energiemenge' # nicht_ladende_LKWs, Lastgang, Energiemenge
###########


if Beispieldaten:
    timesteps = 100


def read_LKW_data():
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
        lkwelement = [row['Akkustand'], row['Kapazität'], row['Ladesäule'], row['Ladezeit']]
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


def create_dataframe_with_dimensions(num_rows,  anzahl_ladesäulen_typ):
    num_columns = 0
    for wert in anzahl_ladesäulen_typ.values():
        num_columns += wert

    data = {f'Ladesäule {i}': [[]] * num_rows for i in range(1, num_columns + 1)}
    df = pd.DataFrame(data)

    namen = anzahl_ladesäulen_typ.keys()
    index = 0
    for name in namen:
        for i in range(1, anzahl_ladesäulen_typ[name] + 1):
            df = df.rename(columns={f'Ladesäule {index+i}': f'Ladesäule {index+i} {name}'})
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


def laden(df_ladesäulen_t, lkws_in_timestep, timestep, df_ladeleistung, lkws_geladen_gesamt, geladen_an_ladesäule_dict):
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
            df_t1.at[timestep, ladesäule] = lkws_t_0
    if len(lkws_in_timestep) == 0:
        return df_t1, df_t1_leistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt
    else:
        for lkw in lkws_in_timestep:
            for ladesäule, wert in df_t1.loc[t].items():
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
    return [lkw_t1]


def gesamte_ladeleistung(df_ladeleistung):
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
    anzahl_tage = int(len(df_ladeleistung)/(1440/timedelta))
    result_df = pd.DataFrame(index=range(anzahl_tage), columns=anzahl_ladesäulen_typ.keys())

    # Iteriere über die Elemente im übergebenen Array
    for name in anzahl_ladesäulen_typ.keys():
        subset = df_ladeleistung.filter(like=name)
        modified_data = {key: [sum(inner) if inner else 0.0 for inner in value] for key, value in subset.items()}
        df = pd.DataFrame(modified_data)
        df_energien = df * timedelta/60
        index = 0
        for i in range(0, len(df_energien), int(1440/timedelta)):

            subset_tag = df_energien.iloc[i:i + int(1440/timedelta), :]

            sum_values = subset_tag.filter(like=name).sum().sum()
            result_df[name][index] = sum_values
            index += 1
    dummy = 0





    return result_df


if __name__ == '__main__':
    if Beispieldaten:
        lkws = read_LKW_data()
        df_lkws = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
        df_ladeleistung = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
        sortierte_lkw_liste = sortiere_lkws_nach_timstep(lkws=lkws)
        timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)

        lkws_geladen_gesamt = 0
        lkws_nicht_geladen_gesamt = 0
        geladen_an_ladesäule_dict = {}
        for name in anzahl_ladesäulen_typ.keys():
            geladen_an_ladesäule_dict[name] = 0
        for t in timestepsarray:
            lkws_in_timestep = sortierte_lkw_liste.at[t, 'Ankommende LKWs']
            df_lkws, df_ladeleistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt = laden(df_ladesäulen_t=df_lkws,
                                                                                             lkws_in_timestep=lkws_in_timestep,
                                                                                             timestep=t,
                                                                                             df_ladeleistung=df_ladeleistung,
                                                                                             lkws_geladen_gesamt=lkws_geladen_gesamt,
                                                                                             geladen_an_ladesäule_dict=geladen_an_ladesäule_dict)
        gesamte_ladeleistung = gesamte_ladeleistung(df_ladeleistung=df_ladeleistung)

        plt.plot(gesamte_ladeleistung.index, gesamte_ladeleistung['Gesamtleistung'], marker='o', linestyle='-')
        plt.xlabel('Zeit [min]')
        plt.ylabel('Gesamtleistung [kW]')
        plt.title('Lastgang')
        plt.show()

    else:
        data = run_simulation()
        timesteps = len(data)


        timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)

        lkws_geladen_gesamt = 0
        lkws_nicht_geladen_gesamt = 0
        gesamt_df = pd.DataFrame()
        gesamt_df_nicht_ladende_lkws = pd.DataFrame()
        ladequoten = {}
        energien_dict = {}

        for run in data.columns:
            # Dataframes für lkws, ladeleistungen und nicht ladende lkws deklarieren
            df_lkws = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
            df_ladeleistung = create_dataframe_with_dimensions(num_rows=timesteps,  anzahl_ladesäulen_typ=anzahl_ladesäulen_typ)
            df_nicht_ladende_lkws = create_dataframe_with_dimensions(num_rows=timesteps, anzahl_ladesäulen_typ={'NCS': 1, 'HPC': 0})
            df_nicht_ladende_lkws.rename(columns={'Ladesäule 1 NCS': run}, inplace=True)

            summe_nicht_ladende_lkws = 0
            lkws_geladen_gesamt = 0
            geladen_an_ladesäule_dict = {}
            for name in anzahl_ladesäulen_typ.keys():
                geladen_an_ladesäule_dict[name]=0

            for t in timestepsarray:
                lkws_in_timestep = data.at[t, run]
                df_lkws, df_ladeleistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt = laden(df_ladesäulen_t=df_lkws,
                                                                                                 lkws_in_timestep=lkws_in_timestep,
                                                                                                 timestep=t,
                                                                                                 df_ladeleistung=df_ladeleistung,
                                                                                                 lkws_geladen_gesamt=lkws_geladen_gesamt,
                                                                                                 geladen_an_ladesäule_dict=geladen_an_ladesäule_dict)
                df_nicht_ladende_lkws.at[t, run] = lkws_nicht_geladen_gesamt
                summe_nicht_ladende_lkws += lkws_nicht_geladen_gesamt
                dummy = 0
            gesamte_ladeleistung_df = gesamte_ladeleistung(df_ladeleistung=df_ladeleistung)
            gesamt_df[run] = gesamte_ladeleistung_df['Gesamtleistung']
            gesamt_df[run] = pd.to_numeric(gesamt_df[run], errors='coerce')
            gesamt_df_nicht_ladende_lkws[run] =df_nicht_ladende_lkws[run]
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



        if plot == 'Lastgang':
            min_values = gesamt_df.min(axis=1, skipna=True)
            max_values = gesamt_df.max(axis=1, skipna=True)
            avg_values = gesamt_df.mean(axis=1, skipna=True)
            plt.plot(gesamt_df.index, avg_values, label='Durchschnitt')
            plt.fill_between(gesamt_df.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

            plt.xlabel('Zeit [min]')
            plt.ylabel('Leistung [kW]')
            plt.legend()
            plt.show()
        elif plot == 'nicht_ladende_LKWs':
            min_values = gesamt_df_nicht_ladende_lkws.min(axis=1, skipna=True)
            max_values = gesamt_df_nicht_ladende_lkws.max(axis=1, skipna=True)
            avg_values = gesamt_df_nicht_ladende_lkws.mean(axis=1, skipna=True)
            plt.plot(gesamt_df_nicht_ladende_lkws.index, avg_values, label='Durchschnitt')
            plt.fill_between(gesamt_df_nicht_ladende_lkws.index, min_values, max_values, alpha=0.2, label='Band (Min-Max)')

            plt.xlabel('Zeit [min]')
            plt.ylabel('Anzahl nicht ladender LKWs')
            plt.legend()
            plt.show()
        elif plot == 'Energiemenge':
            average_data = sum([df.values for df in energien_dict.values()]) / len(energien_dict)
            average_df = pd.DataFrame(average_data, index=energien_dict['Run_0'].index, columns=energien_dict['Run_0'].columns)

            fig, ax = plt.subplots()
            average_df.plot(kind='bar', stacked=True, ax=ax, width=0.8, figsize=(10, 6))
            plt.xlabel('Tag')
            plt.ylabel('Energiemenge [kWh]')
            plt.title('durch. Energiemenge über alle Simulationen pro Tag ')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.show()