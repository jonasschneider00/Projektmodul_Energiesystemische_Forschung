import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# config
############
timedelta = 5  # time resolution in min
anzahl_ncs = 1
anzahl_hpc = 1
max_akkustand = 80  # relative capacity when leaving the charging station
leistung_ncs = 150
leistung_hpc = 350
netzanschlussleistung = (anzahl_ncs * leistung_ncs + anzahl_hpc * leistung_hpc) * 0.8
nachtzeit_ncs = 480
pausenzeit = 60
###########


anzahl_ladesäulen = anzahl_ncs + anzahl_hpc  # number of charging spots
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


def create_dataframe_with_dimensions(num_rows, num_columns, anzahl_ncs, anzahl_hpc):
    data = {f'Ladesäule {i}': [[]] * num_rows for i in range(1, num_columns + 1)}
    df = pd.DataFrame(data)
    num_ncs_columns = anzahl_ncs
    num_hpc_columns = anzahl_hpc

    # Ersetzen der Spaltennamen für NCS
    for i in range(1, num_ncs_columns + 1):
        df = df.rename(columns={f'Ladesäule {i}': f'Ladesäule {i} NCS'})

    # Ersetzen der Spaltennamen für HPC
    for i in range(1, num_hpc_columns + 1):
        df = df.rename(columns={f'Ladesäule {i + num_ncs_columns}': f'Ladesäule {i + num_ncs_columns} HPC'})

    neuer_index = range(0, len(df) * timedelta, timedelta)
    df.set_index(pd.Index(neuer_index), inplace=True)
    return df


def getladeleistung(ladestand, ladekurve, ladesäule):
    if 'NCS' in ladesäule:
        return ladekurve.at[ladestand, 'rel. Ladeleistung'] * leistung_ncs
    elif 'HPC' in ladesäule:
        return ladekurve.at[ladestand, 'rel. Ladeleistung'] * leistung_hpc
    else:
        print('Fehler bei Leistungsdefinition der Ladesäulen')


def laden(df_ladesäulen_t, lkws_in_timestep, timestep, df_ladeleistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt):
    df_t1 = df_ladesäulen_t.copy()
    df_t1_leistung = df_ladeleistung.copy()
    summe_ladender_lkws_dict = {'NCS' : 0, 'HPC': 0}
    anzahl_ladesäulen_dict = {'NCS' : anzahl_ncs, 'HPC': anzahl_hpc}
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
                anzahl_ladesäulen_typ = anzahl_ladesäulen_dict[lkw[2]]
                if len(wert) == 0 and (summe_ladender_lkws < anzahl_ladesäulen_typ) and lkw[2] in ladesäule:
                    df_t1.at[timestep, ladesäule] = [lkw]
                    summe_ladender_lkws += 1
                    summe_ladender_lkws_dict[lkw[2]] = summe_ladender_lkws
                    lkws_geladen_gesamt += 1
                    break
                elif len(wert) == 1 and (summe_ladender_lkws >= anzahl_ladesäulen_typ) and lkw[2] in ladesäule:
                    df_t1.at[timestep, ladesäule] += [lkw]
                    summe_ladender_lkws += 1
                    summe_ladender_lkws_dict[lkw[2]] = summe_ladender_lkws
                    lkws_geladen_gesamt += 1
                    break
                elif (summe_ladender_lkws == 2 * anzahl_ladesäulen_typ):
                    lkws_nicht_geladen_gesamt += 1
                    break
                continue
    return df_t1, df_t1_leistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt


def get_max_time(ladesäule):
    if 'NCS' in ladesäule:
        return nachtzeit_ncs
    elif 'HPC' in ladesäule:
        return pausenzeit
    else:
        print("Fehler bei maximaler Ladezeit")


def lade_lkw(lkw, ladeleistung):
    lkw_t1 = lkw
    lkw_t1[0] += round(((ladeleistung * timedelta / 60) / lkw_t1[1]) * 100, 2)
    lkw_t1[3] += timedelta
    return [lkw_t1]


def gesamte_ladeleistung(df_ladeleistung):
    df_gesamtleistung = create_dataframe_with_dimensions(num_rows=timesteps, num_columns=1, anzahl_ncs=1, anzahl_hpc=0)
    df_gesamtleistung.rename(columns={'Ladesäule 1 NCS': 'Gesamtleistung'}, inplace=True)
    for index, row in df_ladeleistung.iterrows():
        summe = 0
        for column, leistungen in row.items():
            for leistung in leistungen:
                summe += leistung
        df_gesamtleistung.at[index, 'Gesamtleistung'] = summe
    return df_gesamtleistung


if __name__ == '__main__':
    lkws = read_LKW_data()
    ladekurve = erstelle_Ladekurve()
    df_lkws = create_dataframe_with_dimensions(num_rows=timesteps, num_columns=anzahl_ladesäulen, anzahl_hpc=anzahl_hpc,
                                               anzahl_ncs=anzahl_ncs)
    df_ladeleistung = create_dataframe_with_dimensions(num_rows=timesteps, num_columns=anzahl_ladesäulen,
                                                       anzahl_hpc=anzahl_hpc, anzahl_ncs=anzahl_ncs)
    sortierte_lkw_liste = sortiere_lkws_nach_timstep(lkws=lkws)
    timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)

    lkws_geladen_gesamt = 0
    lkws_nicht_geladen_gesamt = 0

    for t in timestepsarray:
        lkws_in_timestep = sortierte_lkw_liste.at[t, 'Ankommende LKWs']
        df_lkws, df_ladeleistung, lkws_geladen_gesamt, lkws_nicht_geladen_gesamt = laden(df_ladesäulen_t=df_lkws,
                                                                                         lkws_in_timestep=lkws_in_timestep,
                                                                                         timestep=t,
                                                                                         df_ladeleistung=df_ladeleistung,
                                                                                         lkws_nicht_geladen_gesamt=lkws_nicht_geladen_gesamt,
                                                                                         lkws_geladen_gesamt=lkws_geladen_gesamt)
    gesamte_ladeleistung = gesamte_ladeleistung(df_ladeleistung=df_ladeleistung)

    plt.plot(gesamte_ladeleistung.index, gesamte_ladeleistung['Gesamtleistung'], marker='o', linestyle='-')
    plt.xlabel('Zeit [min]')
    plt.ylabel('Gesamtleistung [kW]')
    plt.title('Lastgang')
    plt.show()

    print(df_lkws)
