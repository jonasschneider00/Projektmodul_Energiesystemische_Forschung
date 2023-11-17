import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# config
###########
n_tage = 1 # number of simulated days
timedelta = 5 # time resolution in min
anzahl_ladesäulen = 10 # number of charging spots
max_akkustand = 80 # relative capacity when leaving the charging station
###########

timesteps = int(n_tage * 1440/timedelta)

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
        lkwelement = [row['Akkustand'], row['Kapazität']]
        if not df.at[lkws.at[index, 'Ankunftszeit'], 'Ankommende LKWs']:
            df.at[lkws.at[index, 'Ankunftszeit'], 'Ankommende LKWs'] = [lkwelement]
        else:
            df.at[lkws.at[index, 'Ankunftszeit'], 'Ankommende LKWs'] += [lkwelement]
    return df


def erstelle_Ladekurve():
    #num_rows = 101
    #cell_value = 600
    #df = pd.DataFrame({'Ladeleistung': [cell_value] * num_rows})
    daten = []
    for i in range(0, 100, 10):
        for j in range(i, i + 10):
            if j <= 100:
                daten.append((j, 600 - (i // 10) * 10))
    df = pd.DataFrame(daten, columns=['Index', 'Ladeleistung'])
    return df


def create_dataframe_with_dimensions(num_rows, num_columns):
    data = {f'Ladesäule {i}': [[]] * num_rows for i in range(1, num_columns + 1)}
    df = pd.DataFrame(data)
    neuer_index = range(0, len(df) * timedelta, timedelta)
    df.set_index(pd.Index(neuer_index), inplace=True)
    return df


def getladeleistung(ladestand, ladekurve):
    return ladekurve.at[ladestand, 'Ladeleistung']


def laden(df_ladesäulen_t, lkws_in_timestep, timestep, df_ladeleistung):
    df_t1 = df_ladesäulen_t.copy()
    df_t1_leistung = df_ladeleistung.copy()
    summe_ladender_lkws = 0
    if timestep > 0:
        for l, ladesäule in enumerate(df_t1.columns):
            lkws_t_minus_1 = df_t1.at[timestep - timedelta, ladesäule]
            ladeleistungen = lademanegement(lkws_t_minus_1=lkws_t_minus_1)
            lkws_t_0 = []
            for i, lkw in enumerate(lkws_t_minus_1):
                lkw_copy = lkw[:]
                if lkw_copy[0] < max_akkustand:
                    lkw_t_0 = lade_lkw(lkw=lkw_copy, ladeleistung=ladeleistungen[i])
                    if lkw_t_0[0][0] < max_akkustand:
                        lkws_t_0 += lkw_t_0
                        summe_ladender_lkws += 1
                        df_t1_leistung.at[timestep - timedelta, ladesäule] = ladeleistungen
                    else:
                        df_t1_leistung.at[timestep - timedelta, ladesäule] = ladeleistungen
            df_t1.at[timestep, ladesäule] = lkws_t_0
    if len(lkws_in_timestep) == 0:
        return df_t1, df_t1_leistung
    else:
        for lkw in lkws_in_timestep:
            for ladesäule, wert in df_t1.loc[t].items():
                if len(wert) == 0 and (summe_ladender_lkws < anzahl_ladesäulen):
                    df_t1.at[timestep, ladesäule] = [lkw]
                    summe_ladender_lkws += 1
                    break
                elif len(wert) == 1 and (summe_ladender_lkws >= anzahl_ladesäulen):
                    df_t1.at[timestep, ladesäule] += [lkw]
                    summe_ladender_lkws += 1
                    break
                continue
    return df_t1, df_t1_leistung


def lade_lkw(lkw, ladeleistung):
    lkw_t1 = lkw
    lkw_t1[0] += round(((ladeleistung*timedelta/60)/lkw_t1[1])*100, 2)
    return [lkw_t1]


def lademanegement(lkws_t_minus_1):
    ladeleistungen = []
    if len(lkws_t_minus_1) == 1:
        ladeleistungen.append(getladeleistung(ladestand=round(lkws_t_minus_1[0][0]), ladekurve=erstelle_Ladekurve()))
    elif len(lkws_t_minus_1) == 2:
        for lkw in lkws_t_minus_1:
            ladeleistung = 0.5 * getladeleistung(ladestand=round(lkw[0]), ladekurve=erstelle_Ladekurve())
            ladeleistungen.append(ladeleistung)
    return ladeleistungen

def gesamte_ladeleistung(df_ladeleistung):
    df_gesamtleistung = create_dataframe_with_dimensions(num_rows=timesteps, num_columns=1)
    df_gesamtleistung.rename(columns={'Ladesäule 1': 'Gesamtleistung'}, inplace=True)
    for index, row in df_ladeleistung.iterrows():
        summe = 0
        for column, leistungen in row.items():
            for leistung in leistungen:
                summe += leistung
        df_gesamtleistung.at[index, 'Gesamtleistung'] = summe
    return df_gesamtleistung

if __name__ == '__main__':
    lkws = read_LKW_data()
    df_lkws = create_dataframe_with_dimensions(num_rows=timesteps, num_columns=anzahl_ladesäulen)
    df_ladeleistung = create_dataframe_with_dimensions(num_rows=timesteps, num_columns=anzahl_ladesäulen)
    sortierte_lkw_liste = sortiere_lkws_nach_timstep(lkws=lkws)
    timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)
    for t in timestepsarray:
        lkws_in_timestep = sortierte_lkw_liste.at[t, 'Ankommende LKWs']
        df_lkws, df_ladeleistung = laden(df_ladesäulen_t=df_lkws, lkws_in_timestep=lkws_in_timestep, timestep=t, df_ladeleistung=df_ladeleistung)
    gesamte_ladeleistung = gesamte_ladeleistung(df_ladeleistung=df_ladeleistung)

    plt.plot(gesamte_ladeleistung.index, gesamte_ladeleistung['Gesamtleistung'], marker='o', linestyle='-')
    plt.xlabel('Zeit [min]')
    plt.ylabel('Gesamtleistung [kW]')
    plt.title('Lastgang')
    plt.show()

    print(df_lkws)
