import pandas as pd
import numpy as np

timesteps = 15
timedelta = 10
anzahl_ladesäulen = 3

data = {'Ankunftszeit': [0, 0, 0, 0, 20, 30, 30],
        'Kapazität': [600, 800, 800, 800, 600, 600, 700],
        'Akkustand': [10, 12, 30, 22, 24, 14, 21]}
lkws = pd.DataFrame(data)


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
    num_rows = 101
    cell_value = 600
    df = pd.DataFrame({'Ladeleistung': [cell_value] * num_rows})
    return df


def create_dataframe_with_dimensions(num_rows, num_columns):
    data = {f'Ladesäule {i}': [[]] * num_rows for i in range(1, num_columns + 1)}
    df = pd.DataFrame(data)
    neuer_index = range(0, len(df) * timedelta, timedelta)
    df.set_index(pd.Index(neuer_index), inplace=True)
    return df


def getladeleistung(ladestand, ladekurve):
    return ladekurve.at[ladestand, 'Ladeleistung']


def laden(df_ladesäulen_t, lkws_in_timestep, timestep):
    df_t1 = df_ladesäulen_t
    summe_ladender_lkws = 0
    if len(lkws_in_timestep) == 0:
        return df_ladesäulen_t
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
    return df_t1


# def lademanegement(anzahl_ladender_lkw):
#   return dummy=0

if __name__ == '__main__':
    df = create_dataframe_with_dimensions(num_rows=timesteps, num_columns=anzahl_ladesäulen)
    sortierte_lkw_liste = sortiere_lkws_nach_timstep(lkws=lkws)
    timestepsarray = create_timestep_array(timesteps=timesteps, timedelta=timedelta)
    for t in timestepsarray:
        lkws_in_timestep = sortierte_lkw_liste.at[t, 'Ankommende LKWs']
        df = laden(df_ladesäulen_t=df, lkws_in_timestep=lkws_in_timestep, timestep=t)
    print(df)
