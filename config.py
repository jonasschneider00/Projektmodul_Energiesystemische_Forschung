
###################
start_date = 210104
end_date = 210110
anzahl_simulationen = 10

anteil_bev = 0.8374
tankwahrscheinlichkeit = 0.05
verkehrssteigerung = 1.3355

timedelta = 5  # time resolution in min
max_akkustand = 80  # relative capacity when leaving the charging station

# Definition der Raststätte
anzahl_ladesäulen_typ = {'HPC': 23, 'NCS': 32, 'LPC': 32, 'MWC': 11}
max_ladeleistung_ladesäulen_typ = {'HPC': 350, 'NCS': 150, 'LPC': 150, 'MWC': 1000}  # in kW
pausenzeiten_ladesäulen_typ = {'HPC': 60, 'NCS': 480, 'LPC': 480, 'MWC': 60}  # in min
verteilung_kapazitäten = {252: 0.6025, 504: 0.3065, 756: 0.091}  # Verteilung tagsüber (Summe muss 1 sein)

netzanschlussleistung = 15000  # (anzahl_ncs * leistung_ncs + anzahl_hpc * leistung_hpc) * 0.8

Beispieldaten = False

if Beispieldaten:
    timesteps = 100

load_existing_input = True
run_modell = True
###################