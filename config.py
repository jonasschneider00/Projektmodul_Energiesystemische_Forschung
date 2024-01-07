
###################
start_date = 210104
end_date = 210110
anzahl_simulationen = 5

anteil_bev = 0.4
tankwahrscheinlichkeit = 0.05
verkehrssteigerung = 1.0

timedelta = 5  # time resolution in min
max_akkustand = 80  # relative capacity when leaving the charging station

# Definition der Raststätte
anzahl_ladesäulen_typ = {'HPC': 5, 'NCS': 6, 'LPC': 6, 'MWC': 2}
max_ladeleistung_ladesäulen_typ = {'HPC': 350, 'NCS': 150, 'LPC': 150, 'MWC': 1000}  # in kW
pausenzeiten_ladesäulen_typ = {'HPC': 60, 'NCS': 480, 'LPC': 480, 'MWC': 60}  # in min
verteilung_ladesäulen_typ = {'HPC': 0.3, 'LPC': 0.5, 'MWC': 0.2}  # Verteilung tagsüber (Summe muss 1 sein)

netzanschlussleistung = 15000  # (anzahl_ncs * leistung_ncs + anzahl_hpc * leistung_hpc) * 0.8

Beispieldaten = False

if Beispieldaten:
    timesteps = 100

load_existing_input = False
run_modell = True
###################