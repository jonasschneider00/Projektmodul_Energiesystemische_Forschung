import random
import pyomo.environ as pyo

# Definiere das Pyomo-Modell
model = pyo.ConcreteModel()

# Definiere die Anzahl der Anlagen und Aufgaben
num_anlagen = 3
num_aufgaben = 10

# Definiere die Menge der Anlagen und Aufgaben
model.Anlagen = pyo.RangeSet(0, num_anlagen - 1)
model.Aufgaben = pyo.RangeSet(0, num_aufgaben - 1)

# Binäre Variablen, ob eine Aufgabe auf einer Anlage ausgeführt wird
model.x = pyo.Var(model.Aufgaben, model.Anlagen, within=pyo.Binary)
model.z = pyo.Var(within=pyo.NonNegativeReals)

# Zielfunktion, um die Zeit in der dritten Anlage zu minimieren
def objective_rule(model):
    return model.z

model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

# Bearbeitungszeiten dynamisch festlegen
bearbeitungszeiten = {}

for i in model.Aufgaben:
    for j in model.Anlagen:
        if j == 2:
            # Auf Anlage 3 abhängig von der Verfügbarkeit von Anlage 1 oder 2
            anlage1_oder_2_frei = random.choice([True, False])
            if anlage1_oder_2_frei:
                bearbeitungszeiten[i, j] = random.uniform(40, 60)
            else:
                bearbeitungszeiten[i, j] = random.uniform(40, 60)
        else:
            bearbeitungszeiten[i, j] = random.uniform(40, 60)

# Einschränkungen
def assignment_rule(model, i):
    return sum(model.x[i, j] for j in model.Anlagen) == 1

model.assignment_constraint = pyo.Constraint(model.Aufgaben, rule=assignment_rule)

def machine3_time_rule(model):
    return model.z >= sum(model.x[i, 2] * bearbeitungszeiten[i, 2] for i in model.Aufgaben if bearbeitungszeiten[i, 2] is not None)

model.machine3_time_constraint = pyo.Constraint(rule=machine3_time_rule)

# Solver-Konfiguration (z.B. GLPK)
solver = pyo.SolverFactory('glpk')
solver.solve(model)

# Ergebnisse anzeigen
if solver.status == pyo.SolverStatus.ok and solver.termination_condition == pyo.TerminationCondition.optimal:
    print("Optimale Lösung gefunden:")
    for i in model.Aufgaben:
        for j in model.Anlagen:
            if model.x[i, j].value == 1:
                if j == 2:
                    if bearbeitungszeiten[i, j] is not None:
                        print(f"Aufgabe {i + 1} auf Anlage {j + 1}, Bearbeitungszeit: {bearbeitungszeiten[i, j]:.2f} Minuten")
                    else:
                        print(f"Aufgabe {i + 1} auf Anlage {j + 1}, Bearbeitungszeit: Warten bis Anlage 1 oder 2 frei ist")
                else:
                    print(f"Aufgabe {i + 1} auf Anlage {j + 1}, Bearbeitungszeit: {bearbeitungszeiten[i, j]:.2f} Minuten")
    print(f"Zeit in der dritten Anlage: {model.z.value:.2f} Minuten")
else:
    print("Keine optimale Lösung gefunden.")



