import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import math

# Erstellen Sie eine Beispiel-Stufenfunktion mit 8760 Werten
x_values = np.arange(8760)
# Generieren Sie zufällige y-Werte im Bereich von 100 bis 250
y_values = np.random.randint(50, 150, 8760)

# Erstellen Sie eine Kubische-Spline-Interpolation
spline = CubicSpline(x_values, y_values)

# Generieren Sie Werte für die stetige Funktion
x_continuous = np.linspace(0, 8759, 105120)  # Mehr Punkte für eine glattere Kurve
y_continuous = spline(x_continuous)

# 1. Summe aller y_values
sum_y_values = np.sum(y_values)
print(f"Summe aller y_values: {sum_y_values}")

lkws_in_timestep = []
summe_in_timestep = []
summe = 0
summe_lkws= 0
for i in y_continuous:
    summe += i*1/12
    wert = math.floor(summe)
    lkws_in_timestep.append(wert)
    summe_lkws+=wert
    summe = summe - wert
    summe_in_timestep.append(summe)
print(f"Summe aller LKWs: {summe_lkws}")
# Plot der Original-Stufenfunktion und der approximierten stetigen Funktion
plt.step(x_values, y_values, where='mid', label='Stufenfunktion')
plt.plot(x_continuous, y_continuous, label='Approximierte Funktion')
plt.legend()
plt.show()



