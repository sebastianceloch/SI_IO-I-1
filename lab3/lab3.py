import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
from scipy.stats import linregress

years = np.array([2000, 2002, 2005, 2007, 2010])
percentages = np.array([6.5, 7.0, 7.4, 8.2, 9.0])
slope, intercept, r, p, std_err = stats.linregress(years, percentages)
x_pred = np.array([2015, 2018, 2020])
y_pred = slope * x_pred + intercept
print("Przewidywane wartości w latach {}: {}".format(x_pred, np.round(y_pred, 3)))
yr12 = (12 - intercept) / slope
print("Procent bezrobotnych przekroczy 12% w roku:", int(yr12))
fig, ax = plt.subplots()
ax.set_xlim([1995, 2015])
ax.set_ylim([5, 10])
line, = ax.plot([], [], color='red')
def draw_frame(years, percentages, i):
    x = years[:i]
    y = percentages[:i]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    predicted_y = slope * years + intercept
    ax.clear()
    ax.scatter(years, percentages)
    ax.plot(years, predicted_y, color='red')
    ax.set_xlim([1995, 2015])
    ax.set_ylim([5, 10])
    ax.set_xlabel('Rok')
    ax.set_ylabel('Procent bezrobotnych')
    ax.set_title('Regresja liniowa')
for j in range(20):
    for i in range(1, len(years)+1):
        draw_frame(years, percentages, i)
        plt.pause(0.5)
plt.show()