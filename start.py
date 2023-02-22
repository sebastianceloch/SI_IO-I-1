import numpy as np
import pandas as pd
info = np.loadtxt('_info-data-discrete.txt', dtype="str")
diabetes = np.loadtxt('diabetes.txt')
diabetes_type = np.loadtxt('diabetes-type.txt', dtype="str")

print(diabetes_type)

decision_classes = np.genfromtxt("_info-data-discrete.txt", dtype="str")
print(decision_classes[9])

print(diabetes.max(axis=0))
print(diabetes.min(axis=0))
print(np.unique(diabetes,axis=0))
print(np.std(diabetes))

