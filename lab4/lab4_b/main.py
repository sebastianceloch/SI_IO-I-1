import pandas as pd
from pyarc import TransactionDB, CBA

data = pd.read_csv("diabetes.txt", sep=" ", header=None)
data.columns = ["a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "dec"]
txns = TransactionDB.from_DataFrame(data)

cba = CBA(algorithm="m1")
cba.fit(txns)

rules = cba.clf.rules

for rule in rules:
    print(rule)
