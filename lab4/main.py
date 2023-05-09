import pandas as pd

def indiscernibility(attribute, table):

    index = {}

    for i in table.index:
        attr_values = []
        for j in attribute:
            attr_values.append(table.loc[i, j])

        key = "".join(str(k) for k in (attr_values))

        if key in index:
            index[key].add(i)
        else:
            index[key] = set()
            index[key].add(i)

    return list(index.values())


def lower_approximation(R, X):

    l_approx = set()

    for i in range(len(X)):
        for j in range(len(R)):

            if R[j].issubset(X[i]):
                l_approx.update(R[j])

    return l_approx

def gamma_measure(describing_attributes, attributes_tbd, U, tab):

    f_ind = indiscernibility(describing_attributes, tab)
    t_ind = indiscernibility(attributes_tbd, tab)
    f_lapprox = lower_approximation(f_ind, t_ind)

    return len(f_lapprox) / len(U)

def quick_reduct(C, D, tab):

    red = set()

    gamma_C = gamma_measure(C, D, tab.index, tab)
    gamma_R = 0

    while gamma_R < gamma_C:
        T = red

        for x in set(C) - red:
            feature = set()
            feature.add(x)
            new_red = red.union(feature)
            gamma_new_red = gamma_measure(new_red, D, tab.index, tab)
            gamma_T = gamma_measure(T, D, tab.index, tab)

            if gamma_new_red > gamma_T:
                T = red.union(feature)

        red = T
        gamma_R = gamma_measure(red, D, tab.index, tab)

    return red

figure1 = pd.read_csv("fig1.csv")
figure2 = pd.read_csv("fig2.csv")
print(quick_reduct({"a", "b", "c", "d"}, {"dec"}, figure1))
print(quick_reduct({"a1", "a2", "a3"}, {"dec"}, figure2))
