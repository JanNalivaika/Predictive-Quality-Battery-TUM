import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random as rd

df = pd.read_excel("../Data/Datensatz_Batteriekontaktierung.xlsx")

# print(df.index)
# row = df.loc[1, "Signal1_  1":"Signal1_112"].to_numpy()
# print(type(row))
# row2 = df.loc[2, "Signal1_  1":"Signal1_112"].to_numpy()
# row3 = np.append(row , row2)
# print(row3)
row_new = []

for i in range(5):
    row = df.loc[i, "Signal1_dn_  1":"Signal1_dn_112"].to_numpy()
    row_new = np.append(row, row_new)
plt.plot(row_new)
plt.show()



# def plot_examples(df: pd.DataFrame, n: int):
#     if n > len(df.index):
#         print("Please enter a numer smaller than ", str(len(df.index)))
#         return
#     examples = []
#
#     # df = df[df["not OK"] == 1]
#     # df.reset_index(inplace=True)
#
#     for i in range(n):
#         row = rd.randrange(len(df.index))
#         while row in examples:  # for unique examples
#             row = rd.randrange(len(df.index))
#         examples.append(row)
#         plt.figure()
#         plt.title("Sample " + str(row))
#         plt.plot(df.loc[row, "Signal1_  1":"Signal1_112"].tolist(), label="Signal 1")
#         plt.plot(df.loc[row, "Signal1_dn_  1":"Signal1_dn_112"].tolist(), label="Signal 1 DN")
#         plt.plot(df.loc[row, "Singal2_  1":"Singal2_112"].tolist(), label="Signal 2")
#         plt.legend()
#
#     plt.show()
#
#
# plot_examples(df, 5)
# # print(df.loc[4, "Singal2_108":"Singal2_112"].tolist())
# # print(df.columns[200])

