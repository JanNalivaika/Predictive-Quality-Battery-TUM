import numpy as np
import pandas as pd

S1 = "S1.xlsx"
S1_DN = "S1.xlsx"
S2 = "S2.xlsx"
TH = "Threshold.xlsx"


S1 = np.asarray(pd.read_excel(io=S1))
S1_DN = np.asarray(pd.read_excel(io=S1_DN))
S2 = np.asarray(pd.read_excel(io=S2))
TH = np.asarray(pd.read_excel(io=TH))




print("done")
