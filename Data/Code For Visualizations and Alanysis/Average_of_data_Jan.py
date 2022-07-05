import numpy as np
import pandas as pd

S1 = "../S1.xlsx"
S1_DN = "../S1_DN.xlsx"
S2 = "../S2.xlsx"

TH1 = "../OOT/S1_OOT_ONLY.xlsx"
TH1_DN = "../OOT/S1_DN_OOT_ONLY.xlsx"
TH2 = "../OOT/S2_OOT_ONLY.xlsx"


S1 = np.asarray(pd.read_excel(io=S1))
S1_DN = np.asarray(pd.read_excel(io=S1_DN))
S2 = np.asarray(pd.read_excel(io=S2))
TH1 = np.asarray(pd.read_excel(io=TH1))
TH1_DN = np.asarray(pd.read_excel(io=TH1_DN))
TH2 = np.asarray(pd.read_excel(io=TH2))





print("done")
