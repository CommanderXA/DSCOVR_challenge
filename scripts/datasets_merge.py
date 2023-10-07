from datetime import timedelta

import pandas as pd
from tqdm import tqdm

year: int = 2017
print("Merging year: ", year)

raw_data = pd.read_csv(
    "./data/dsc_fc_summed_spectra_" + str(year) + "_v01.csv",
    delimiter=",",
    parse_dates=[0],
    na_values="0",
    header=None,
)

# add a new column for kp
raw_data[54] = 0.0

kp_data = pd.read_csv(
    "./data/k_index_" + str(year) + ".csv",
    delimiter=";",
    parse_dates=[0],
    na_values=0,
    header=None,
)


def prepare_data() -> None:
    # drop rows where magnetic field vector contains NaN
    raw_data.dropna(subset=[1, 2, 3], inplace=True)
    fc_range = [a for a in range(4, 54, 1)]
    # drop rows where fc data has > 50% NaNs
    raw_data.dropna(subset=fc_range, thresh=0.5, inplace=True)
    # convert NaN to 0
    raw_data.fillna(0, inplace=True)


prepare_data()

time_delta = timedelta(hours=1, minutes=50)
kp_row_index = 0


# get planetary k-index
def get_kp(timestamp) -> float:
    global kp_row_index
    for i in range(kp_row_index, len(kp_data), 1):
        ts = kp_data.iloc[i][0].to_pydatetime()
        if ts <= timestamp <= (ts + time_delta):
            kp_row_index = i
            return kp_data.iloc[i][1]
    return None


# iterate over the rows of raw data from the satellite and assign k-index
# to each row that lies in the range of data from k index
for i in tqdm(range(len(raw_data))):
    rd = raw_data.iloc[i]
    kp = get_kp(rd[0].to_pydatetime())
    if kp == None:
        raw_data.iloc[i, 54] = -1.0
        continue
    raw_data.iloc[i, 54] = kp

# delete all rows that were not in the range
raw_data = raw_data[raw_data[54] > -1]
print(raw_data[0:100])
raw_data.to_csv("./data/data_" + str(year) + ".csv", header=False, index=False)
