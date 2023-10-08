from datetime import timedelta

import pandas as pd
from tqdm import tqdm

year: int = 2023
print("Merging year: ", year)

raw_data = pd.read_csv(
    "./data/data_" + str(year) + ".csv",
    delimiter=",",
    parse_dates=[0],
    na_values="0",
    header=None,
)

raw_data.fillna(0, inplace=True)

# add a new column for kp
raw_data[55] = 0.0

kp_data = pd.read_csv(
    "./data/k_index_" + str(year) + ".csv",
    delimiter=";",
    parse_dates=[0],
    na_values=0,
    header=None,
)

time_future = timedelta(hours=3)
time_delta = timedelta(hours=1, minutes=50)
kp_row_index = 0


# get planetary k-index
def get_kp_future(timestamp) -> float:
    global kp_row_index
    for i in range(kp_row_index, len(kp_data), 1):
        ts = kp_data.iloc[i][0].to_pydatetime()
        if ts <= (timestamp + time_future) <= (ts + time_delta):
            kp_row_index = i
            return kp_data.iloc[i][1]
    return None


# iterate over the rows of raw data from the satellite and assign k-index
# to each row that lies in the range of data from k index
for i in tqdm(range(len(raw_data))):
    rd = raw_data.iloc[i]
    kp_future = get_kp_future(rd[0].to_pydatetime())
    if kp_future == None:
        raw_data.iloc[i, 55] = -1.0
        continue
    raw_data.iloc[i, 55] = kp_future

# delete all rows that were not in the range
raw_data = raw_data[raw_data[55] > -1]
print(raw_data[0:100])
raw_data.to_csv("./data/data2_" + str(year) + ".csv", header=False, index=False)
