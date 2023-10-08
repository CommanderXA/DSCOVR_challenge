# DSCOVRY

NASA SpaceAppsChallenge: Develop the Oracle of DSCOVR

## Data

Our Deep Learning model had been trained on data from 2 datasets that we merged together:

- [raw data from the satellite](https://www.spaceappschallenge.org/develop-the-oracle-of-dscovr-experimental-data-repository/) - German Research Center for Geosciences
- [planetary k-index](https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt) - NASA

## Project structure

This project encorporates:

- `/app` - _*Backend*_
- `/nn` - _*Deep Learning model*_

Backend is written in Flask and the Deep Learning Model that we called `DSCOVR(Y)` was developed in PyTorch.

## Data Cleaning

We have used 2 scripts to clean the data, they are located in the `/scripts` directory.

- `kp_dataset_clean.py` used to clean the Kp indices data that contained a lot of unrelevant data to the problem in the first place. It also produces well structured file with the delimeters, makeing it possible for `pandas` to open it.
- `datasets_merge.py` iterates over the satellite raw data, for each entry it then iterates over the Kp indices data and if the entry is in range of time measurement of any Kp index, then it adds the Kp index column to the entry.
