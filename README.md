# DSCOVRY

NASA SpaceAppsChallenge: Develop the Oracle of DSCOVR

## Requirements

- python >= 3.10
- npm >= 9.2.0

## To run

- Create virtual environment:

```sh
python3 -m venv ./venv
```

- Activate the environment

  - for linux:

  ```sh
  source ./venv/bin/activate
  ```

  - for windows:

  ```sh
  .\venv\activate
  ```

- Install the dependencies:

  - python:

  ```sh
  pip install -r requirements.txt
  ```

  - npm:

  ```sh
  cd ./client && npm i
  ```

- Download [data](https://github.com/CommanderXA/dscovry/releases/tag/data) and place it inside `/data` directory _*(create it beforehand)*_ in the project root.
- Download [model](https://github.com/CommanderXA/dscovry/releases/tag/model) and place it inside `/models` directory _(create it beforehand)_ in the project root.

- Run server:

```sh
python run.py
```

- Run client:

```sh
cd ./client && npm start
```

## Project structure

This project encorporates:

- `/app` - _*Backend*_
- `/nn` - _*Deep Learning model*_

Backend is written in Flask and the Deep Learning Model that we called `DSCOVR(Y)` was developed in PyTorch.

## Data

Our Deep Learning model had been trained on data from 2 datasets that we merged together:

- [raw data from the satellite](https://www.spaceappschallenge.org/develop-the-oracle-of-dscovr-experimental-data-repository/) - German Research Center for Geosciences
- [planetary k-index](https://www-app3.gfz-potsdam.de/kp_index/Kp_ap_since_1932.txt) - NASA

Short names for datasets:

- `RDS_D` - _*raw data from the satellite dataset*_
- `KP_D` - _*planetary k-index dataset*_

## Data Cleaning

`RDS_D` has data for each minute of a year, whereas `KP_D` has data for each 3 hour period. Moreover `KP_D` data entry is not for singe discrete hour, it is a range of 1 hour and 50 minutes.

This gives as:

- `RDS_D` - data for each minute
- `KP_D` - aggregated? data for 1 hour and 50 minutes of each 3 hour period (1 hour and 10 minutes thereby is a dark spot)

Basically, we have checked each `RDS_D` entry if it's in range of time (1 hour and 50 minutes) in any `KP_D` period of observations.

We have used 2 scripts to clean the data, they are located in the `/scripts` directory.

- `kp_dataset_clean.py` used to clean the Kp indices data that contained a lot of unrelevant data to the problem in the first place. It also produces well structured file with the delimeters, makeing it possible for `pandas` to open it.
- `datasets_merge.py` iterates over the satellite raw data, for each entry it then iterates over the Kp indices data and if the entry is in range of time measurement of any Kp index, then it adds the Kp index column to the entry.

## Model Architecture

DSCIVR(Y) model has 53 inputs and 1 output.
It has 1,175,681 parameters.

The architecture:

- linear (53, 128)
- batch_norm (128)
- linear1 (128, 256)
- batch_norm (256)
- linear2 (256, 256)
- batch_norm (256)
- lstm (256, 256, 2 layers)
- linear3 (256, 64)
- linear4 (64, 1)

## License

Plantary k-index data is subjected to `CC BY 4.0` [license](https://creativecommons.org/licenses/by/4.0/).

The license of our project is `Apache 2.0`.
