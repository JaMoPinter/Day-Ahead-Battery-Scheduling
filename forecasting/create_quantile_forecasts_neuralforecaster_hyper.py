import pandas as pd
from datetime import datetime
import os
import numpy as np  

from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.auto import AutoPatchTST
from neuralforecast import NeuralForecast
from ray import tune

# By default the hyperparameter tuning is disabled to enable it set HYPERPARAMETER_TUNING = True
HYPERPARAMETER_TUNING = False


# Read in Data
data = pd.read_csv('https://data.open-power-system-data.org/household_data/2020-04-15/household_data_60min_singleindex.csv', date_format='%Y-%m-%dT%H:%M:%SZ', index_col = "utc_timestamp", parse_dates=True , sep=',')
data.index = pd.to_datetime(data.index)

# parse time stamp from utc to local time utc to cest
data.index = data.index + pd.Timedelta(hours=2)

# rename index to cest_timestamp
data.index.name = 'cest_timestamp'
valid_buildings = ['residential4']

def get_building_prosumption_data(data, building):
    prosumption = data[f"DE_KN_{building}_grid_import"].diff(1) - data[f"DE_KN_{building}_grid_export"].diff(1)
    # fix it that the beginning of the interval is denoted not the end
    prosumption.index = prosumption.index - pd.Timedelta(hours=1)
    return prosumption.asfreq('1h').dropna()

data_selected_buildings_prosumption = [(building, get_building_prosumption_data(data, building)) for building in valid_buildings ]

os.makedirs('../data/ground_truth', exist_ok=True)
for building, prosumption in data_selected_buildings_prosumption:
    prosumption.to_csv(f"../data/ground_truth/{building}_prosumption_fixed_diff.csv")

# Define train/validation split
train_split = 365 * 24 - 1  # 1 year of hourly data
validation_split = 200 * 24  # 200 Days

# List to store results for multiple buildings
forecasts = []

for building, prosumption in data_selected_buildings_prosumption:
    # Ensure the time index is correctly formatted
    prosumption.index = pd.to_datetime(prosumption.index)

    # Filter for 06:00 timestamps
    prosumption_filtered = prosumption[prosumption.index.hour == 6]
    prosumption = prosumption[prosumption_filtered.index[0]:]

    # Train/validation split
    training_cutoff = prosumption.index[train_split]
    validation_cutoff = prosumption.index[train_split + validation_split]


    # Create DataFrame with proper format for neuralforecast
    df = pd.DataFrame({
        'ds': prosumption.index,  # Timestamps
        'y': prosumption.values,  # target variable
        'unique_id': building  # Unique identifier for different time series
    })

    df['ds'] = pd.to_datetime(df['ds'])
    

    # Split into train and validation
    df_train = df[df['ds'] <= validation_cutoff]

    # reset index
    df_train.reset_index(drop=True, inplace=True)


# Define the quantiles used in MQLoss    
quantiles = [np.round(i,2) for i in np.arange(0.01, 1, 0.01)]

# Define the model
horizon = 24
random_seed=123456789

# Define the ajusted configuration for the patch TST model
config = AutoPatchTST.default_config.copy()
config["max_steps"]= tune.choice([5000])
# 3 Days 7 Days and 10 Days
config['input_size_multiplier'] = [3, 7, 10]
config["input_size"] = tune.choice(
            [horizon * x for x in config["input_size_multiplier"]]
        )
config["step_size"] = tune.choice([1, horizon])
config["early_stop_patience_steps"] = 5
config["random_seed"] = random_seed
del config["input_size_multiplier"]
num_samples = 2500

# Marks this as the best run from the hyperparameter tuning process.
#
# If set to True, hyperparameter tuning is performed to optimize runtime complexity and reproducibility.
# If set to False, the best hyperparameter configuration found during tuning is applied,
# and the number of samples is set to 1.


if HYPERPARAMETER_TUNING is False:
    config = { 
                'hidden_size': 128,
                'n_heads': 4,
                'patch_len': 16,
                'learning_rate': 0.00012413796427583382,
                'scaler_type': 'robust',
                'revin': False,
                'max_steps': 5000,
                'batch_size': 32,
                'windows_batch_size': 1024,
                'random_seed': 123456789,
                'input_size': 240,
                'step_size': 1,
                'early_stop_patience_steps': 5}
    num_samples = 1



# Define models
models = [

   AutoPatchTST(h=horizon,
                 loss=MQLoss(quantiles=quantiles), config=config, num_samples=num_samples)

]

# Initialize NeuralForecast with all models
nf = NeuralForecast(models=models, freq='H')

# Train models on training data
nf.fit(df_train, val_size=validation_split)
    
results = nf.models[0].results.get_dataframe()
results = results.sort_values('loss', ascending=True)

# get input size for the prediction to forecast
input_size = results["config/input_size"].values[0]

# start the prediction
time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# create a folder for the results
os.makedirs(f"results/{time_stamp}_fixed_diff", exist_ok=True)

results.to_csv(f"results/{time_stamp}_fixed_diff/results_hyper_{time_stamp}.csv")

# First prediction at 06:00 the other predictions are made for subsequent task which need every hour a forecast
for i in range(1, 25):
    # let the first prediction begin at the first timestamp of the test data
    start_prediction = validation_cutoff + pd.Timedelta(hours=i)
    end = df['ds'].iloc[-1]

    print(f"Start prediction at {start_prediction} until {end}")
    # Generate time steps every 24 hours
    time_steps_to_predict = pd.date_range(start=start_prediction, end=df['ds'].iloc[-1], freq='d')

    # begin to predict the next 24 hours for each timestamp 
    predictions = pd.DataFrame()

    for time_step in time_steps_to_predict:

        # get the last input size hours of data
        last_hours = df[df['ds'] < time_step].tail(input_size)

        # predict the next 24 hours
        prediction = nf.predict(df=last_hours,verbose=False)

        # add the prediction to the predictions DataFrame
        predictions = pd.concat([predictions, prediction])

        
    predictions.set_index('ds', inplace=True) 
    # plot one day
    predictions.filter(like='PatchTST', axis=1)

    # get the index for a new df 

    predictions.index = pd.to_datetime(predictions.index)

    # get the values and sort them accending

    prediction_values = predictions.filter(like='PatchTST', axis=1).values

    # sort prediction values
    prediction_values.sort(axis=1)

    # quantiles as collumns

    result_df = pd.DataFrame(prediction_values, index=predictions.index, columns=quantiles)

    # name index cest_timestamp
    result_df.index.name = 'cest_timestamp'

    hour = result_df.index[0].hour

    # save the result to a csv file
    result_df.to_csv(f"results/{time_stamp}_fixed_diff/patch_tst_{time_stamp}_prosumption_hour_{hour}.csv")

