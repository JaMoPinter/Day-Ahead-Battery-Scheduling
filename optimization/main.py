''' Main script to start an optimization process. '''

import sys
sys.path.append('../')
from input_data import load_forecasts, load_params, preprocess_data
from optimization_model import BaseOptimizationModel
from results_processing import postprocess_results, validate_expected_values
from experiment_tracking import start_experiment, log_data, end_experiment, log_results

def main():
    # Example 1: Normal Distribution
    # fc_folder = 'data/parametric_forecasts/normal_dist_forecast_2025-03-14/'
    # params_path = 'data/parameters/params_normal_dist.json'
    # timeframe = ['2017-05-18 06:00:00', '2017-05-19 05:00:00']

    # Example 2: Gaussian Mixture Model - 2 components (Recreate paper results)
    fc_folder = 'data/parametric_forecasts/gmm2_forecast_2025-03-14_WindowSize5_patch_tst_prosumption_10_03_2025/' 
    params_path = 'data/parameters/params_gmm2_case1.json'
    timeframe = ['2017-05-18 06:00:00', '2017-05-19 05:00:00']  


    # Load data
    forecasts = load_forecasts(fc_folder, timeframe=timeframe)
    params = load_params(params_path)
    input_data = preprocess_data(forecasts, params)


    # Track the experiment in MLflow
    # start_experiment(params['experiment_name'])
    # log_data(input_data)


    # Run the optimization model
    model = BaseOptimizationModel(input_data)
    model.solve()


    # Process and visualize the results
    validate_expected_values(model)
    postprocess_results(model)
    

    # Track results in MLflow
    # log_results(model.model)
    # end_experiment()


    print('Fin')



if __name__ == '__main__':
    main()
