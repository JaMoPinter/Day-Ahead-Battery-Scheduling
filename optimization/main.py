''' Main script to start an optimization process. '''

import sys
sys.path.append('../')
from input_data import load_forecasts, load_params, preprocess_data
from optimization_model import BaseOptimizationModel
from results_processing import postprocess_results, validate_expected_values
from experiment_tracking import start_experiment, log_data, end_experiment, log_results

def main():
    # Example 1: Normal Distribution
    fc_folder = 'data/parametric_forecasts/normal_dist_forecast_2024-10-31/'
    params_path = 'data/parameters/params_normal_dist.json'
    timeframe = ['2017-05-18 06:00:00', '2017-05-19 05:00:00']

    # Example 2: Sum of two logistic functions (Recreate paper results)
    #fc_folder = 'data/parametric_forecasts/s2l_dist_forecast_2024-10-02/' 
    #params_path = 'data/parameters/params_case2.json'
    #timeframe = ['2017-05-18 06:00:00', '2017-05-19 05:00:00']  # Forecasted nighttime PDFs can be too tight for plotting or solving the optimization model (See limitations section in the paper).
                                                                # Simple solution: Either choose a different date, adjust the timeframe to exclude nighttime (e.g. 06:00-23:00), or use gaussian distribution.
                                                                # Better solution: Redo the forecasts with additional constraints, or approximate nighttime PDFs with a different distribution (GMM looks promising).

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
