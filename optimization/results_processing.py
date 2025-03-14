''' Plot the results of the optimization. '''
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import fsolve
import numpy as np
import os

def postprocess_results(model):
    ''' Postprocess the results of the optimization. '''

    plot_battery_evolution(model)

    plot_power_exchange(model)

    plot_probabilistic_power_schedule(model)


def plot_battery_evolution(model):
    ''' Plots the optimal battery evolution over time. '''

    e_exp = list(model.model.e_exp.get_values().values())
    e_nominal = list(model.model.e_nom.get_values().values())
    e_prob_max = list(model.model.e_max.get_values().values())
    e_prob_min = list(model.model.e_min.get_values().values())

    e_max = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_max)]
    e_min = [e_nom + e_prob for e_nom, e_prob in zip(e_nominal, e_prob_min)]

    time_e = [str(t) for t in model.model.time_e]
    ordered_time_e = model.model.time_e.ordered_data()

    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_e, e_nominal, '-', color='limegreen', linewidth=3, label='Nominal Battery State')
    ax.plot(time_e, e_exp, linewidth=2, color='navy', label='Expected Battery State')
    ax.plot(time_e, e_min, linewidth=2, color='magenta', label='Min/Max Battery State')
    ax.plot(time_e, e_max, linewidth=2, color='magenta')

    ax.axhline(y=model.e_limit_max, color='k', linestyle='--', linewidth='2', label='Battery Limits')
    ax.axhline(y=model.e_limit_min, color='k', linestyle='--', linewidth='2')

    ax.fill_between(time_e, e_min, e_max, color='limegreen', alpha=0.2)


    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, ordered_time_e)))
    plt.xticks(np.arange(0, len(e_max), 2), rotation=45)
    plt.ylim([-0.5, model.e_limit_max + 0.6])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.2)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 0.94), bbox_transform=ax.transAxes)
    plt.ylabel('Battery Storage [kWh]')
    file_path = get_file_path('battery_evolution.png')
    plt.savefig(file_path, dpi=200)
    #plt.show()


def plot_power_exchange(model):
    ''' Plots the power exchange over time.'''

    pg_nom = list(model.model.pg_nom.get_values().values())
    pb_nom = list(model.model.pb_nom.get_values().values())
    pl = [model.model.pl[t] for t in model.model.time]


    pg_exp_low_cond = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value / model.model.prob_low[t].value for t in model.model.time]
    pg_exp_high_cond = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value / model.model.prob_high[t].value for t in model.model.time]

    prob_low = list(model.model.prob_low.get_values().values())
    prob_high = list(model.model.prob_high.get_values().values())

    time = list(model.model.time)

    plt.figure(figsize=(10, 6))
    plt.plot(time, pb_nom, linewidth=4, label='Nominal Battery Power')
    plt.plot(time, pl, linewidth=4, label='Expected Prosumption')
    plt.plot(time, pg_nom, linewidth=4, label='Grid Power')
    plt.plot(time, pg_exp_low_cond, linewidth=1, label='Expectation of downward deviations from Grid Power (Conditional)', color='navy')
    plt.plot(time, pg_exp_high_cond, linewidth=1, label='Expectation of upward deviations from Grid Power (Conditional)', color='navy')
    plt.plot(time, prob_low, linewidth=1, label='Probability of downward deviations from Grid Power', color='magenta')
    plt.plot(time, prob_high, linewidth=1, label='Probability of upward deviations from Grid Power', color='magenta')
    plt.grid()
    plt.legend()
    plt.title('Power Exchange')
    plt.xlabel('Time')
    plt.ylabel('Power [kW]')
    file_path = get_file_path('power_exchange.png')
    plt.savefig(file_path, dpi=200)
    # plt.show()



def plot_probabilistic_power_schedule(model, quantiles=[0.05, 0.95]):
    ''' Plot the probabilistic power schedule. '''

    time = [str(t.hour) for t in model.model.time]
    pg_nom = list(model.model.pg_nom.get_values().values())

    quant_low, quant_high  = quantiles

    # Compute the quantiles of the prosumption uncertainty
    prosumption_low = []
    prosumption_high = []
    for t in model.model.time:
        # Function with form cdf(x) - quantile = 0
        func_temp_low = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t]) - quant_low
        func_temp_high = lambda x: model.cdf_numpy(x, *model.model.pdf_weights[t]) - quant_high

        prosumption_low_temp = fsolve(func_temp_low, x0=-0.5)[0]
        prosumption_high_temp = fsolve(func_temp_high, x0=0.5)[0]

        prosumption_low.append(prosumption_low_temp)
        prosumption_high.append(prosumption_high_temp)

    # Compute the quantile values of the truncated distribution (i.e. the grid power uncertainty)
    pg_truncated_quantile_low = []
    pg_truncated_quantile_high = []
    for i in range(len(prosumption_low)):
        pg_trunc_low_temp = prosumption_low[i] - model.model.x_low[model.model.time.at(i+1)].value
        pg_trunc_high_temp = prosumption_high[i] - model.model.x_high[model.model.time.at(i+1)].value

        pg_truncated_quantile_low.append(pg_trunc_low_temp)
        pg_truncated_quantile_high.append(pg_trunc_high_temp)

    # Shift the uncertaintites to represent quantiles of the deviations from the nominal grid power
    pg_quantile_low = [pg_nomi + deviation if deviation < 0 else pg_nomi for pg_nomi, deviation in zip(pg_nom, pg_truncated_quantile_low)]
    pg_quantile_high = [pg_nomi + deviation if deviation > 0 else pg_nomi for pg_nomi, deviation in zip(pg_nom, pg_truncated_quantile_high)]

    # Unconditional Expected Deviations
    pg_exp_low = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value for t in model.model.time]
    pg_exp_high = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value for t in model.model.time]

    # Conditional ExpectedDeviations
    pg_exp_low_cond = [model.model.pg_nom[t].value + model.model.exp_pg_low[t].value / model.model.prob_low[t].value for t in model.model.time]
    pg_exp_high_cond = [model.model.pg_nom[t].value + model.model.exp_pg_high[t].value / model.model.prob_high[t].value for t in model.model.time]

    # Ensure that the conditional expected deviations are within the quantiles for visualization purposes.
    # With specifying the quantiles, we limit the range that interests us. Thus, the conditional expected deviations
    # should be within the quantiles. For highly asymmetrical distributions, this could not hold. Thus, for asymmetrical
    # PDFs, either the quantiles should be large (5%-95% or 1%-99%) or the following 5 lines should be commented out.
    # Otherwise, a plot that is easily misinterpreted could be generated.
    for i in range(len(pg_exp_low_cond)):
        if pg_exp_low_cond[i] < pg_quantile_low[i]:
            pg_exp_low_cond[i] = pg_quantile_low[i]
        if pg_exp_high_cond[i] > pg_quantile_high[i]:
            pg_exp_high_cond[i] = pg_quantile_high[i]

    time.append(str(int(time[-1])+0.95))  # Add a last time point for the step plot.
    pg_nom.append(pg_nom[-1])  # Repeat the last y-value (nominal grid power) for the step plot.
    pg_exp_low_cond.append(pg_exp_low_cond[-1])  
    pg_exp_high_cond.append(pg_exp_high_cond[-1])
    pg_quantile_low.append(pg_quantile_low[-1])
    pg_quantile_high.append(pg_quantile_high[-1])

    # Plot the results
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.step(time, pg_nom, label='Nominal Grid Power', color='limegreen', linewidth=5, where='post')
    #ax.step(time, pg_exp_low, label='Expectation of Deviations', color='aqua', linewidth=2, where='post')
    #ax.step(time, pg_exp_high, color='aqua', linewidth=2, where='post')
    ax.step(time, pg_exp_low_cond, label='Expectation of Deviations', color='navy', linewidth=2, where='post')
    ax.step(time, pg_exp_high_cond, color='navy', linewidth=2, where='post')

    ax.step(time, np.ravel(pg_quantile_low), '--', label=f'{int(100*quant_low)} - {int(100*quant_high)}% Quantile', color='black', linewidth=1.5, where='post')
    ax.step(time, np.ravel(pg_quantile_high), '--', color='black', linewidth=1.5, where='post')

    cmap = plt.get_cmap('PuRd') # 'viridis'
    probs_low = list(model.model.prob_low.get_values().values())
    probs_high = list(model.model.prob_high.get_values().values())

    norm = mcolors.Normalize(vmin=0.0, vmax=0.6) # Specify the range of the colormap => for symmetric pdfs, 0.5 is the maximum.

    for i in range(len(time) - 1):
        color_value_low = cmap(norm(probs_low[i]))
        color_value_high = cmap(norm(probs_high[i]))
        ax.fill_between(time[i:i+2], pg_nom[i:i+2], pg_quantile_low[i:i+2], color=color_value_low, alpha=1.0, step='post')
        ax.fill_between(time[i:i+2], pg_nom[i:i+2], pg_quantile_high[i:i+2], color=color_value_high, alpha=1.0, step='post')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Probability of Deviations', rotation=270, labelpad=15)

    plt.ylim([-6.5, 2.5])
    plt.xlim([-0.8, len(probs_low) + 0.3])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=1.05, top=0.95, bottom=0.2)

    ordered_time = model.model.time.ordered_data()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: custom_x_axis_formatter(x, pos, ordered_time)))
    plt.xticks(np.arange(0, len(probs_low), 2), rotation=45)

    plt.legend(loc='lower right')
    plt.ylabel('Grid Power [kW]')
    file_path = get_file_path('dispatch_schedule.png')
    plt.savefig(file_path, dpi=200)
    plt.show()

    pass


def custom_x_axis_formatter(x, pos, time):
    ''' Custom formatter for the x-axis. '''
    # Convert position (x) to index and get the corresponding timestamp
    index = int(x) if x < len(time) else len(time) - 1
    date = time[index]  # Access the corresponding Timestamp
    if date.hour == 0 or index == 0:  # First tick of the day
        return date.strftime('%d/%m/%Y\n%H:%M')  # Show "DD/MM HH"
    else:
        return date.strftime('%H:%M')  # Otherwise show "HH:MM"


def get_file_path(filename):
    ''' Get file path to store plots in a temporary folder for logging in mlflow. '''
    temp_folder = os.path.join(os.getcwd(), 'temporary_folder')
    os.makedirs(temp_folder, exist_ok=True)

    return os.path.join(temp_folder, filename)


def validate_expected_values(model):
    ''' In the model, the expected value of the battery power uncertainty should match the one of the grid (see paper 
    Equation 8). This function checks said condition. The most probable cause for violating this condition are poor 
    integration bounds or breakpoints in very steep PDFs. Try to increase number of bkpts or adjust the bounds. '''

    pb_tilde = model.model.pb_tilde.get_values().values()

    pb_tilde = [model.model.pb_tilde[t].value for t in model.model.time]
    pg_low = [model.model.exp_pg_low[t].value for t in model.model.time]
    pg_high = [model.model.exp_pg_high[t].value for t in model.model.time]

    pg_expected = [pg_low[i] + pg_high[i] for i in range(len(pg_low))]
  
    print("Sum of expected values for validation purposes:")
    for i in range(len(pb_tilde)):
        print(f'Sum of expected values at time {list(model.model.time)[i]}: {pb_tilde[i] + pg_expected[i]}')

