''' This module contains the base class for the optimization model. '''
import pyomo.environ as pyo
import pandas as pd

from utils import simpsons_rule, cdf_formula, cdf_formula_numpy, pdf_formula, dynamic_bounds


class BaseOptimizationModel:
    ''' Base class for the optimization model. '''

    def __init__(self, input_data):
        self.name = "Base Optimization Model"
        self.model = pyo.ConcreteModel()
        self._initialize_class_vars(input_data)
        self._build_model()

    def _initialize_class_vars(self, data):
        ''' Initialize the parameter values of the model. '''
        self.t_inc = data['t_inc']

        self.e0 = data['e0']
        self.e_limit_min = data['e_min']
        self.e_limit_max = data['e_max']

        self.pb_limit_min = data['pb_min']
        self.pb_limit_max = data['pb_max']

        self.mu = data['mu']

        self.fc_exp = data['fc_exp']
        self.fc_weights = data['fc_weights']

        self.c11 = data['c11']
        self.c12 = data['c12']
        self.c21 = data['c21']
        self.c22 = data['c22']
        self.c31 = data['c31']
        self.c32 = data['c32']
        self.c_energy_final = data['c_energy_final']
        self.enforce_deterministic_battery = data['enforce_deterministic_battery']

        self.probability_distribution_name = data['pdf/cdf name']

        self.cdf = cdf_formula(self.probability_distribution_name)
        self.cdf_numpy = cdf_formula_numpy(self.probability_distribution_name) # needed for plotting, similar to self.cdf
        self.pdf = pdf_formula(self.probability_distribution_name)

        self.lowest_bound = -30.0  # minimal possible deviation from expectation in kW (for numerical integration)
        self.highest_bound = 30.0
    
    def _build_model(self):
        ''' Builds the optimization model. '''
        self._define_sets()

        self._define_parameters()

        self._define_decision_variables()

        self._define_constraints()

        self._define_objective_function()

    
    def _define_sets(self):
        ''' Defines pyomo sets for the model. '''
        time = self.fc_exp.index
        self.model.time = pyo.Set(initialize=time)
        self.model.time_e = pyo.Set(initialize=time.append(pd.Index([time[-1] + time.freq])))
        self.model.time_e0 = pyo.Set(initialize=time.to_list()[:1])

    def _define_parameters(self):
        ''' Defines pyomo parameters for the model. '''
        self.model.pl = pyo.Param(self.model.time, initialize=self.fc_exp.to_dict())
        self.model.e0 = pyo.Param(self.model.time_e0, initialize=self.e0)
        self.model.pdf_weights = pyo.Param(self.model.time, initialize=self.fc_weights.to_dict(), domain=pyo.Any)

        # Enable time varying objective coefficients
        if type(self.c31) == float:
            self.model.c31_varying = pyo.Param(self.model.time, initialize=self.c31)
        elif type(self.c31) == dict:
            c_varying = {}
            for i in range(len(self.model.time)):
                if str(i) in self.c31['c_extra']:
                    c_varying[self.model.time.at(i+1)] = self.c31['c_extra'][str(i)]
                else:
                    c_varying[self.model.time.at(i+1)] = self.c31['c_default']
            self.model.c31_varying = pyo.Param(self.model.time, initialize=c_varying)
        else:
            raise ValueError('c31 must be either a float or a dictionary with time varying values')
        
        #print('c31_varying:', self.model.c31_varying.extract_values())

        # Create dynamic bounds to minimize integration errors
        dynamic_bound_low, dynamic_bound_high = dynamic_bounds((self.lowest_bound, self.highest_bound), self.pdf, self.fc_weights)
        self.model.dynamic_bound_low = pyo.Param(self.model.time, initialize=dynamic_bound_low)
        self.model.dynamic_bound_high = pyo.Param(self.model.time, initialize=dynamic_bound_high)



    def _define_decision_variables(self):
        ''' Defines the pyomo decision variables of the model (they are not necessarily a Degree of Freedom). '''

        # Battery Energy
        self.model.e_exp = pyo.Var(self.model.time_e, domain=pyo.Reals)
        self.model.e_nom = pyo.Var(self.model.time_e, domain=pyo.Reals)
        self.model.e_min = pyo.Var(self.model.time_e, domain=pyo.NonPositiveReals)
        self.model.e_max = pyo.Var(self.model.time_e, domain=pyo.NonNegativeReals)

        # Grid Power
        self.model.pg_nom = pyo.Var(self.model.time, domain=pyo.Reals)
        self.model.pg_nom_plus = pyo.Var(self.model.time, domain=pyo.NonNegativeReals)
        self.model.pg_nom_minus = pyo.Var(self.model.time, domain=pyo.NonPositiveReals)

        # Battery Power
        #self.model.pb = pyo.Var(self.model.time, domain=pyo.Reals)
        self.model.pb_nom = pyo.Var(self.model.time, domain=pyo.Reals)
        self.model.pb_nom_plus = pyo.Var(self.model.time, domain=pyo.NonNegativeReals)
        self.model.pb_nom_minus = pyo.Var(self.model.time, domain=pyo.NonPositiveReals)

        self.model.pb_tilde = pyo.Var(self.model.time, domain=pyo.Reals)  # is the expected portion of the power uncertainty
        self.model.pb_tilde_plus = pyo.Var(self.model.time, domain=pyo.NonNegativeReals)
        self.model.pb_tilde_minus = pyo.Var(self.model.time, domain=pyo.NonPositiveReals)

        # Reserved Space in Battery for Power Uncertainties
        self.model.x_low = pyo.Var(self.model.time, domain=pyo.NonPositiveReals)
        self.model.x_high = pyo.Var(self.model.time, domain=pyo.NonNegativeReals)

        # Expected Value of continuous part of Power Uncertainties shifted into battery
        self.model.exp_pb_continuous = pyo.Var(self.model.time, domain=pyo.Reals)

        # Probability of Deviations from pg_nom
        self.model.prob_low = pyo.Var(self.model.time, domain=pyo.NonNegativeReals)  # todo: Either create bounds here or test that prob is between 0 and 1 afterwards
        self.model.prob_high = pyo.Var(self.model.time, domain=pyo.NonNegativeReals)  # todo: Either create bounds here or test that prob is between 0 and 1 afterwards

        # Expected Value of Deviations from pg_nom
        self.model.exp_pg_low = pyo.Var(self.model.time, domain=pyo.NonPositiveReals)
        self.model.exp_pg_high = pyo.Var(self.model.time, domain=pyo.NonNegativeReals)
                                                                                    


    def _define_constraints(self):
        ''' Defines the constraints of the model. This includes equality and inequality constraints. '''
        
        def constr_power_balance(model, t):
            ''' pg_nom[t] = pl[t] - pb_nom[t] '''
            return model.pg_nom[t] == model.pl[t] - model.pb_nom[t]
        self.model.constr_power_balance = pyo.Constraint(self.model.time, rule=constr_power_balance)

        ########################################### battery evolution ##################################################
        def constr_e_nom_evolution(model, t):
            ''' e_nom[t] = e_nom[t-1] - t * pb_nom[t-1] - t * mu * |pb_nom[t-1]| '''
            if t ==  model.time_e.first():
                return model.e_nom[t] == model.e0[t]
            else:
                t_prev = model.time_e.prev(t)
                return model.e_nom[t] == model.e_nom[t_prev] - self.t_inc * model.pb_nom[t_prev] - self.t_inc * self.mu * (model.pb_nom_plus[t_prev] - model.pb_nom_minus[t_prev]) 
        self.model.constr_e_nom_evolution = pyo.Constraint(self.model.time_e, rule=constr_e_nom_evolution)

        def constr_e_exp_evolution(model, t):
            ''' e_exp[t] = e_exp[t-1] - t * pb_nom[t-1] - t * mu * |pb_nom[t-1]| - t * pb_tilde[t-1] - t * mu * |pb_tilde[t-1]| '''
            if t == model.time_e.first():
                return model.e_exp[t] == model.e0[t]
            else:
                t_prev = model.time_e.prev(t)
                return model.e_exp[t] == model.e_exp[t_prev] - self.t_inc * model.pb_nom[t_prev] - self.t_inc * self.mu * (model.pb_nom_plus[t_prev] - model.pb_nom_minus[t_prev]) - self.t_inc * model.pb_tilde[t_prev] - self.t_inc * self.mu * (model.pb_tilde_plus[t_prev] - model.pb_tilde_minus[t_prev])
        self.model.constr_e_evolution = pyo.Constraint(self.model.time_e, rule=constr_e_exp_evolution)


        def constr_e_min_evolution(model, t):
            ''' e_min[t] = e_min[t-1] - t * x_high[t-1] - t * mu * x_high[t-1] '''
            if t == model.time_e.first():
                return model.e_min[t] == 0.0  # improvement: enable flexible initial value for uncertainty
            else:
                t_prev = model.time_e.prev(t)
                return model.e_min[t] == model.e_min[t_prev] - self.t_inc * model.x_high[t_prev] - self.t_inc * self.mu * model.x_high[t_prev]
        self.model.constr_e_min_evolution = pyo.Constraint(self.model.time_e, rule=constr_e_min_evolution)

        def constr_e_max_evolution(model, t):
            ''' e_max[t] = e_max[t-1] - t * x_low[t-1] + t * mu * x_low[t-1] '''
            if t == model.time_e.first():
                return model.e_max[t] == 0.0  # improvement: enable flexible initial value for uncertainty
            else:
                t_prev = model.time_e.prev(t)
                return model.e_max[t] == model.e_max[t_prev] - self.t_inc * model.x_low[t_prev] + self.t_inc * self.mu * model.x_low[t_prev]
        self.model.constr_e_max_evolution = pyo.Constraint(self.model.time_e, rule=constr_e_max_evolution)
        ################################################################################################################

        ######################################## battery constraints ###################################################
        def constr_e_limit_min(model, t):
            ''' e_nom[t] + e_min[t] >= e_limit_min '''
            return model.e_nom[t] + model.e_min[t] >= self.e_limit_min
        self.model.constr_e_limit_min = pyo.Constraint(self.model.time_e, rule=constr_e_limit_min)

        def constr_e_limit_max(model, t):
            ''' e_nom[t] + e_max[t] <= e_limit_max '''
            return model.e_nom[t] + model.e_max[t] <= self.e_limit_max
        self.model.constr_e_limit_max = pyo.Constraint(self.model.time_e, rule=constr_e_limit_max)


        def constr_pb_limit_min(model, t):
            ''' pb_nom[t] + x_low >= pb_limit_min '''
            return model.pb_nom[t] + model.x_low[t] >= self.pb_limit_min
        self.model.constr_pb_limit_min = pyo.Constraint(self.model.time, rule=constr_pb_limit_min)

        def constr_pb_limit_max(model, t):
            ''' pb_nom[t] + x_high <= pb_limit_max '''
            return model.pb_nom[t] + model.x_high[t] <= self.pb_limit_max
        self.model.constr_pb_limit_max = pyo.Constraint(self.model.time, rule=constr_pb_limit_max)
        ################################################################################################################

        ############################################ expected values ###################################################
        def constr_exp_pb_continuous(model, t):
            ''' exp_pb_continuous[t] = INTEGRAL[z*pdf(weights[t], z)] bounds=[x_low[t], x_high[t]] '''
            return model.exp_pb_continuous[t] == simpsons_rule(model.x_low[t], model.x_high[t], n=200, pdf=self.pdf, weights=model.pdf_weights[t], offset=0)
        self.model.constr_exp_pb_continuous = pyo.Constraint(self.model.time, rule=constr_exp_pb_continuous)

        def constr_pb_tilde(model, t):
            ''' pb_tilde = prob_low * x_low + prob_high * x_high + INTEGRAL(z*pdf(z)) bounds=[x_low, x_high] '''
            return model.pb_tilde[t] == model.prob_low[t] * model.x_low[t] + model.prob_high[t] * model.x_high[t] + model.exp_pb_continuous[t]
        self.model.constr_pb_tilde = pyo.Constraint(self.model.time, rule=constr_pb_tilde)


        def constr_exp_pg_low(model, t):
            ''' exp_pg_low[t] = INTEGRAL[z*pdf(weights[t], z+x_low[t])] bounds=[-inf, 0]'''
            return model.exp_pg_low[t] == simpsons_rule(model.dynamic_bound_low[t] , 0, n=200, pdf=self.pdf, weights=model.pdf_weights[t], offset=model.x_low[t])
        self.model.constr_exp_pg_low = pyo.Constraint(self.model.time, rule=constr_exp_pg_low)

        def constr_exp_pg_high(model, t):
            ''' exp_pg_high[t] = INTEGRAL[z*pdf(weights[t], z+x_high[t])] bounds=[0, inf]'''
            return model.exp_pg_high[t] == simpsons_rule(0, model.dynamic_bound_high[t], n=200, pdf=self.pdf, weights=model.pdf_weights[t], offset=model.x_high[t])
        self.model.constr_exp_pg_high = pyo.Constraint(self.model.time, rule=constr_exp_pg_high)
        ################################################################################################################

        ############################################# probability ######################################################
        def constr_prob_low(model, t):
            ''' prob_low[t] = CDF[t](x_low[t])'''
            return model.prob_low[t] == self.cdf(model.x_low[t], *model.pdf_weights[t]) 
        self.model.constr_prob_low = pyo.Constraint(self.model.time, rule=constr_prob_low)

        def constr_prob_high(model, t):
            ''' prob_high[t] = 1 - CDF[t](x_high[t])'''
            return model.prob_high[t] == 1 - self.cdf(model.x_high[t], *model.pdf_weights[t])
        self.model.constr_prob_high = pyo.Constraint(self.model.time, rule=constr_prob_high)
        ################################################################################################################

        ############################################# power splits #####################################################
        def constr_pg_split(model, t):
            ''' pg_nom = pg_nom_plus + pg_nom_minus '''
            return model.pg_nom[t] == model.pg_nom_plus[t] + model.pg_nom_minus[t]
        self.model.constr_pg_split = pyo.Constraint(self.model.time, rule=constr_pg_split)

        def constr_pg_relaxation(model, t):
            ''' -1e-8 <= pg_nom_plus * pg_nom_minus <= 0 '''
            return (-1e-8, model.pg_nom_plus[t] * model.pg_nom_minus[t], 0)
        self.model.constr_pg_relaxation = pyo.Constraint(self.model.time, rule=constr_pg_relaxation)

        
        def constr_pb_nom_split(model, t):
            ''' pb_nom = pb_nom_plus + pb_nom_minus '''
            return model.pb_nom[t] == model.pb_nom_plus[t] + model.pb_nom_minus[t]
        self.model.constr_pb_nom_split = pyo.Constraint(self.model.time, rule=constr_pb_nom_split)

        def constr_pb_nom_relaxation(model, t):
            ''' -1e-8 <= pb_nom_plus * pb_nom_minus <= 0 '''
            return (-1e-8, model.pb_nom_plus[t] * model.pb_nom_minus[t], 0)
        self.model.constr_pb_nom_relaxation = pyo.Constraint(self.model.time, rule=constr_pb_nom_relaxation)


        def constr_pb_tilde_split(model, t):
            ''' pb_tilde = pb_tilde_plus + pb_tilde_minus '''
            return model.pb_tilde[t] == model.pb_tilde_plus[t] + model.pb_tilde_minus[t]
        self.model.constr_pb_tilde_split = pyo.Constraint(self.model.time, rule=constr_pb_tilde_split)

        def constr_pb_tilde_relaxation(model, t):
            ''' -1e-8 <= pb_tilde_plus * pb_tilde_minus <= 0 '''
            return (-1e-8, model.pb_tilde_plus[t] * model.pb_tilde_minus[t], 0)
        self.model.constr_pb_tilde_relaxation = pyo.Constraint(self.model.time, rule=constr_pb_tilde_relaxation)
        ##########################################################################################################

        ############################################# Reproduce Case 1 ###########################################
        if self.enforce_deterministic_battery:
            def constr_x_low(model, t):
                return model.x_low[t] == 0
            self.model.constr_x_low = pyo.Constraint(self.model.time, rule=constr_x_low)

            def constr_x_high(model, t):
                return model.x_high[t] == 0
            self.model.constr_x_high = pyo.Constraint(self.model.time, rule=constr_x_high)
        ##########################################################################################################



    def _define_objective_function(self):
        def objective(model):
            return sum(
                self.c11 * model.pg_nom_plus[t] **2 
                + self.c12 * model.pg_nom_plus[t] 
                + self.c21 * model.pg_nom_minus[t] **2 
                + self.c22 * model.pg_nom_minus[t] # Careful with the sign here (If c22 !=0) => Does minus make more sense?
                - model.c31_varying[t] * model.prob_low[t] * model.exp_pg_low[t] 
                + self.c32 * model.prob_high[t] * model.exp_pg_high[t]
                #- self.c31 * model.exp_pg_low[t] 
                #+ self.c32 * model.exp_pg_high[t]
                for t in model.time) + self.c_energy_final * (model.e_exp[self.model.time_e.last()] - self.e_limit_max/2)**2
        self.model.objective = pyo.Objective(rule=objective, sense=pyo.minimize)



    def solve(self):
        solver = pyo.SolverFactory('ipopt')
        #solver.options['mu_strategy'] = 1       
        #solver.options['tol'] = 1e-8            
        #solver.options['acceptable_tol'] = 1e-8 
        #solver.options['max_step'] = 1e-1 
        solver.options['max_iter'] = 5000
        result = solver.solve(self.model, tee=True)
        return result
    