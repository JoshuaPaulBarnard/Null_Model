"""
Author:  Joshua P. Barnard
Topic:    The Null Model (also known as the Mean Model or Baseline Model)

"""

import numpy as np
import random


def null_model( dataset, target_x = None, prediction_error = False, distribution = "normal", verbose = True, return_values = False ):
    input_data = dataset.copy()
    try:
        y_mean = np.mean( input_data )
        y_stndev = np.std( input_data )
        lower_bound_95 = y_mean - 1.96 * y_stndev
        upper_bound_95 = y_mean + 1.96 * y_stndev
    except:
        return print( "There was an error with reading in the data" )

    if verbose is True:
        print( "mean:", y_mean,  ", standard deviation: ",  y_stndev )
        print( "Which gives us a 95 percent chance for a score to fall between (",  lower_bound_95,  ", ",  upper_bound_95,  ").")

    if return_values is True:
        return y_mean, y_stndev, lower_bound_95, upper_bound_95

    if prediction_error is True:
        #   Create simulated data using monte carlo sampling for the specific distribution
        if distribution == "normal":
            simulated_data = [y_mean + r * y_stndev for r in np.random.standard_normal( len( input_data ) ) ]
        elif distribution == "gauss":
            simulated_data = [y_mean + r * y_stndev for r in np.random.normal( len( input_data ) ) ]
        elif distribution == "exponential":
            simulated_data = [y_mean + r * y_stndev for r in np.random.exponential( len( input_data ) ) ]
        elif distribution == "gamma":
            simulated_data = [y_mean + r * y_stndev for r in np.random.gamma( len( input_data ) ) ]
        else:
            #   To add more distributions, refer to:  https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator
            return print("invalid distribution selected")

        prediction_error = []
        if target_x is None:
            for counter_i in range( len( input_data ) ):
                prediction_error.append( input_data[counter_i] - simulated_data[counter_i] )
            total_error = np.sum([np.abs(e) for e in prediction_error])
            avg_prediction_error = total_error / len(prediction_error)
            if verbose is True:
                print("Our prediction of the actual data compared to sampled data is off by", avg_prediction_error,
                      "on average.")
            return avg_prediction_error, total_error

        elif isinstance(target_x, int) or isinstance(target_x, float) is True:
            actual_error = []
            for counter_i in range( len( simulated_data ) ):
                prediction_error.append( target_x - simulated_data[counter_i] )
            for counter_i in range( len( input_data ) ):
                actual_error.append( target_x - input_data[counter_i] )
            total_prediction_error = np.sum([np.abs(e) for e in prediction_error])
            avg_prediction_error = total_prediction_error / len(prediction_error)
            total_actual_error = np.sum([np.abs(e) for e in actual_error])
            avg_actual_error = total_actual_error / len(actual_error)
            if verbose is True:
                print("Our prediction of", target_x, "is off by", avg_prediction_error, "on average.")
                print("Our value of", target_x, "is actually off by an average of", avg_actual_error)
        else:
            return print( "Error:  target_x for prediction_error recevied an incompatible value.  It should receive an integer, float, or null." )
