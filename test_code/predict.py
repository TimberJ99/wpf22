import time
import numpy as np
import linear_predict
import gru_single_model_predict
import lightgbm_predict
import gru_multi_output_predict


def multi_output_forecast(setting):
    """
    Desc:
        Forecasting the wind power in lightgbm and gru multi output model
    Args:
        settings:
    Returns:
        The predictions
    """
    lgb_pred = lightgbm_predict.forecast(setting)
    gru_pred = gru_multi_output_predict.forecast(setting)
    result = (np.array(lgb_pred) * 5 + gru_pred * 5) / 10
    return result

def multi_days_forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in linear and gru single model
    Args:
        settings:
    Returns:
        The predictions
    """
    linear_pred_rsts = linear_predict.forecast(settings)
    gru_pred_rsts = gru_single_model_predict.forecast(settings)
    combine_pred = linear_pred_rsts * 0.59 + gru_pred_rsts * 0.41
    test_result = np.concatenate((linear_pred_rsts[:,:22], combine_pred[:,22:]), axis=1)
    pred = test_result.reshape(134, 288, 1)
    pred[pred < 0] = 0
    pred[pred > 1567.02] = 1567.02
    return pred

def forecast_all(setting):
    """
    Desc:
        Forecasting the wind power in all model
    Args:
        settings:
    Returns:
        The predictions
    """
    result = multi_output_forecast(setting)
    multi_days_forecast_predictions = multi_days_forecast(setting)
    result = (result * 7.5 + multi_days_forecast_predictions * 2.5) / 10
    result[result < 0] = 0
    result[result > 1567.02] = 1567.02
    return result


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    start_time = time.time()
    result = forecast_all(settings)
    print("\nElapsed time for predicting 134 turbines is {} secs".format(end_time - start_time))
    return result