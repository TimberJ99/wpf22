import os
import numpy as np
import paddle
from base_model import BaselineGruModel
from common import Experiment
from wind_turbine_data import WindTurbineData

import warnings
warnings.filterwarnings('ignore')

def forecast_n(input_len, output_len, model_path, experiment, test_turbines, train_data, turb_id):
    # type: (Experiment, TestData, WindTurbineData) -> np.ndarray
    """
    Desc:
        Forecasting the power of one turbine
    Args:
        experiment:
        test_turbines:
        train_data:
    Returns:
        Prediction for one turbine
    """
    args = experiment.get_args()
    tid = turb_id#args["turbine_id"]
    args["output_len"] = output_len
    args["input_len"] = input_len
    model = BaselineGruModel(args)

    path_to_model = os.path.join(model_path, "model_{}".format(str(tid)))
    model.set_state_dict(paddle.load(path_to_model))
    model.eval()
 
    test_x, test_df = test_turbines.get_turbine(tid)

    scaler = train_data.get_scaler(tid)
    #print(tid,":scaler:", scaler.mean,scaler.std)
    test_x = scaler.transform(test_x)
    
    last_observ = test_x[-args["input_len"]:]
    seq_x = paddle.to_tensor(last_observ)
    sample_x = paddle.reshape(seq_x, [-1, seq_x.shape[-2], seq_x.shape[-1]])
    prediction = experiment.inference_one_sample(model, sample_x)
    prediction = scaler.inverse_transform(prediction)
    prediction = prediction[0]
    return prediction.numpy()
    
    
def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power
    Args:
        settings:
    Returns:
        The predictions
    """
    predictions = []
    exp = Experiment(settings.copy())
    if DataSet.train_data is None:
        DataSet.train_data = exp.load_train_data()
    test_x = Experiment.get_test_x(settings)
    filepath = os.path.join(settings["checkpoints"], 'gru-multi-models')
    files = []
    for file in os.listdir(filepath):
        model_path = os.path.join(filepath, file)
        if os.path.isfile(model_path):
            continue
        files.append(file)

    files.sort(key=lambda x:int(x[x.find("_o")+2:x.find("_l")]), reverse=False)
    # use top 5 turbines
    turb_list = [0,5,10,2,4]
    prediction_turb_list = []
    for tid in turb_list:
        prediction =  np.zeros((288, 1))
        times = 1
        pre_output_len = 0
        for fname in files:
            model_path = os.path.join(filepath, fname)
            input_len = int(fname[fname.find("_i")+2:fname.find("_o")])
            output_len = int(fname[fname.find("_o")+2:fname.find("_l")])
            #print(output_len)
            one_pred = forecast_n(input_len, output_len, model_path, exp, test_x, DataSet.train_data, turb_id = tid)
            if times == 1:
                prediction[0:output_len,:] = one_pred
                
            else:
                prediction = merge_prediction(prediction, one_pred, pre_output_len, next_output_len = output_len)
            pre_output_len = output_len    
            times += 1
            paddle.device.cuda.empty_cache()
        prediction_turb_list.append(prediction)    
    prediction = np.mean(prediction_turb_list, axis=0)

 
    for i in range(0, settings["capacity"]):
        predictions.append(prediction)
    return np.array(predictions)

def merge_prediction(prediction, next_prediction, pre_output_len, next_output_len, coefficient=0.4):
    first_prediction_mean = np.mean(prediction[:pre_output_len])
    
    next_predictions_mean = np.mean(next_prediction[:pre_output_len])
    prediction[pre_output_len:next_output_len,:] = next_prediction[pre_output_len:] - (next_predictions_mean-first_prediction_mean)*coefficient
    
    return prediction

class DataSet:
    train_data = None
    step = 0
    def __init__(self):
        print("init")