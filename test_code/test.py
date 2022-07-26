
settings = {
        "path_to_test_x": r"./datasets/sdwpf_baidukddcup2022_test_toy/test_x/0001in.csv",
        "path_to_test_y": r"./datasets/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": r"./datasets",
        "filename": "wtbdata_245days.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "models",
        "input_len": 144,
        "output_len": 288,
        "start_col": 3,
        "in_var": 10,
        "out_var": 1,
        "day_len": 144,
        "train_size": 223,
        "val_size": 20,
        "total_size": 245,
        "lstm_layer": 1,
        "dropout": 0.05,
        "num_workers": 5,
        "train_epochs": 10,
        "batch_size": 32,
        "patience": 3,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "gpu": 0,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "is_debug": True
    }

# import linear_predict
# rst = linear_predict.forecast(settings)
# print("linear_predict rst ====> ", rst)
#
# import gru_single_model_predict
# rst = gru_single_model_predict.forecast(settings)
# print("gru_single_model_predict rst ====> ", rst)
#
#
# import lightgbm_predict
# rst = lightgbm_predict.forecast(settings)
# print("lightgbm_predict forecast rst ====> ", rst)
#
# import gru_multi_output_predict
# rst = gru_multi_output_predict.forecast(settings)
# print("gru_multi_output_predict forecast rst ====> ", rst)


# import predict
# rst = predict.multi_days_forecast(settings)
# print("multi_days_forecast rst ====> ", rst.shape)

# import predict
# rst = predict.multi_output_forecast(settings)
# print("multi_output_forecast rst ====> ", rst.shape)

import predict
rst = predict.forecast(settings)
print("forecast_all rst ====> ", rst.shape)
