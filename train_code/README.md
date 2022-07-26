### 00 - before
    put "wtbdata_245days.csv" file to "datasets/" folder

### 01-lightgbm-multi-output
    1. cd 01-lightgbm-multi-output
    2. run: python filter_data_by_group.py
        output: ../datasets/wtbdata_245days_filternan_by_group.csv
    3. run: python lightgbm_multi_output.py
        output: ./lightgbm_models/288 model files
                

### 02-gru-multi-output
```
1. cd 02-gru-multi-output
2. run:
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=6 --output_len=12 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_6_12/ --start_col=3
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=18 --output_len=36 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_18_36/ --start_col=3
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=144 --output_len=288 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_144_288/ --start_col=3
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=72 --output_len=144 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_72_144/ --start_col=3
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=36 --output_len=72 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_36_72/ --start_col=3
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=126 --output_len=252 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_126_252/ --start_col=3
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=108 --output_len=216 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_108_216/ --start_col=3
python train.py --in_var=10 --train_size=223 --gpu=0 --val_size=20 --test_size=2 --total_size=245 --is_debug=True --num_workers=1 --capacity=134 --input_len=90 --output_len=180 --day_len=144 --lstm_layer=1 --checkpoints=./gru-multi-models/1lstm_244checkpoints_90_180/ --start_col=3

output: output to matched model file in gru-multi-models folder
```


### 03-linear-models
    
    run: python linear_polynomial.py

    output: 
        a. ./linear_models/LinearRegression model file
        b. ./linear_models/Pipeline(PolynomialFeatures(degree=4)+LinearRegression) model file

### 04-gru-single-model

    run: python gru_single_model.py

    output: ./gru-single-models/model file