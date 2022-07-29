<h1>KDD Cup 2022 - Baidu Spatial Dynamic Wind Power Forecasting</h1>

This is the solution from team trymore, winning 2th place in 2490 teams.

<h2>0. Overview</h2>
Based on the weighted hybrid prediction method, our method designs three main modules, 
each of which plays a different role in the final prediction.
In order to obtain better integration results, we used different feature sets for different models to build differences. 
Next, we will introduce our data processing and the details of the three component models.
<img width="1071" alt="model" src="https://user-images.githubusercontent.com/43618727/181726007-ae7f9ddd-f5e2-4c43-af70-327b0e1b4bdd.png">

<h2>1. Dataset</h2>
You can download the data 'wtbdata_245days.csv' from Baidu AI studio, https://aistudio.baidu.com/aistudio/competition/detail/152/0/datasets.

<h2>2. Train</h2>
Details can be find in ./train_code/Readme.md, fellow the steps you will get all the models.

<h2>3. Inference</h2>
The inference code is located in ./test_code.
