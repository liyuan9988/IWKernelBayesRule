# IWKernelBayesRule

Code for "Importance Weighting Approach in Kernel Bayes' Rule" (https://arxiv.org/abs/2202.02474)

## How to Run codes?

1. Install all dependencies
   ```
   pip install -r requirements.txt
   ```
2. Create empty directories for logging
   ```
   mkdir logs
   mkdir dumps
   ```
3. (Optional) Download data for dSprite experiments. 
   ```
   mkdir data
   wget -P data/ https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
   ```
   We are currently preparing publishing the data for Maze experiment.
   
4. Run codes
   ```
   python main.py <path-to-configs> experiment
   ```
   The result can be found in `dumps` folder. You can run in parallel by specifing  `-t` option.
