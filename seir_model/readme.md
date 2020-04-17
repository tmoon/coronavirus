## How to run the SIR model

The basic command for the file is 

```
./sir_model.py
```

Which would run the model on a bunch of default parameters. To modify the parameters, the following flags can be used:
* `--infile` : To pass in the directory of the input file, which should be in CSV format with two columns that contain the date and the number of people who tested positive in the country.
* `--outfile` : The file where the output data is going to be generated.
* `--params` : The parameters, as a list, in the following order: `beta, q, delta, gamma_mild, gamma_wild, N` (default: `(0.2, 0.05, 0.4, 0.18, 0.33, 0.25)`).
* `--n` : The smoothing factor for the read dataset (default: 3)
* `--offset`: Days since January 22 that we ignore on the dataset. (default: 30)
* `--last_offset`: Days until Last offset is the number of days to ignore from the end of the dataset (default: 1). So, a last offset of 1 means data ends on March 23 if the data ends on March 24.
* `--lockdown` : The day on which lockdown was implemented. (default: 37, *should be changed*).
* `--rand_walk_stds` : The standard deviation of the random walks that is used to learn each parameter in MCMC. Defaults to `(0.01, 0.005, 0.01, 0.005, 0.005, 0.005)`.

For different countries, we recommend changing the `last_offset, offset,` and `lockdown`. You could keep the `params` and `random_walk_stds` as is.
