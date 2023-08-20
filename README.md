# Structural Hawkes Processes for Learning Causal Structure from Discrete-Time Event Sequences
The python implementation of paper [Structural Hawkes Processes for Learning Causal Structure from Discrete-Time Event Sequences](https://arxiv.org/abs/2305.05986). (IJCAI 2023)

### Usage
The running example of SHP is given below.

```python
from SHP import SHP_exp
from utils import get_performance

likelihood, fited_alpha, fited_mu, real_edge_mat, real_alpha, real_mu = SHP_exp(n=20, sample_size=20000,
                                                                                out_degree_rate=1.5,
                                                                                mu_range_str="0.00005,0.0001",
                                                                                alpha_range_str="0.5,0.7",
                                                                                decay=5, model_decay=0.35, seed=0,
                                                                                time_interval=5, penalty='BIC',
                                                                                hill_climb=True, reg=0.85)
res = get_performance(fited_alpha, real_edge_mat)
```

# Create environment

The environment config is given in `environment.yml` and your can create the environment using:

```shell
conda env create -f environment.yml
conda activate shp
```







