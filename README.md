#  HDP-Flow: Generalizable Bayesian Nonparametric Model for Time Series State Discovery

HDP-Flow is a Bayesian nonparametric model that combines normalizing flows with scalable variational inference to uncover evolving latent states in non-stationary time series. The code in this repo enables efficient, unsupervised state discovery and demonstrates strong performance and transferability across diverse real-world datasets. For more details, please refer to our paper []().

<p align="center">
  <img src="HDPFlow.png" alt="HDP-Flow Overview" width="600"/>
</p>

## Prerequisites

All codes are written for Python 3 (https://www.python.org/) on Linux platform. 

1. Create and activate a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Training the Model:

The dataset-specific model parameters are defined in the [config file](experiment_config.ini), where you can add your datasets and specify the desired training paradigm. You can train the model using a command like the following:

```
python main.py --data sim_easy --train
```

You can use the `--cont` flag to continue training the model from the last saved checkpoint.
Evaluation is automatically performed immediately after training completes.

### Clone this repository
```
git clone https://github.com/sanatonek/HDP-Flow.git
```

## Citation

