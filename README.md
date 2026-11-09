# Guiding Data Allocation for Robust Subpopulation Generalization

Zhaoying Pan, Yipei Wang, Shenyu Lu, Xiaoqian Wang, NeurIPS 2026.

<!-- #### [[arXiv]]() | [[Poster]]() -->

## Table of contents

- [Introduction](#introduction)
- [Repo overview](#repo-overview)
- [Installation](#installation)
- [Theory](#theory)
  - [Trajectory estimation](#trajectory-estimation)
  - [Feature separability estimation](#feature-separability-estimation)
  - [Analysis and visualization](#analysis-and-visualization)
- [Experiments](#experiments)
  - [Data](#data)
  - [Configuration files](#configuration-files)
  - [Acc, Grad, and Bal experiments](#acc-grad-and-bal-experiments)
  - [Allocation along the trajectory](#allocation-along-the-trajectory)
  - [3-simplex visualization](#3-simplex-visualization)
- [Outputs](#outputs)
<!-- - [Citation](#citation) -->

## Introduction

Machine learning models often achieve high average accuracy in the training data while performing poorly on test data due to subpopulation shift. A common data-centric solution is to enforce balanced subgroup proportions, but full balance can be costly, infeasible, and not always necessary. In this work, we study whether balanced training data is the unique optimal configuration for robust subpopulation generalization. We extend a Gaussian feature-model framework to theoretically characterize the performance landscape across general training subgroup distributions. We show that multiple configurations, including substantially imbalanced ones, can perform comparably to the balanced one. We further derive gradient-based directions for adjusting subgroup proportions to improve adjusted accuracy and worst-group accuracy more efficiently. These findings suggest that strategically allocating data offers a more flexible and principled path to robust performance than simple balancing.


## Repo overview

The repo contains the code to reproduce the theoretical estimation and empirical experiments on four datasets: Waterbirds-syn, UrbanCars-syn, CelebA-subset, and CivilComments-subset.

```text
configs/       Base, experiment, and trajectory-allocation configurations
datasets/      Dataset loaders and synthetic allocation datasets
methods/       Models, optimizers, and training methods
theory/        Theoretical estimators and an example notebook
scripts/       Visualization script
main.py        Training and evaluation entry point
```


## Installation

We applied Python 3.7.16 for our experiments. Create an environment and install the dependencies:

```bash
conda create -n subpop-allocation python=3.7.16 -y
conda activate subpop-allocation
pip install -r requirements.txt
```

## Theory

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/16YTrJrZmwfV0ouYPOGck-83qyNjR1KB6/view?usp=sharing)

### Trajectory estimation

Run the analytic allocation estimates used in the paper with:

```bash
python theory/estimate.py --m1 2 --m2 2 --params 0.976 0.184 0.128 --num-steps 10
```

### Feature separability estimation

Estimate feature separability using either two adjusted accuracies with their allocations or one allocation with four group accuracies. The notebook `theory/estimate_example.ipynb` or the Colab notebook demonstrates the estimation with two and plots a trajectory. Run the notebook from the repository root, or use:

```python
from theory.estimate import Estimate
est = Estimate()
m1, m2, fit = est.solve_ms(acc0=0.78, acc1=0.95, params0=[0.12, 0.22, 0.96], params1=[0.5, 0.5, 0.5])
```

### Analysis and visualization

Analyze the paper's 7^3 or 49^3 theoretical configurations and optionally save the estimated performance values:

```bash
python theory/analyze_configurations.py --m1 2 --m2 2 --grid-size 7 --output results/configurations.npz
```

The `theory/estimate_example.ipynb` notebook loads the saved reference NPZ file of grid-size of 49 and visualize the theoretical high-performance, direction-similarity, accuracy-gap, and Grad/Bal direction plots.

The same NPZ can be passed directly to the shared visualization script. For example:

```bash
python -m scripts.visualize_simplex theory/reference/grid49_m1_2_m2_2.npz \
  --plot high-performance --output results/theory_high_performance.png
python -m scripts.visualize_simplex theory/reference/grid49_m1_2_m2_2.npz \
  --plot direction-similarity --output results/theory_direction_similarity.png
python -m scripts.visualize_simplex theory/reference/grid49_m1_2_m2_2.npz \
  --plot accuracy-gap --output results/theory_accuracy_gap.png
```

## Experiments

### Data

The code expects the following directories below `data.data_root`:

| Dataset | Data directory |
|---|---|
| Waterbirds-syn | `./data/waterbirds_zeta/` |
| UrbanCars-syn | `data/Whac-A-Mole/` and `data/urbancars/` |
| CelebA-subset | `data/celeba/` |
| CivilComments-subset | `data/civilcomments/` |

If the data are stored elsewhere, replace the link target or set `data.data_root` in `configs/base.yaml`.

### Configuration files

`configs/base.yaml` contains settings shared by all runs. The remaining release configurations are organized as follows:

| Dataset | Acc/Grad/Bal experiment | Trajectory allocation |
|---|---|---|
| Waterbirds-syn | `configs/waterbirds.yaml` | `configs/waterbirds_allocation.yaml` |
| UrbanCars-syn | `configs/urbancars.yaml` | `configs/urbancars_allocation.yaml` |
| CelebA | `configs/celeba.yaml` | `configs/celeba_allocation.yaml` |
| CivilComments | `configs/civilcomments.yaml` | `configs/civilcomments_allocation.yaml` |

The four experiment files each contain one balanced example configuration. Each `*_allocation.yaml` file contains the 21 configurations used for the allocation experiment: the initial configuration and the Grad, Bal, Sign, and Random allocations at 5%, 10%, 15%, 20%, and 25% additional data.

### Acc, Grad, and Bal experiments

The same dataset YAML is used for the Acc sweep and for evaluation after moving configurations in the Grad or Bal direction. Replace `data.params_dict` with the configurations for that experiment and use task names such as `acc_0`, `grad_0`, or `bal_0`. Waterbirds and UrbanCars store each configuration directly as `[alpha1, alpha0, beta]`. CelebA and CivilComments store an index in `params_dict`. The corresponding configuration and subset CSV must be supplied at the same position in `params_list` and `metadata_list`. Generate the 343 subset CSVs for both datasets with:

```bash
python scripts/generate_subsets.py
```

The script writes to `data/celeba/subsets_n8`, `data/civilcomments/subsets_n8`, and companion pickle files under `datasets/`. The files can alternatively be accessed at [Google Drive](https://drive.google.com/drive/folders/1PN7yn-r7PDLwBzocdtJGlttQctGj7g58?usp=sharing).

For example, run the configuration currently specified in each release YAML with:

```bash
python main.py --config_path configs/waterbirds.yaml --model resnet50 --exp_name wb_acc
python main.py --config_path configs/urbancars.yaml --model resnet50 --exp_name uc_acc
python main.py --config_path configs/celeba.yaml --model resnet50 --exp_name ca_acc
python main.py --config_path configs/civilcomments.yaml --model bert --exp_name cc_acc
```

After substituting Grad or Bal configurations, change `--exp_name` accordingly, for example to `ca_grad` or `ca_bal`.

### Allocation along the trajectory

Run all trajectory-allocation experiments with:

```bash
python main.py --config_path configs/waterbirds_allocation.yaml --model resnet50 --exp_name waterbirds_allocation
python main.py --config_path configs/celeba_allocation.yaml --model resnet50 --exp_name celeba_allocation
python main.py --config_path configs/civilcomments_allocation.yaml --model bert --exp_name civilcomments_allocation
python main.py --config_path configs/urbancars_allocation.yaml --model resnet50 --exp_name urbancars_allocation
```

Each command trains and evaluates every entry in `data.params_dict`, using the corresponding entry in `train.num_steps`. CelebA and CivilComments use four-integer subgroup counts. UrbanCars uses `[alpha1, alpha0, beta, sample_count]`. Waterbirds additionally records the fixed initial configuration and its sample count in each entry.

### 3-simplex visualization

For a full empirical simplex, replace the single example in the corresponding experiment YAML with the 343 configurations from the reference grid (`0.125` to `0.875` in each coordinate).

Visualize configurations at or above the 90th-percentile adjusted accuracy from the included CelebA result with:

```bash
python -m scripts.visualize_simplex results/ca_acc.pkl \
  --plot high-performance \
  --output results/empirical_high_performance.png
```

To visualize the empirical adjusted-accuracy gap, use the Grad result as the main input and the Bal result as its comparison:

```bash
python -m scripts.visualize_simplex results/ca_grad.pkl \
  --plot accuracy-gap \
  --comparison results/ca_bal.pkl \
  --output results/empirical_accuracy_gap.png
```


## Outputs

Runs are written below:

```text
results/<dataset>/<model>/<method>/<timestamp>_<experiment-name>/
  config.yaml
  result.pkl
  exp.log
  ckpt/
```

Every run writes `result.pkl`. It is a dictionary keyed by the names in `data.params_dict`; each value stores the merged run configuration and per-seed `va_loss`, `va_acc_grps`, `te_loss`, and `te_acc_grps` arrays. By default, `log.disable_saving: true` disables model checkpoints only. Set it to `false` to retain checkpoints. Generated results and pickle files are ignored by Git.

<!-- ## Citation

```bibtex

``` -->
