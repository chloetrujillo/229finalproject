# Full 10-Class Tuning Findings (My Writeup)

## Goal
I wanted to run Jaduk's `data_v2` model on the full 10-class star prediction task (stars `1-10`).

## What I changed from Jaduk's default pass
- Kept the same model family and feature pipeline (`data_v2`, Adam, two-stage L1 feature selection with `n_select=25`).
- Tuned hyperparameters instead of using only defaults.
- Final config I selected:
  - `loss=l1` (instead of `mse`)
  - `batch_size=4` (instead of `8`)
  - `lr=1e-4` (instead of `3e-5`)
  - `hidden_size=128`, `l1_lambda=1e-3`, `weight_decay=0`, seed `42`

## Final confirmed result (full 10 classes)
From my confirm run:
- `val_acc = 0.3264`
- `val_off_by_one = 0.7431`
- `test_acc = 0.3048`
- `test_off_by_one = 0.7136`

Saved outputs:
- `experiments/full10_data_v2_final_confirm.csv`
- `experiments/full10_data_v2_final_confirm_json/`

## Why I chose these settings (short)
- `L1` loss is less sensitive to outlier misses than `MSE`, which helped stability on this ordinal target.
- Smaller batch (`4`) improved generalization in my sweeps.
- Higher learning rate (`1e-4`) converged better within the epoch budget.

## How to replicate

### 1) Install dependencies
```bash
python -m pip install --user torch matplotlib seaborn
```

### 2) Run quick tuning sweeps (what I used to pick the final config)
```bash
python src/model/pyperparam_tuning.py \
  --data_dir data_v2 \
  --results_csv experiments/full10_data_v2_quick_results.csv \
  --results_json_dir experiments/full10_data_v2_quick_json \
  --hidden_size 128 --batch_size 8 --lr 3e-5 \
  --weight_decay 0 --epochs_stage1 40 --epochs_stage2 40 \
  --l1_lambda 1e-3 --n_select 25 \
  --loss mse --huber_beta 1.0 --seed 42 \
  --tag quick_lr --sweep_param lr \
  --sweep_values 1e-5,3e-5,1e-4,3e-4,1e-3

python src/model/pyperparam_tuning.py \
  --data_dir data_v2 \
  --results_csv experiments/full10_data_v2_quick_results.csv \
  --results_json_dir experiments/full10_data_v2_quick_json \
  --hidden_size 128 --batch_size 8 --lr 1e-4 \
  --weight_decay 0 --epochs_stage1 40 --epochs_stage2 40 \
  --l1_lambda 1e-3 --n_select 25 \
  --loss mse --huber_beta 1.0 --seed 42 \
  --tag quick_batch --sweep_param batch_size \
  --sweep_values 4,8,16,32,64

python src/model/pyperparam_tuning.py \
  --data_dir data_v2 \
  --results_csv experiments/full10_data_v2_quick_results.csv \
  --results_json_dir experiments/full10_data_v2_quick_json \
  --hidden_size 128 --batch_size 4 --lr 1e-4 \
  --weight_decay 0 --epochs_stage1 40 --epochs_stage2 40 \
  --l1_lambda 1e-3 --n_select 25 \
  --loss mse --huber_beta 1.0 --seed 42 \
  --tag quick_loss --sweep_param loss \
  --sweep_values mse,l1,huber

python src/model/pyperparam_tuning.py \
  --data_dir data_v2 \
  --results_csv experiments/full10_data_v2_quick_results.csv \
  --results_json_dir experiments/full10_data_v2_quick_json \
  --hidden_size 128 --batch_size 4 --lr 1e-4 \
  --weight_decay 0 --epochs_stage1 40 --epochs_stage2 40 \
  --l1_lambda 1e-3 --n_select 25 \
  --loss huber --huber_beta 1.0 --seed 42 \
  --tag quick_huber_beta --sweep_param huber_beta \
  --sweep_values 0.5,1.0,2.0,3.0,5.0

python src/model/pyperparam_results_summary.py \
  --results_csv experiments/full10_data_v2_quick_results.csv \
  --output_csv experiments/full10_data_v2_quick_top5.csv \
  --top_k 5
```

### 3) Run my final confirm config (100/100 epochs)
```bash
python src/model/pyperparam_tuning.py \
  --data_dir data_v2 \
  --results_csv experiments/full10_data_v2_final_confirm.csv \
  --results_json_dir experiments/full10_data_v2_final_confirm_json \
  --hidden_size 128 --batch_size 4 --lr 1e-4 \
  --weight_decay 0 --epochs_stage1 100 --epochs_stage2 100 \
  --l1_lambda 1e-3 --n_select 25 \
  --loss l1 --huber_beta 1.0 --seed 42 \
  --tag final_confirm_l1
```

## Notes
- This is full 10-class prediction using `data_v2` and `stars` (not the 3-class `label` subset setup).
- Off-by-one is still useful, but I prioritize exact `test_acc` when comparing configs.
