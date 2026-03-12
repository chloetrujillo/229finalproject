import argparse
import os
import subprocess
import sys


def run_cmd(cmd):
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Run staged 5x5 hyperparameter plan.")
    parser.add_argument("--data_dir", type=str, default="data_v2")
    parser.add_argument("--results_csv", type=str, default="experiments/v2_id_results.csv")
    parser.add_argument("--results_json_dir", type=str, default="experiments/json_runs")
    parser.add_argument("--python_bin", type=str, default=sys.executable)
    args = parser.parse_args()

    tuner = os.path.join("src", "model", "pyperparam_tuning.py")
    common = [
        args.python_bin,
        tuner,
        "--data_dir",
        args.data_dir,
        "--results_csv",
        args.results_csv,
        "--results_json_dir",
        args.results_json_dir,
    ]

    # Baseline (current v2_id defaults)
    base = [
        "--hidden_size", "128",
        "--batch_size", "8",
        "--lr", "3e-5",
        "--weight_decay", "0",
        "--epochs_stage1", "100",
        "--epochs_stage2", "100",
        "--l1_lambda", "1e-3",
        "--n_select", "25",
        "--loss", "mse",
        "--huber_beta", "1.0",
        "--seed", "42",
    ]

    # 1) LR (5)
    run_cmd(common + base + ["--tag", "sweep_lr", "--sweep_param", "lr", "--sweep_values", "1e-5,3e-5,1e-4,3e-4,1e-3"])

    # 2) Batch size (5)
    run_cmd(common + base + ["--tag", "sweep_batch", "--sweep_param", "batch_size", "--sweep_values", "4,8,16,32,64"])

    # 3) Loss / robustness (5)
    run_cmd(common + base + ["--tag", "sweep_loss", "--sweep_param", "loss", "--sweep_values", "mse,l1,huber"])
    run_cmd(common + base + ["--tag", "sweep_huber_beta", "--loss", "huber", "--sweep_param", "huber_beta", "--sweep_values", "0.5,1.0,2.0,3.0,5.0"])

    # 4) Randomization (5 seeds)
    run_cmd(common + base + ["--tag", "sweep_seed", "--sweep_param", "seed", "--sweep_values", "0,7,21,42,1337"])

    # 5) Regularization (5)
    run_cmd(common + base + ["--tag", "sweep_weight_decay", "--sweep_param", "weight_decay", "--sweep_values", "0,1e-6,1e-5,1e-4,1e-3"])

    print("Completed staged tuning plan.")


if __name__ == "__main__":
    main()
