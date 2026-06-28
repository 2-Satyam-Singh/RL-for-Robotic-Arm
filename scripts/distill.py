"""
scripts/distill.py — Train all 4 distillation models from a PPO test log.

Mirrors the feature engineering in Gazebo_sim.ipynb (Cells 2A/2C/4/5/6)
but runs fully locally without Colab/Drive.

Usage:
    python scripts/distill.py --log results/log_test_panda_PPO_sparse_s42_<ts>.csv --robot panda

Output:
    models/LR/7dof.pkl
    models/RF/7dof.pkl
    models/SVM/7dof.pkl
    models/XGBoost/7dof.pkl
    models/SCALER/7dof.pkl
"""

import argparse
import os
import joblib
import numpy as np

from config import ROBOT_CONFIGS


def load_and_engineer(log_path, dof):
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Install pandas: pip install pandas")

    df = pd.read_csv(log_path)
    print(f"Loaded {log_path}. Original rows: {len(df)}")

    # Keep only successful episodes (same as notebook)
    successful_episodes = df[df["is_success"] == 1]["episode"].unique()
    df = df[df["episode"].isin(successful_episodes)].copy()
    print(f"After filtering successful episodes: {len(df)} rows, {len(successful_episodes)} episodes")

    if len(df) == 0:
        raise ValueError("No successful episodes found in the log. Cannot train distillation models.")

    df = df.sort_values(["episode", "step_count"])

    # Spatial features
    df["ee_error_x"] = df["ent_0_x"] - 0.5
    df["ee_error_y"] = df["ent_0_y"] - 0.0
    df["ee_error_z"] = df["ent_0_z"] - 0.5
    df["ee_dist"] = np.sqrt(df["ee_error_x"]**2 + df["ee_error_y"]**2 + df["ee_error_z"]**2)

    # Time delta — fall back to fixed dt if step_time was not logged
    if "step_time" in df.columns:
        df["dt"] = df.groupby("episode")["step_time"].diff().fillna(0)
        dt_safe = np.where(df["dt"] <= 0, 1e-5, df["dt"])
    else:
        dt_safe = np.full(len(df), 1e-5)

    # Progress
    episode_len = df.groupby("episode").size()
    df["episode_len"] = df["episode"].map(episode_len)
    df["progress"] = df["step_count"] / df["episode_len"]

    for i in range(dof):
        df[f"sin_joint_{i}"] = np.sin(df[f"joint_{i}"])
        df[f"cos_joint_{i}"] = np.cos(df[f"joint_{i}"])

        joint_diff = df[f"joint_{i}"] - df.groupby("episode")[f"joint_{i}"].shift(1).fillna(df[f"joint_{i}"])
        df[f"vel_{i}"] = joint_diff / dt_safe

        vel_diff = df[f"vel_{i}"] - df.groupby("episode")[f"vel_{i}"].shift(1).fillna(0)
        df[f"accel_{i}"] = vel_diff / dt_safe

        accel_diff = df[f"accel_{i}"] - df.groupby("episode")[f"accel_{i}"].shift(1).fillna(0)
        df[f"jerk_{i}"] = accel_diff / dt_safe

        df[f"vel_accel_{i}"] = df[f"vel_{i}"] * df[f"accel_{i}"]

        df[f"prev_action_{i}"] = df.groupby("episode")[f"action_{i}"].shift(1).fillna(0)
        df[f"prev_action_2_{i}"] = df.groupby("episode")[f"action_{i}"].shift(2).fillna(0)
        df[f"prev_vel_{i}"] = df.groupby("episode")[f"vel_{i}"].shift(1).fillna(0)

        df[f"delta_action_{i}"] = df[f"action_{i}"] - df[f"prev_action_{i}"]

    OUTPUT_COLS = [f"delta_action_{i}" for i in range(dof)]
    exclude_cols = OUTPUT_COLS + [f"action_{i}" for i in range(dof)] + [f"joint_{i}" for i in range(dof)] + [
        'episode', 'step_count', 'reward', 'done', 'is_success',
        'episode_len', 'ent_0_x', 'ent_0_y', 'ent_0_z', 'step_time', 'dt',
    ]
    exclude_cols = [c for c in exclude_cols if c in df.columns]
    INPUT_COLS = [col for col in df.columns if col not in exclude_cols]

    X = df[INPUT_COLS].copy()
    y = df[OUTPUT_COLS].copy()

    print(f"Input features: {len(INPUT_COLS)}, Outputs: {len(OUTPUT_COLS)}")
    return X, y


def train_and_save(X, y, dof, save_root="models"):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import r2_score

    try:
        from xgboost import XGBRegressor
        has_xgb = True
    except ImportError:
        print("⚠️  XGBoost not found, skipping. Install with: pip install xgboost")
        has_xgb = False

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    fname = f"{dof}dof.pkl"

    models = {
        "LR": LinearRegression(),
        "RF": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        "SVM": MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale')),
    }
    if has_xgb:
        models["XGBoost"] = MultiOutputRegressor(
            XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1)
        )

    print(f"\nTraining {len(models)} models on {len(X_train)} samples...\n")
    for name, model in models.items():
        print(f"  Training {name}...", end=" ", flush=True)
        model.fit(X_train_s, y_train)
        r2 = r2_score(y_test, model.predict(X_test_s))
        print(f"R² = {r2:.4f}")

        out_dir = os.path.join(save_root, name)
        os.makedirs(out_dir, exist_ok=True)
        joblib.dump(model, os.path.join(out_dir, fname))
        print(f"    Saved → {os.path.join(out_dir, fname)}")

    scaler_dir = os.path.join(save_root, "SCALER")
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(scaler_dir, fname))
    print(f"\n  Scaler saved → {os.path.join(scaler_dir, fname)}")
    print("\n✅ All distillation models saved.")


def main():
    ap = argparse.ArgumentParser(description="Train distillation models from a PPO test log.")
    ap.add_argument("--log", required=True, help="Path to results/log_test_panda_PPO_*.csv")
    ap.add_argument("--robot", default="panda", choices=list(ROBOT_CONFIGS.keys()))
    ap.add_argument("--model_root", default="models")
    args = ap.parse_args()

    cfg = ROBOT_CONFIGS[args.robot]
    dof = len(cfg["joints"])
    print(f"Robot: {args.robot.upper()} ({dof} DOF)\n")

    X, y = load_and_engineer(args.log, dof)
    train_and_save(X, y, dof, save_root=args.model_root)


if __name__ == "__main__":
    main()
