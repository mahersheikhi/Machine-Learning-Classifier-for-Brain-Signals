"""Reproducible educational EEG experiment. All data are synthetic, in volts."""
from pathlib import Path
import json
import platform
import importlib.metadata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.signal import welch
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent / "results"


def simulate(seed=42):
    rng = np.random.default_rng(seed)
    fs = 250
    t = np.arange(180 * fs) / fs
    data = np.array([12e-6 * np.sin(2 * np.pi * 10 * t + phase)
                     + rng.normal(0, 12e-6, t.size) for phase in [0, .4, .8, 1.2]])
    samples = (np.arange(1, 179, 1.5) * fs).astype(int)
    labels = np.ones(samples.size, dtype=int)
    labels[rng.choice(samples.size, samples.size // 5, replace=False)] = 2
    # Explicitly injected target response; this makes an educational positive control.
    for onset, label in zip(samples, labels):
        if label == 2:
            latency = rng.normal(.30, .025)
            response = 14e-6 * np.exp(-.5 * ((t - onset / fs - latency) / .065) ** 2)
            data += np.array([.5, .8, 1., .7])[:, None] * response
    for onset in [5, 10, 15]:
        data[0] += 100e-6 * np.exp(-.5 * ((t - onset) / .06) ** 2)
    info = mne.create_info(["sim_Fz", "sim_Cz", "sim_Pz", "sim_Oz"], fs, "eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    events = np.column_stack([samples, np.zeros(samples.size, dtype=int), labels])
    return raw, events


def bandpower(x, fs):
    f, p = welch(x, fs=fs, nperseg=250, axis=-1)
    # Integrate in V^2, then convert to microvolt^2 to avoid tiny tree split scales.
    return 1e12 * np.stack([np.trapezoid(p[..., (f >= lo) & (f < hi)],
                                  f[(f >= lo) & (f < hi)], axis=-1)
                     for lo, hi in [(4, 8), (8, 13), (13, 30)]], axis=-1).reshape(len(x), -1)


def main():
    OUT.mkdir(exist_ok=True)
    mne.set_log_level("ERROR")
    raw, events = simulate()
    filtered = raw.copy().filter(1, 30, verbose=False)
    epochs = mne.Epochs(filtered, events, {"standard": 1, "target": 2},
                        tmin=-.2, tmax=.8, baseline=(-.2, 0), preload=True, verbose=False)
    x = epochs.get_data(copy=True)
    y = epochs.events[:, -1]
    train, test = train_test_split(np.arange(len(y)), test_size=.25,
                                   stratify=y, random_state=42)
    temporal = x.reshape(len(x), -1)
    bp = bandpower(x, raw.info["sfreq"])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    models = {
        "majority_baseline": (DummyClassifier(strategy="most_frequent"), temporal),
        "temporal_pca_logistic": (make_pipeline(StandardScaler(),
            PCA(n_components=20, svd_solver="full"),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)), temporal),
        "temporal_linear_svm": (make_pipeline(StandardScaler(),
            SVC(kernel="linear", class_weight="balanced", random_state=42)), temporal),
        "bandpower_random_forest": (RandomForestClassifier(n_estimators=150,
            class_weight="balanced", random_state=42, n_jobs=1), bp),
    }
    metrics = {}
    for name, (model, features) in models.items():
        # CV uses training trials only; scaling and PCA are fit inside each fold.
        scores = cross_val_score(model, features[train], y[train], cv=cv,
                                 scoring="balanced_accuracy")
        model.fit(features[train], y[train])
        pred = model.predict(features[test])
        score = (model.decision_function(features[test]) if hasattr(model, "decision_function")
                 else model.predict_proba(features[test])[:, list(model.classes_).index(2)])
        fpr, tpr, _ = roc_curve(y[test], score, pos_label=2)
        metrics[name] = {
            "accuracy": float(accuracy_score(y[test], pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
            "target_f1": float(f1_score(y[test], pred, pos_label=2, zero_division=0)),
            "roc_auc": float(roc_auc_score(y[test] == 2, score)),
            "roc_curve": {"false_positive_rate": fpr.tolist(),
                          "true_positive_rate": tpr.tolist()},
            "cv_balanced_accuracy_mean": float(scores.mean()),
            "cv_balanced_accuracy_std": float(scores.std()),
            "confusion_matrix": confusion_matrix(y[test], pred, labels=[1, 2]).tolist(),
        }
    null_y = np.random.default_rng(2026).permutation(y[train])
    null_scores = cross_val_score(models["temporal_pca_logistic"][0], temporal[train],
                                  null_y, cv=cv, scoring="balanced_accuracy")
    report = {"seed": 42, "data": "synthetic; injected target ERP", "sampling_hz": 250,
              "trials": len(y), "train_trials": len(train), "test_trials": len(test),
              "class_counts": {str(k): int((y == k).sum()) for k in [1, 2]},
              "models": metrics, "shuffled_label_cv_balanced_accuracy": null_scores.tolist(),
              "python": platform.python_version(),
              "versions": {p: importlib.metadata.version(p) for p in
                           ["numpy", "scipy", "mne", "scikit-learn", "matplotlib"]}}
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    t = raw.times
    axes[0, 0].plot(t[:1250], raw.get_data()[0, :1250] * 1e6, alpha=.4, label="Raw")
    axes[0, 0].plot(t[:1250], filtered.get_data()[0, :1250] * 1e6, label="1–30 Hz")
    axes[0, 0].set(xlabel="Time (s)", ylabel="Amplitude (µV)", title="Synthetic EEG preprocessing")
    axes[0, 0].legend()
    for signal, label in [(raw, "Raw"), (filtered, "Filtered")]:
        f, p = welch(signal.get_data()[0], fs=250, nperseg=500)
        axes[0, 1].semilogy(f, p * 1e12, label=label)
    axes[0, 1].set(xlim=(0, 60), xlabel="Frequency (Hz)", ylabel="PSD (µV²/Hz)", title="Welch spectrum")
    axes[0, 1].legend()
    for label in ["standard", "target"]:
        axes[1, 0].plot(epochs.times, epochs[label].average().data[2] * 1e6, label=label)
    axes[1, 0].set(xlabel="Time from event (s)", ylabel="Amplitude (µV)", title="Injected ERP: simulated Pz")
    axes[1, 0].legend()
    cm = np.array(metrics["temporal_pca_logistic"]["confusion_matrix"])
    axes[1, 1].imshow(cm, cmap="Blues")
    for (row, col), value in np.ndenumerate(cm):
        axes[1, 1].text(col, row, str(value), ha="center", va="center")
    axes[1, 1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Standard", "Target"],
                   yticklabels=["Standard", "Target"], xlabel="Predicted", ylabel="True",
                   title="Held-out PCA + logistic regression")
    fig.suptitle("BCI portfolio — synthetic educational experiment", fontsize=16)
    fig.savefig(OUT / "overview.png", dpi=160)
    plt.close(fig)
    model_labels = {
        "majority_baseline": "Always predict standard",
        "temporal_pca_logistic": "PCA + logistic regression",
        "temporal_linear_svm": "Linear SVM",
        "bandpower_random_forest": "Bandpower + random forest",
    }
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    # Distinct line styles keep overlapping curves visible.
    for (name, values), style in zip(metrics.items(), [":", "-", "--", "-."]):
        curve = values["roc_curve"]
        ax.plot(curve["false_positive_rate"], curve["true_positive_rate"],
                linestyle=style, linewidth=2,
                label=f"{model_labels[name]} (AUC = {values['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=.5,
            label="Chance-level ranking (AUC = 0.500)", zorder=0)
    ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02),
           xlabel="False alarm rate (standard trials flagged as targets)",
           ylabel="Target detection rate (targets correctly flagged)",
           title=f"ROC curves: {len(test)} held-out synthetic EEG trials")
    ax.set_aspect("equal")
    ax.grid(alpha=.2)
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(OUT / "roc_curve.png", dpi=160)
    plt.close(fig)
    rows = ["| Model | Accuracy | Balanced accuracy | Target F1 | ROC AUC | Train CV BA (mean ± SD) |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, v in metrics.items():
        rows.append(f"| {name} | {v['accuracy']:.3f} | {v['balanced_accuracy']:.3f} | "
                    f"{v['target_f1']:.3f} | {v['roc_auc']:.3f} | "
                    f"{v['cv_balanced_accuracy_mean']:.3f} ± {v['cv_balanced_accuracy_std']:.3f} |")
    (OUT / "RESULTS.md").write_text(
        "# Reproducible experiment results\n\nAll data are synthetic. Target ERPs are explicitly injected.\n\n"
        + f"Seed 42; {len(y)} trials ({len(train)} train / {len(test)} test); class counts: {report['class_counts']}.\n\n"
        + "\n".join(rows) + f"\n\nShuffled-label training CV balanced accuracy: {null_scores.mean():.3f} "
        + f"± {null_scores.std():.3f}. This is a single negative control, not a permutation significance test.\n\n"
        + "Models and hyperparameters were fixed before evaluation. CV uses only the training partition. "
        + "The holdout compares these fixed baselines; it should not be reused for tuning. "
        + "These results assess this simulator only, not performance on human EEG.\n\n"
        + "## Results in plain language\n\n"
        + f"The PCA + logistic regression model correctly classified all {len(test)} test trials "
        + f"({cm[1, 1]} targets and {cm[0, 0]} standard events); the linear SVM also achieved "
        + f"{metrics['temporal_linear_svm']['accuracy']:.0%} accuracy. "
        + f"Always guessing 'standard' achieved {metrics['majority_baseline']['accuracy']:.0%} accuracy "
        + "because most events were standard, but it missed every target. "
        + f"The bandpower model achieved {metrics['bandpower_random_forest']['accuracy']:.0%} accuracy "
        + "and missed some targets, suggesting that the signal's shape over time was more useful "
        + "here than its frequency summary. These strong scores come from a small, deliberately "
        + "simplified simulation and do not show how well the models would work on real people.\n\n"
        + "## Reading the ROC curve\n\n"
        + "![ROC curves for all four models on held-out trials](roc_curve.png)\n\n"
        + "Each curve shows the trade-off as we change how easily a model flags an event as a target: "
        + "moving upward means finding more targets, while moving right means raising more false alarms. "
        + "Curves closer to the top-left corner are better; AUC summarizes this ranking ability "
        + "(1.0 means perfect separation on these test trials, while 0.5 means chance-level ranking). "
        + f"The two time-based models both reached AUC {metrics['temporal_pca_logistic']['roc_auc']:.3f} "
        + "and overlap on this chart; the bandpower model reached "
        + f"{metrics['bandpower_random_forest']['roc_auc']:.3f}, and the always-standard baseline "
        + "lies on the chance diagonal.\n\n"
        + "![Signal processing and evaluation](overview.png)\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
