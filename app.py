import io
import os
import base64
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash

from clustering import (
    preprocess, compute_elbow, compute_silhouette, suggest_best_k,
    fit_kmeans, profile_clusters, FEATURES
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.secret_key = "customer-segmentation-demo-secret"

sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": "#FBFAF7",
    "figure.facecolor": "#FBFAF7",
    "grid.color": "#E4E0D8",
})

PALETTE = ["#C9A227", "#4C7A9A", "#E07A5F", "#3D9970", "#6B5B95", "#D62839", "#457B9D", "#8D6E63"]


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_elbow_plot(ks, inertias, chosen_k):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ks, inertias, marker="o", color="#C9A227", linewidth=2, markersize=6)
    if chosen_k in ks:
        idx = ks.index(chosen_k)
        ax.plot(chosen_k, inertias[idx], marker="o", color="#D62839", markersize=12, zorder=5)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (WCSS)")
    ax.set_title("Elbow Method")
    sns.despine(fig)
    return fig_to_base64(fig)


def make_silhouette_plot(ks, scores, chosen_k):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(ks, scores, marker="o", color="#4C7A9A", linewidth=2, markersize=6)
    if chosen_k in ks:
        idx = ks.index(chosen_k)
        ax.plot(chosen_k, scores[idx], marker="o", color="#D62839", markersize=12, zorder=5)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score by k")
    sns.despine(fig)
    return fig_to_base64(fig)


def make_scatter_plot(df, x, y, personas):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    color_map = {p["cluster_id"]: p["color"] for p in personas}
    name_map = {p["cluster_id"]: p["persona_name"] for p in personas}
    for cid, sub in df.groupby("Cluster"):
        ax.scatter(sub[x], sub[y], s=38, alpha=0.75,
                   color=color_map.get(cid, "#999"), edgecolor="white", linewidth=0.4,
                   label=f"{name_map.get(cid, cid)}")
    ax.set_xlabel(x.replace("_", " "))
    ax.set_ylabel(y.replace("_", " "))
    ax.set_title(f"{y.replace('_',' ')} vs {x.replace('_',' ')}")
    ax.legend(fontsize=8, frameon=True, loc="best")
    sns.despine(fig)
    return fig_to_base64(fig)


def make_cluster_size_plot(personas):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    names = [p["persona_name"] for p in personas]
    sizes = [p["size"] for p in personas]
    colors = [p["color"] for p in personas]
    bars = ax.barh(names, sizes, color=colors)
    ax.set_xlabel("Number of Customers")
    ax.set_title("Segment Sizes")
    ax.invert_yaxis()
    for bar, size in zip(bars, sizes):
        ax.text(bar.get_width() + max(sizes) * 0.01, bar.get_y() + bar.get_height() / 2,
                str(size), va="center", fontsize=9)
    sns.despine(fig)
    return fig_to_base64(fig)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    use_sample = request.form.get("use_sample")

    if use_sample or not file or file.filename == "":
        df = pd.read_csv(os.path.join(BASE_DIR, "data", "sample_customers.csv"))
    else:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f"Could not read CSV: {e}")
            return redirect(url_for("index"))

    # keep only usable numeric features that exist in this dataset
    available_features = [f for f in FEATURES if f in df.columns]
    if len(available_features) < 2:
        flash("Dataset needs at least 2 of: Age, Annual_Income_k, Spending_Score, "
              "Membership_Years, Visits_Per_Month")
        return redirect(url_for("index"))

    k_choice = request.form.get("k_choice", "auto")

    X_scaled, features_used, scaler = preprocess(df, available_features)

    k_max = min(10, max(3, len(df) // 10))
    ks, inertias = compute_elbow(X_scaled, k_min=2, k_max=k_max)
    _, sil_scores = compute_silhouette(X_scaled, k_min=2, k_max=k_max)
    best_k = suggest_best_k(ks, sil_scores)

    if k_choice != "auto":
        try:
            k = int(k_choice)
            k = max(2, min(k_max, k))
        except ValueError:
            k = best_k
    else:
        k = best_k

    model, labels, sil = fit_kmeans(X_scaled, k)
    labeled_df, personas = profile_clusters(df, labels, features_used)

    elbow_plot = make_elbow_plot(ks, inertias, k)
    sil_plot = make_silhouette_plot(ks, sil_scores, k)
    size_plot = make_cluster_size_plot(personas)

    x_feat = "Annual_Income_k" if "Annual_Income_k" in features_used else features_used[0]
    y_feat = "Spending_Score" if "Spending_Score" in features_used else features_used[-1]
    scatter_plot = make_scatter_plot(labeled_df, x_feat, y_feat, personas)

    run_id = uuid.uuid4().hex[:8]
    csv_path = os.path.join(UPLOAD_DIR, f"segmented_{run_id}.csv")
    labeled_df.to_csv(csv_path, index=False)

    return render_template(
        "results.html",
        k=k, best_k=best_k, silhouette=round(sil, 3),
        n_customers=len(df), features_used=features_used,
        elbow_plot=elbow_plot, sil_plot=sil_plot,
        scatter_plot=scatter_plot, size_plot=size_plot,
        personas=personas, run_id=run_id,
        x_feat=x_feat, y_feat=y_feat,
    )


@app.route("/download/<run_id>")
def download(run_id):
    from flask import send_file
    path = os.path.join(UPLOAD_DIR, f"segmented_{run_id}.csv")
    if not os.path.exists(path):
        flash("File not found or expired.")
        return redirect(url_for("index"))
    return send_file(path, as_attachment=True, download_name=f"segmented_customers_{run_id}.csv")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
