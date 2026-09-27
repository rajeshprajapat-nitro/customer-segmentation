"""
Core ML engine for the Customer Segmentation app.

Pipeline:
  1. preprocess()      -> scale numeric features
  2. compute_elbow()   -> inertia across k range (Elbow Method)
  3. compute_silhouette() -> silhouette score across k range
  4. fit_kmeans()      -> final KMeans model for chosen k
  5. profile_clusters()-> turns numeric centroids into human-readable
                          marketing personas via rule-based heuristics
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

FEATURES = ["Age", "Annual_Income_k", "Spending_Score", "Membership_Years", "Visits_Per_Month"]


def preprocess(df: pd.DataFrame, features=None):
    features = features or FEATURES
    features = [f for f in features if f in df.columns]
    X = df[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features, scaler


def compute_elbow(X_scaled, k_min=2, k_max=10, random_state=42):
    ks, inertias = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        ks.append(k)
        inertias.append(km.inertia_)
    return ks, inertias


def compute_silhouette(X_scaled, k_min=2, k_max=10, random_state=42):
    ks, scores = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        ks.append(k)
        scores.append(score)
    return ks, scores


def suggest_best_k(ks, silhouette_scores):
    best_idx = int(np.argmax(silhouette_scores))
    return ks[best_idx]


def fit_kmeans(X_scaled, k, random_state=42):
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    return km, labels, sil


# ---------------------------------------------------------------------------
# Persona engine: translate numeric centroids into marketing-ready personas
# ---------------------------------------------------------------------------

def _percentile_bucket(value, series):
    """Return 'low' / 'medium' / 'high' based on where value falls vs the
    overall population distribution (tertiles)."""
    q1, q2 = series.quantile([0.33, 0.66])
    if value <= q1:
        return "low"
    elif value <= q2:
        return "medium"
    else:
        return "high"


PERSONA_RULES = {
    ("high", "high"):  ("Premium Loyalists", "#C9A227",
        "High income and high spend — your most valuable segment.",
        ["VIP loyalty tiers & early access to new products",
         "Personalized concierge / relationship marketing",
         "Premium bundles & cross-sell of high-margin items",
         "Referral incentives — they influence peers"]),
    ("high", "medium"): ("Comfortable Spenders", "#4C7A9A",
        "Solid income with moderate, considered spending.",
        ["Highlight quality/value trade-off in messaging",
         "Targeted upsell campaigns around key purchase moments",
         "Limited-time offers to nudge spend upward",
         "Cross-category recommendations based on past purchases"]),
    ("high", "low"): ("Cautious Affluents", "#6B5B95",
        "High earning potential but currently low engagement — an untapped segment.",
        ["Re-engagement campaigns explaining value clearly",
         "Trust-building content (reviews, guarantees, case studies)",
         "Low-commitment entry offers (free trial, small bundle)",
         "Investigate friction points in the purchase journey"]),
    ("medium", "high"): ("Value-Driven Enthusiasts", "#E07A5F",
        "Moderate income but high spend — price-sensitive but engaged shoppers.",
        ["Loyalty points & rewards programs to retain engagement",
         "Flash sales / limited-time discounts drive strong response",
         "Instalment or 'buy now pay later' options",
         "Bundle deals to increase basket size without raising price"]),
    ("medium", "medium"): ("Steady Mainstream", "#3D9970",
        "The broad middle — reliable, moderate spenders with room to grow.",
        ["Regular newsletters & seasonal promotions",
         "Tiered loyalty programs to incentivize movement upward",
         "A/B test messaging to find what drives incremental spend",
         "Bundle popular items to lift average order value"]),
    ("medium", "low"): ("Price-Conscious Browsers", "#7F8C8D",
        "Average income but low spend — likely comparison shopping.",
        ["Discount-led acquisition campaigns",
         "Retargeting ads highlighting best-sellers",
         "Simplify the path to purchase (reduce friction/checkout steps)",
         "Educational content to build purchase confidence"]),
    ("low", "high"): ("Aspirational Spenders", "#D62839",
        "Lower income but high spend — enthusiastic, brand-loyal shoppers who stretch their budget.",
        ["Instalment/financing options to sustain spend sustainably",
         "Reward loyalty heavily — they're highly responsive to it",
         "Community & social proof marketing (they enjoy status signaling)",
         "Watch for churn risk if budget tightens — proactive retention offers"]),
    ("low", "medium"): ("Emerging Shoppers", "#457B9D",
        "Limited income, moderate spend — building their relationship with the brand.",
        ["Entry-level product lines and starter bundles",
         "Student/young-professional discount programs",
         "Educational, low-pressure nurture email sequences",
         "Social media-first engagement (high channel affinity)"]),
    ("low", "low"): ("Budget Minimalists", "#8D6E63",
        "Low income and low spend — price is the primary decision factor.",
        ["Deep-discount and clearance campaigns",
         "Essentials-only bundles at low price points",
         "Minimize marketing spend on this segment; focus on retention cost efficiency",
         "Loyalty punch-cards for small, frequent purchases"]),
}


def profile_clusters(df: pd.DataFrame, labels, features):
    """Attach cluster labels to df and build a persona summary per cluster."""
    out = df.copy()
    out["Cluster"] = labels

    personas = []
    income_series = out["Annual_Income_k"] if "Annual_Income_k" in out else None
    spend_series = out["Spending_Score"] if "Spending_Score" in out else None

    for cluster_id in sorted(out["Cluster"].unique()):
        sub = out[out["Cluster"] == cluster_id]
        centroid = {f: round(sub[f].mean(), 1) for f in features}

        income_bucket = _percentile_bucket(sub["Annual_Income_k"].mean(), income_series) \
            if income_series is not None else "medium"
        spend_bucket = _percentile_bucket(sub["Spending_Score"].mean(), spend_series) \
            if spend_series is not None else "medium"

        name, color, blurb, strategies = PERSONA_RULES.get(
            (income_bucket, spend_bucket),
            ("Mixed Segment", "#999999", "A blended customer profile.", ["Run further sub-segmentation."])
        )

        personas.append({
            "cluster_id": int(cluster_id),
            "size": int(len(sub)),
            "pct_of_total": round(100 * len(sub) / len(out), 1),
            "centroid": centroid,
            "income_level": income_bucket,
            "spend_level": spend_bucket,
            "persona_name": name,
            "color": color,
            "blurb": blurb,
            "strategies": strategies,
            "gender_split": sub["Gender"].value_counts(normalize=True).round(3).mul(100).to_dict()
                if "Gender" in sub.columns else {},
        })

    personas.sort(key=lambda p: p["size"], reverse=True)
    return out, personas
