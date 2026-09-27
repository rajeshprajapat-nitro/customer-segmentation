# Segment — Customer Segmentation Studio

A full-stack customer segmentation tool: upload customer data, run K-Means
clustering with model diagnostics (Elbow Method + Silhouette Score), and get
back **marketing-ready personas** for each cluster — all through a web UI.

## Project 9 requirements — how it's covered

| Requirement | Where |
|---|---|
| Pandas | `clustering.py`, `app.py` — data loading & feature prep |
| Scikit-Learn (K-Means) | `clustering.py` — `fit_kmeans`, `compute_elbow`, `compute_silhouette` |
| Seaborn | `app.py` — all chart styling (`sns.set_theme`, `sns.despine`) |
| Inertia / Elbow Method | `compute_elbow()` → plotted on results page |
| Silhouette Score | `compute_silhouette()` → plotted, and used to auto-pick k |
| Cluster → persona translation | `PERSONA_RULES` in `clustering.py` — rule-based mapping of income/spend tertiles to 9 named personas with tailored strategies |
| Web interface | Flask app (`app.py` + `templates/`) |

## Features

- **Upload your own CSV** or use the built-in 500-row synthetic dataset
  (`data/generate_data.py`) modeled on the classic "Mall Customers" dataset.
- **Auto or manual k** — the app picks the k with the best silhouette score,
  or you can force k = 3–6.
- **Diagnostics dashboard** — Elbow plot, Silhouette-vs-k plot, cluster
  scatter plot, and segment-size bar chart, all rendered server-side with
  Seaborn/Matplotlib.
- **Persona engine** — every cluster centroid is bucketed into income/spend
  tertiles (low/medium/high) and mapped to one of 9 personas (e.g. *"Premium
  Loyalists"*, *"Aspirational Spenders"*, *"Budget Minimalists"*), each with
  a blurb and 4 concrete marketing actions.
- **Downloadable results** — export the original data with an added
  `Cluster` column as CSV.

## Setup

```bash
cd customer-segmentation
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5050** in your browser.

## Using your own data

Your CSV should contain at least 2 of these columns (case-sensitive):

- `Age`
- `Annual_Income_k` (annual income in thousands)
- `Spending_Score` (1–100)
- `Membership_Years`
- `Visits_Per_Month`
- `Gender` (optional, used for persona gender-split stats)

Extra columns are preserved and passed through to the downloadable output,
they just aren't used as clustering features unless named above.

## Project structure

```
customer-segmentation/
├── app.py                  # Flask routes: upload → analyze → results
├── clustering.py           # K-Means pipeline + persona rule engine
├── data/
│   ├── generate_data.py    # synthetic dataset generator
│   └── sample_customers.csv
├── templates/
│   ├── index.html          # upload / config page
│   └── results.html        # diagnostics + persona dashboard
├── static/css/style.css
└── requirements.txt
```

## Customizing personas

Edit `PERSONA_RULES` in `clustering.py`. Each key is an
`(income_tier, spend_tier)` tuple → `(name, color, blurb, [strategies])`.
Add your own rules or change the tertile logic in `_percentile_bucket()` if
you want quartiles instead.

## Notes for the resume / portfolio write-up

This project intentionally goes past "just running KMeans.fit()" — the
interesting engineering problem is **Step 4: translating unlabeled numeric
centroids into a business-readable persona**, which is what recruiters in
marketing analytics roles will ask about. Be ready to explain:

- Why you standardized features before clustering (K-Means is distance-based).
- Why silhouette score (cluster cohesion/separation) is a better k-selection
  signal than inertia alone (inertia always decreases with k).
- How the rule-based persona mapping could be extended to LLM-generated
  persona copy for production use.
