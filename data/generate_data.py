"""
Generates a synthetic 'Mall Customers'-style dataset with realistic,
overlapping customer archetypes so that K-Means finds interesting,
non-trivial clusters.

Columns: CustomerID, Gender, Age, Annual_Income_k, Spending_Score,
         Membership_Years, Visits_Per_Month
"""
import numpy as np
import pandas as pd

np.random.seed(42)


def make_group(n, age_range, income_range, spend_range, member_range, visit_range, gender_bias=0.5):
    age = np.random.randint(age_range[0], age_range[1], n)
    income = np.random.normal(np.mean(income_range), (income_range[1] - income_range[0]) / 5, n)
    income = np.clip(income, 8, 150)
    spend = np.random.normal(np.mean(spend_range), (spend_range[1] - spend_range[0]) / 5, n)
    spend = np.clip(spend, 1, 100)
    membership = np.random.randint(member_range[0], member_range[1], n)
    visits = np.random.normal(np.mean(visit_range), (visit_range[1] - visit_range[0]) / 5, n)
    visits = np.clip(visits, 0, 30)
    gender = np.random.choice(["Male", "Female"], n, p=[1 - gender_bias, gender_bias])
    return pd.DataFrame({
        "Age": age,
        "Annual_Income_k": income.round(1),
        "Spending_Score": spend.round(0).astype(int),
        "Membership_Years": membership,
        "Visits_Per_Month": visits.round(1),
        "Gender": gender,
    })


def generate(n_total=500):
    groups = [
        # (n, age, income, spend, membership_years, visits/mo, female_bias)
        make_group(int(n_total*0.20), (18, 30), (18, 40), (65, 95), (0, 2), (6, 14), 0.55),   # Young high spenders
        make_group(int(n_total*0.18), (28, 45), (70, 130), (60, 90), (2, 8), (4, 10), 0.5),   # Affluent big spenders
        make_group(int(n_total*0.17), (30, 55), (60, 120), (5, 30), (1, 10), (0, 3), 0.45),   # High income, low spend (savers)
        make_group(int(n_total*0.17), (20, 40), (15, 35), (5, 30), (0, 3), (0, 3), 0.5),      # Budget-conscious low spenders
        make_group(int(n_total*0.16), (40, 65), (35, 65), (35, 60), (3, 12), (3, 7), 0.55),   # Average, stable mid-tier
        make_group(n_total - int(n_total*0.20) - int(n_total*0.18) - int(n_total*0.17)
                   - int(n_total*0.17) - int(n_total*0.16), (55, 75), (20, 55), (30, 55), (5, 15), (2, 6), 0.5),  # Senior moderate
    ]
    df = pd.concat(groups, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.insert(0, "CustomerID", range(1, len(df) + 1))
    return df


if __name__ == "__main__":
    df = generate(500)
    df.to_csv("sample_customers.csv", index=False)
    print(f"Generated {len(df)} rows -> sample_customers.csv")
    print(df.head())
