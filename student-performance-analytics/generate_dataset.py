"""Generate a synthetic student dataset (no real student data)."""
from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N = 10_000
SUBJECTS = ["maths", "english", "mandarin", "science", "arts", "exercise"]

FIRST = ["Kinara", "Lea", "Corny", "Yao", "Luka", "Bianca", "Alpha", "Aria",
         "Noah", "Mei", "Arjun", "Sara", "Omar", "Zoe", "Ravi", "Anya",
         "Leo", "Nia", "Ethan", "Priya", "Tariq", "Hana", "Ivan", "Lina"]
LAST = ["Zuri", "Jabulani", "Niang", "Ming", "Magic", "Shangwe", "Kenya",
        "Sharma", "Chen", "Okafor", "Patel", "Silva", "Khan", "Novak",
        "Tanaka", "Mensah", "Rossi", "Das", "Lopez", "Weber"]


def main():
    rng = np.random.default_rng(SEED)
    out = Path("data")
    out.mkdir(exist_ok=True)

    grades = rng.choice([1, 2, 3, 4, 5], size=N, p=[0.13, 0.22, 0.29, 0.16, 0.20])
    ability = rng.normal(0, 10, N) + (grades - 3) * 1.2

    df = pd.DataFrame({
        "student_id": [f"STU{i:05d}" for i in range(1, N + 1)],
        "student_name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(N)],
        "academic_year": rng.choice(["2023-24", "2024-25", "2025-26"], N, p=[.25, .35, .40]),
        "grade": grades,
        "gender": rng.choice(["Male", "Female"], N),
        "attendance": np.clip(82 + 0.5 * ability + rng.normal(0, 9, N), 30, 100).round(1),
    })

    for s in SUBJECTS:
        df[s] = np.clip(rng.normal(63, 14, N) + ability + rng.normal(0, 5, N), 0, 100).round(1)

    avg = df[SUBJECTS].mean(axis=1)
    df["gpa"] = (avg / 20).round(2)

    # previous-term values, used for the "Most Improved" cards
    df["prev_average"] = np.clip(avg + rng.normal(0, 6, N), 0, 100).round(1)
    df["prev_attendance"] = np.clip(df["attendance"] + rng.normal(0, 6, N), 0, 100).round(1)

    df.to_csv(out / "students.csv", index=False)
    print(f"Created {len(df):,} synthetic records -> {(out / 'students.csv').resolve()}")


if __name__ == "__main__":
    main()
