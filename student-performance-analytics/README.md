# Student Performance Analytics Dashboard
**Big Data Analytics Mini Project — Python, Apache Spark (PySpark), Streamlit, Plotly**

An interactive dashboard that uses Apache Spark to clean, process and aggregate 10,000 student records, then visualises academic performance (grades, subjects, attendance, pass/fail, top performers and students needing review).

> All data is **synthetic** (randomly generated). No real student information is used.

---

## 1. Problem Statement
Institutions collect marks, attendance and demographic data for many students. Analysing it manually makes it hard to spot at-risk students, compare subjects, and see trends. This project uses Spark to process the data and an interactive dashboard to present the results.

## 2. Features
| Area | What it shows |
|---|---|
| KPI cards | Student count, attendance %, exam average, pass rate, students needing review |
| Student Count | Donut chart + bar per grade (count and percentage) |
| Top Performers | Best in Marks, Best in Attendance, Most Improved in Marks, Most Improved in Attendance |
| Examination Results | Pass / Average / Fail counts for each of 6 subjects |
| Average Score | Ring chart per subject |
| Student Details | Top 5 students of a selected grade |
| Review list | Students with average < 50% or attendance below threshold, with CSV download |
| Statistics tab | Descriptive stats (mean, median, std dev, quartiles, skewness, CV%), score histogram, box plot by grade, ANOVA across grades, Welch t-test for gender (p-values, effect sizes) |
| Correlation & Trends tab | Spark MLlib correlation matrix, attendance vs marks regression (Pearson r, R²), grade × subject heatmap, year-wise trend, gender violin plot |
| ML Insights tab | Random Forest early-warning model (ROC-AUC, accuracy, F1, confusion matrix, feature importance) and K-Means student segmentation (silhouette score) using Spark MLlib |
| Student Lookup tab | Search by Student ID: percentile rank, change since last term, radar chart vs grade average |
| Auto insights | Key findings generated from the data on the Overview tab |
| Filters | Academic year, grade, gender, fail/pass marks, attendance threshold |

## 3. Technology Stack
| Layer | Technology |
|---|---|
| Processing engine | Apache Spark 3.5.5 (PySpark, local mode) |
| Dashboard | Streamlit + Plotly |
| Machine learning | Spark MLlib (RandomForestClassifier, KMeans, Correlation) |
| Statistics | SciPy (ANOVA, t-test), Spark aggregations |
| Data handling | Pandas, NumPy |
| Storage | CSV (input), Parquet (processed output); optional HDFS |
| Language | Python 3.10 / 3.11 |

## 4. Architecture
```
Student CSV (10,000 rows)
        |
Data cleaning & validation (Spark)
        |
Derived metrics: average score, marks gain, attendance gain
        |
Parquet output  ----> optional HDFS storage
        |
Distributed aggregations (groupBy, agg, filter)
        |
Streamlit dashboard: charts, filters, review list
```

## 5. Folder Structure
```
student-performance-analytics/
├── app.py                # Spark processing + Streamlit dashboard
├── generate_dataset.py   # Synthetic dataset generator
├── requirements.txt      # Python dependencies
├── run.bat               # One-click start (Windows)
├── run.sh                # One-click start (Mac/Linux)
├── README.md
├── data/
│   └── students.csv      # Generated dataset (auto-created if missing)
└── output/
    └── students_parquet/ # Processed Spark output
```

## 6. Prerequisites
1. **Python 3.10 or 3.11**
2. **Java 17 or 21 — IMPORTANT.** Spark 3.5 does **not** work with Java 22 or newer (e.g. Java 24/25/27). Download Java 17 (Temurin) from https://adoptium.net.
3. A terminal opened **inside the project folder** (the folder containing `app.py`).

Check Java: `java -version` must show version 17 (or 21).

## 7. Installation & Running

### Quick start
- **Windows:** double-click `run.bat`
- **Mac/Linux:** `./run.sh`

### Manual steps
```bash
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Mac/Linux:    source .venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py        # optional; app creates data if missing
streamlit run app.py
```
Open **http://localhost:8501**. The first load takes 30–60 seconds while Spark starts. Keep the terminal open while using the dashboard.

### Setting Java 17 (Windows)
**Temporary (current Command Prompt window only):**
```
where java
set "JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-17.0.20.1-hotspot"
set "PATH=%JAVA_HOME%\bin;%PATH%"
java -version
```
Use the real folder name shown on your PC (do not include `\bin`).

**Permanent:** Start menu → "Edit the system environment variables" → Environment Variables → New user variable `JAVA_HOME` = your Java 17 folder → edit `Path`, add `%JAVA_HOME%\bin` and move it to the top → OK → open a **new** terminal.

## 8. Dataset
10,000 synthetic records (fixed random seed 42, so results are reproducible).

| Column | Description |
|---|---|
| student_id | Unique ID (STU00001...) |
| student_name | Synthetic name |
| academic_year | 2023-24, 2024-25, 2025-26 |
| grade | 1 to 5 |
| gender | Male / Female |
| maths, english, mandarin, science, arts, exercise | Subject marks (0–100) |
| attendance | Attendance % |
| gpa | Average mark / 20 (5-point scale) |
| prev_average, prev_attendance | Previous-term values, used for "Most Improved" |

## 9. How Spark Is Used
1. **Read** the CSV into a Spark DataFrame (header + inferSchema).
2. **Validate:** check required columns, drop duplicate `student_id`, cast types, filter out-of-range marks/attendance/GPA.
3. **Derive** `average_score`, `marks_gain`, `attendance_gain`.
4. **Persist** processed data as Parquet (skipped safely if the Windows Hadoop libraries are missing).
5. **Aggregate** with `groupBy`/`agg`/`filter` for every chart and KPI, then convert only the small results to Pandas for plotting.

Spark runs in **local mode** (`local[*]`, all CPU cores). This demonstrates Spark-based processing; it is **not** a multi-node Hadoop cluster.

## 10. Business Rules
- **Pass/Average/Fail per subject:** Fail < "Fail below" (default 40), Average 40–59, Pass ≥ "Pass from" (default 60). Adjustable in the sidebar.
- **Overall result:** Pass if average score ≥ "Fail below".
- **Needs Review:** average score < 50 **or** attendance < threshold (default 75). This is only a screening indicator, not a judgement about any student.

## 11. Testing Checklist
| Test | Expected result |
|---|---|
| Run generator | Creates 10,000 records in `data/students.csv` |
| Open dashboard | KPIs: Student Count 10,000 with default filters |
| Change grade filter | Counts and all charts update |
| Change thresholds | Pass/Average/Fail and review list recalculate |
| Download review list | CSV file downloads |
| Check `output/` | `students_parquet` folder created |

## 12. Troubleshooting
| Problem | Cause / Fix |
|---|---|
| "Dataset not found" | Run from the folder containing `app.py`; app auto-creates data otherwise |
| "Another SparkContext is being constructed" repeating | Spark failed to start, almost always **wrong Java version**. Install Java 17/21 |
| Red box: "Java 27 detected..." | Same: switch to Java 17 or 21 |
| "The system cannot find the path specified" | `JAVA_HOME` points to a folder that doesn't exist. Run `where java` and use the real folder |
| winutils.exe / native-hadoop warnings | Harmless on Windows, ignore |
| `java.lang.System` / `--enable-native-access` warnings | Harmless |
| `use_container_width` warnings | Harmless Streamlit deprecation notice |
| `contentscript.js` / `MaxListenersExceededWarning` in browser console | Caused by a browser extension, not this project |
| First load very slow | Spark startup, wait up to 60 seconds |
| Parquet export skipped | Missing winutils on Windows; the analysis still works |

## 13. Optional: Hadoop HDFS Integration
On a configured Hadoop system:
```bash
hdfs dfs -mkdir -p /student-analytics/input
hdfs dfs -put -f data/students.csv /student-analytics/input/
```
Then in `load_data` in `app.py`, change the CSV path to:
`hdfs://localhost:9000/student-analytics/input/students.csv` (use your NameNode address).

## 14. Limitations
- Synthetic data, so results demonstrate the method, not real school outcomes.
- Spark local mode on one machine; no cluster.
- The review flag is a simple rule; the ML model is trained on synthetic data, so its accuracy is illustrative only.

## 15. Future Scope
- Real institutional datasets and HDFS/cloud storage
- Spark MLlib model to predict at-risk students
- Trend analysis across multiple terms
- Login and role-based access for teachers/principals
- Deployment on a Spark cluster

## 16. Viva Quick Answers
- **Why Spark?** Fast in-memory distributed processing that scales from 10,000 rows to millions without changing the code.
- **Why Parquet?** Columnar, compressed, and efficient for analytical queries.
- **What is local mode?** Spark runs on one machine using all cores; the same code runs on a cluster.
- **Why synthetic data?** Privacy and reproducibility.
- **What does Streamlit do?** Turns Python scripts into interactive web dashboards.
- **What is ANOVA / t-test?** Tests whether differences between grades / genders are real or due to chance (p < 0.05 means significant).
- **What is ROC-AUC?** How well the model separates at-risk from on-track students (0.5 = random, 1.0 = perfect).
- **Why K-Means?** Unsupervised grouping of students into segments without predefined labels.
- **Why `cache_resource`?** The cached object is a Spark DataFrame, which can't be serialised like ordinary data.

## 17. Credits
Dashboard layout inspired by a student performance dashboard design on Dribbble by Lydia Were. All code and data in this project are original and synthetic.
