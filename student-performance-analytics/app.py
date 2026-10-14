"""Student Performance Analytics Dashboard - Apache Spark (SQL + MLlib) + Streamlit."""
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (BinaryClassificationEvaluator, ClusteringEvaluator,
                                   MulticlassClassificationEvaluator)
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.sql import SparkSession, functions as F
from scipy import stats

st.set_page_config(page_title="Student Performance Dashboard", page_icon="🎓", layout="wide")

DATA_PATH = Path("data/students.csv")
OUT = Path("output")
OUT.mkdir(exist_ok=True)
SUBJECTS = ["maths", "english", "mandarin", "science", "arts", "exercise"]
PURPLE, BLUE, AMBER = "#3b00c4", "#3b9cff", "#f59e0b"
GRADE_COLORS = ["#1e6bff", "#22d3ee", "#d03bff", "#3ee0c5", "#f0d51a"]


def check_java():
    """Spark 3.5 supports Java 8/11/17/21 only. Newer Java (22+) makes Spark fail to start."""
    import re, subprocess
    try:
        out = subprocess.run(["java", "-version"], capture_output=True, text=True).stderr
        major = int(re.search(r'version "(\d+)', out).group(1))
    except Exception:
        st.error("Java not found. Install Java 17 (Temurin) and set JAVA_HOME.")
        st.stop()
    if major > 21:
        st.error(f"Java {major} detected, but Spark 3.5 needs Java 17 or 21. "
                 "Install Java 17 (adoptium.net), set JAVA_HOME to it, restart the terminal, then run again.")
        st.stop()


@st.cache_resource
def get_spark():
    return (SparkSession.builder.appName("StudentPerformance")
            .master("local[*]").config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.showConsoleProgress", "false").getOrCreate())


@st.cache_resource  # returns a Spark DataFrame, so cache_resource (not cache_data)
def load_data(path, modified):
    spark = get_spark()
    df = spark.read.option("header", True).option("inferSchema", True).csv(path)

    required = ["student_id", "student_name", "academic_year", "grade", "gender",
                "attendance", "gpa", "prev_average", "prev_attendance", *SUBJECTS]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ---- cleaning & validation ----
    df = df.dropDuplicates(["student_id"])
    for c in ["grade", "attendance", "gpa", "prev_average", "prev_attendance", *SUBJECTS]:
        df = df.withColumn(c, F.col(c).cast("double"))
    valid = (F.col("student_id").isNotNull() & F.col("academic_year").isNotNull()
             & F.col("grade").between(1, 5) & F.col("attendance").between(0, 100)
             & F.col("gpa").between(0, 5))
    valid = reduce(lambda a, c: a & F.col(c).between(0, 100), SUBJECTS, valid)
    df = df.filter(valid)

    # ---- derived metrics ----
    avg = reduce(lambda a, b: a + b, [F.col(c) for c in SUBJECTS]) / len(SUBJECTS)
    df = (df.withColumn("average_score", F.round(avg, 2))
            .withColumn("marks_gain", F.round(F.col("average_score") - F.col("prev_average"), 2))
            .withColumn("attendance_gain", F.round(F.col("attendance") - F.col("prev_attendance"), 2)))

    df = df.cache()
    try:  # Parquet export; on Windows without winutils this can fail, so don't crash the app
        df.write.mode("overwrite").parquet(str(OUT / "students_parquet"))
    except Exception as exc:
        print(f"[warning] Parquet export skipped: {exc}")
    return df


@st.cache_data(show_spinner="Training Spark MLlib models...")
def train_ml(_df, key):
    """Random Forest early-warning model + K-Means student segmentation (Spark MLlib)."""
    # ---- 1. Random Forest: predict 'needs review' from PREVIOUS-term data only ----
    d = (_df.withColumn("gender_num", F.when(F.col("gender") == "Female", 1.0).otherwise(0.0))
            .withColumn("label", F.when((F.col("average_score") < 50) | (F.col("attendance") < 75), 1.0).otherwise(0.0)))
    feats = ["prev_average", "prev_attendance", "grade", "gender_num"]
    data = VectorAssembler(inputCols=feats, outputCol="features").transform(d).select("features", "label")
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    model = RandomForestClassifier(numTrees=50, maxDepth=6, seed=42).fit(train)
    pred = model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC").evaluate(pred)
    mce = lambda m: MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName=m).evaluate(pred)
    cm = pred.groupBy("label", "prediction").count().toPandas()
    matrix = np.zeros((2, 2), dtype=int)
    for _, r in cm.iterrows():
        matrix[int(r["label"]), int(r["prediction"])] = int(r["count"])
    rf = dict(auc=auc, accuracy=mce("accuracy"), f1=mce("f1"), train=train.count(), test=test.count(),
              matrix=matrix, importance=dict(zip(["Previous Avg", "Previous Attendance", "Grade", "Gender"],
                                                 model.featureImportances.toArray())))

    # ---- 2. K-Means: segment students by marks + attendance ----
    x = VectorAssembler(inputCols=["average_score", "attendance"], outputCol="raw").transform(
        _df.select("student_id", "average_score", "attendance"))
    x = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True).fit(x).transform(x)
    km = KMeans(k=3, seed=42).fit(x)
    out = km.transform(x)
    sil = ClusteringEvaluator().evaluate(out)
    summ = (out.groupBy("prediction").agg(F.count("*").alias("Students"),
                                          F.round(F.avg("average_score"), 1).alias("Avg Score"),
                                          F.round(F.avg("attendance"), 1).alias("Avg Attendance"))
            .toPandas().sort_values("Avg Score").reset_index(drop=True))
    names = dict(zip(summ["prediction"], ["At-Risk", "Average", "High Achievers"]))
    summ.insert(0, "Segment", summ["prediction"].map(names))
    summ = summ.drop(columns="prediction")
    samp = out.sample(False, 0.25, seed=42).select("average_score", "attendance", "prediction").toPandas()
    samp["Segment"] = samp["prediction"].map(names)
    return rf, dict(silhouette=sil, summary=summ, sample=samp)


# ---------------- styling ----------------
st.markdown("""
<style>
.stApp { background:#f4f5ff; }
header[data-testid="stHeader"] { background:transparent; }
.topbar { background:#1e0a5c; color:white; padding:16px 24px; border-radius:12px; margin-bottom:18px; }
.topbar h2 { color:white; margin:0; font-size:22px; }
.topbar span { color:#b9b0e6; font-size:13px; }
[data-testid="stMetric"] { background:white; padding:14px 16px; border:1px solid #dedcf0; border-radius:12px; }
.card { background:white; border:1px solid #dedcf0; border-radius:12px; padding:14px; }
.card .big { font-size:24px; font-weight:700; color:#4a4a68; }
.card .lbl { color:#30205e; font-weight:600; margin-top:6px; }
.card .sub { color:#8a88a8; font-size:12px; }
.insight { background:white; border-left:5px solid #3b00c4; border-radius:8px; padding:10px 16px; margin:6px 0; color:#30205e; }
h3 { color:#30205e; }
</style>
<div class="topbar"><h2>🎓 EduPulse Academy</h2>
<span>Student Performance Dashboard &nbsp;•&nbsp; Apache Spark SQL + MLlib + Statistics (synthetic data)</span></div>
""", unsafe_allow_html=True)

if not DATA_PATH.exists():  # auto-create the synthetic dataset on first run
    import generate_dataset
    generate_dataset.main()

check_java()
try:
    sdf = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not process dataset: {exc}")
    st.stop()

# ---------------- sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    years = [r[0] for r in sdf.select("academic_year").distinct().orderBy("academic_year").collect()]
    grades = [int(r[0]) for r in sdf.select("grade").distinct().orderBy("grade").collect()]
    genders = [r[0] for r in sdf.select("gender").distinct().orderBy("gender").collect()]
    sel_years = st.multiselect("Academic year", years, default=years)
    sel_grades = st.multiselect("Grade", grades, default=grades)
    sel_genders = st.multiselect("Gender", genders, default=genders)
    st.caption("Result thresholds")
    fail_below = st.slider("Fail below (avg %)", 0, 100, 40)
    pass_from = st.slider("Pass from (avg %)", 0, 100, 60)
    att_threshold = st.slider("Attendance threshold (%)", 0, 100, 75)

f = sdf.filter(F.col("academic_year").isin(sel_years) & F.col("grade").isin(sel_grades)
               & F.col("gender").isin(sel_genders))
f = f.withColumn("risk_flag", F.when((F.col("average_score") < 50) | (F.col("attendance") < att_threshold),
                                     "Needs Review").otherwise("On Track")).cache()

s = f.agg(F.count("*").alias("n"), F.avg("attendance").alias("att"), F.avg("average_score").alias("avg"),
          F.avg(F.when(F.col("average_score") >= fail_below, 100.0).otherwise(0.0)).alias("pass"),
          F.sum(F.when(F.col("risk_flag") == "Needs Review", 1).otherwise(0)).alias("risk")).first()
n = s["n"] or 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Student Count", f"{n:,}")
k2.metric("Student Attendance", f"{(s['att'] or 0):.1f}%")
k3.metric("Exam Average", f"{(s['avg'] or 0):.1f}%")
k4.metric("Pass Rate", f"{(s['pass'] or 0):.1f}%")
k5.metric("Needs Review", f"{(s['risk'] or 0):,}")
if n < 30:
    st.warning("Please select at least 30 students (adjust filters) for meaningful analysis.")
    st.stop()

# shared random sample (for box plots / scatter)
samp = f.select("student_id", "grade", "gender", "attendance", "average_score").sample(
    False, min(1.0, 3000 / n), seed=42).toPandas()
samp["Grade"] = "Grade " + samp["grade"].astype(int).astype(str)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "📐 Statistics", "🔗 Correlation & Trends", "🤖 ML Insights", "🔎 Student Lookup"])

# =====================================================================
# TAB 1 - OVERVIEW
# =====================================================================
with tab1:
    avgs = f.agg(*[F.round(F.avg(c), 1).alias(c) for c in SUBJECTS]).first()
    r_att = f.stat.corr("attendance", "average_score")
    best = max(SUBJECTS, key=lambda c: avgs[c]); worst = min(SUBJECTS, key=lambda c: avgs[c])
    st.subheader("Key Insights (auto-generated by Spark)")
    for text in [
        f"Strongest subject is <b>{best.title()}</b> ({avgs[best]}%); weakest is <b>{worst.title()}</b> ({avgs[worst]}%).",
        f"<b>{(s['risk'] or 0) / n * 100:.1f}%</b> of students ({s['risk']:,}) are flagged for review "
        f"(average below 50% or attendance below {att_threshold}%).",
        f"Attendance and exam average have a Pearson correlation of <b>r = {r_att:.2f}</b> "
        f"({'strong' if abs(r_att) > .5 else 'moderate' if abs(r_att) > .3 else 'weak'} "
        f"{'positive' if r_att > 0 else 'negative'} relationship).",
    ]:
        st.markdown(f'<div class="insight">{text}</div>', unsafe_allow_html=True)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Student Count")
        g = f.groupBy("grade").agg(F.count("*").alias("students")).orderBy("grade").toPandas()
        g["grade"] = "Grade " + g["grade"].astype(int).astype(str)
        g["share"] = (g["students"] / g["students"].sum() * 100).round(1)
        c1, c2 = st.columns([1, 1.3])
        with c1:
            fig = px.pie(g, names="grade", values="students", hole=0.62, color_discrete_sequence=GRADE_COLORS)
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(g, y="grade", x="students", orientation="h", text="share",
                         color="grade", color_discrete_sequence=GRADE_COLORS)
            fig.update_traces(texttemplate="%{text}%")
            fig.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                              yaxis=dict(autorange="reversed", title=""), xaxis_title="Students")
            st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top Performers")

        def top(col):
            return f.orderBy(F.col(col).desc()).first()

        def card(title, r, big, extra):
            icon = "👧" if r["gender"] == "Female" else "👦"
            return (f'<div class="card"><div style="font-size:30px">{icon}</div>'
                    f'<div class="big">{big}</div><div><b>{r["student_name"]}</b></div>'
                    f'<div class="sub">Grade {int(r["grade"])} • GPA {r["gpa"]} • {extra}</div>'
                    f'<div class="lbl">{title}</div></div>')

        b_m, b_a, i_m, i_a = top("average_score"), top("attendance"), top("marks_gain"), top("attendance_gain")
        a, b = st.columns(2)
        a.markdown(card("Best In Marks", b_m, f"{b_m['average_score']:.1f}%", f"Attend {b_m['attendance']:.1f}%"), unsafe_allow_html=True)
        b.markdown(card("Best In Attendance", b_a, f"{b_a['attendance']:.1f}%", f"Marks {b_a['average_score']:.1f}%"), unsafe_allow_html=True)
        st.write("")
        a, b = st.columns(2)
        a.markdown(card("Most Improved In Marks", i_m, f"+{i_m['marks_gain']:.1f}%", f"Marks {i_m['average_score']:.1f}%"), unsafe_allow_html=True)
        b.markdown(card("Most Improved In Attendance", i_a, f"+{i_a['attendance_gain']:.1f}%", f"Attend {i_a['attendance']:.1f}%"), unsafe_allow_html=True)

    st.divider()
    st.subheader("Examination Results")
    aggs = []
    for sub in SUBJECTS:
        aggs += [F.sum(F.when(F.col(sub) >= pass_from, 1).otherwise(0)).alias(f"{sub}|Pass"),
                 F.sum(F.when((F.col(sub) >= fail_below) & (F.col(sub) < pass_from), 1).otherwise(0)).alias(f"{sub}|Average"),
                 F.sum(F.when(F.col(sub) < fail_below, 1).otherwise(0)).alias(f"{sub}|Fail")]
    row = f.agg(*aggs).first()
    res = pd.DataFrame([{"Subject": k.split("|")[0].title(), "Result": k.split("|")[1], "Students": row[k] or 0}
                        for k in row.asDict()])
    fig = px.bar(res, x="Subject", y="Students", color="Result", barmode="group",
                 color_discrete_map={"Pass": PURPLE, "Average": BLUE, "Fail": AMBER},
                 category_orders={"Result": ["Pass", "Average", "Fail"]})
    fig.update_layout(height=380, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Average Score")
    cols = st.columns(len(SUBJECTS))
    for col, sub in zip(cols, SUBJECTS):
        v = avgs[sub] or 0
        fig = go.Figure(go.Pie(values=[v, 100 - v], hole=0.76, sort=False, direction="clockwise",
                               marker=dict(colors=[PURPLE if v >= 75 else BLUE, "#ece8fb"]),
                               textinfo="none", hoverinfo="skip"))
        fig.update_layout(showlegend=False, height=190, margin=dict(t=5, b=5, l=5, r=5),
                          annotations=[dict(text=f"<b>{v}%</b><br>{sub.title()}", showarrow=False, font=dict(size=14))])
        col.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Student Details")
    pick = st.selectbox("Grade", sorted(sel_grades))
    detail = (f.filter(F.col("grade") == pick).orderBy(F.col("average_score").desc())
              .select("student_id", "student_name", "gender", "average_score", "gpa", "attendance").limit(5).toPandas())
    cols = st.columns(5)
    for col, (_, r) in zip(cols, detail.iterrows()):
        icon = "👧" if r["gender"] == "Female" else "👦"
        col.markdown(f'<div class="card" style="text-align:center"><div style="font-size:34px">{icon}</div>'
                     f'<div class="sub">{r["gender"]}</div><b>{r["student_name"]}</b>'
                     f'<div class="sub">Marks {r["average_score"]}% • GPA {r["gpa"]} • Attend {r["attendance"]}%</div></div>',
                     unsafe_allow_html=True)

    st.subheader("Students Requiring Review")
    st.caption("Screening flag only: average below 50% or attendance below the threshold. Not a final judgment on any student.")
    risk = (f.filter(F.col("risk_flag") == "Needs Review")
            .select("student_id", "student_name", "academic_year", "grade", "gender", "average_score", "attendance", "gpa")
            .orderBy("average_score", "attendance").limit(500).toPandas())
    st.dataframe(risk, use_container_width=True, hide_index=True)
    if not risk.empty:
        st.download_button("Download review list (CSV)", risk.to_csv(index=False).encode(),
                           "students_for_review.csv", "text/csv")

    with st.expander("Data quality & processing"):
        st.write(f"Valid unique records after Spark cleaning: **{sdf.count():,}**")
        if (OUT / "students_parquet").exists():
            st.write(f"Processed data stored as Parquet in `{OUT / 'students_parquet'}`")
        else:
            st.write("Parquet export skipped on this machine (Windows Hadoop libraries missing); analysis still ran in Spark.")

# =====================================================================
# TAB 2 - STATISTICS
# =====================================================================
with tab2:
    st.subheader("Descriptive Statistics (computed with Spark)")
    metrics_cols = SUBJECTS + ["attendance", "average_score"]
    ex = []
    for c in metrics_cols:
        ex += [F.round(F.avg(c), 2).alias(f"{c}|Mean"),
               F.round(F.expr(f"percentile_approx({c}, 0.5)"), 2).alias(f"{c}|Median"),
               F.round(F.stddev(c), 2).alias(f"{c}|Std Dev"),
               F.min(c).alias(f"{c}|Min"), F.max(c).alias(f"{c}|Max"),
               F.round(F.expr(f"percentile_approx({c}, 0.25)"), 2).alias(f"{c}|Q1"),
               F.round(F.expr(f"percentile_approx({c}, 0.75)"), 2).alias(f"{c}|Q3"),
               F.round(F.skewness(c), 2).alias(f"{c}|Skewness")]
    drow = f.agg(*ex).first()
    stat_names = ["Mean", "Median", "Std Dev", "Min", "Q1", "Q3", "Max", "Skewness"]
    desc = pd.DataFrame({c.replace("_", " ").title(): [drow[f"{c}|{m}"] for m in stat_names] for c in metrics_cols},
                        index=stat_names).T
    desc["CV %"] = (desc["Std Dev"] / desc["Mean"] * 100).round(1)
    st.dataframe(desc, use_container_width=True)
    st.caption("Std Dev = spread of marks; CV% (coefficient of variation) = Std Dev / Mean, so subjects can be compared fairly; "
               "Skewness < 0 means most students score above the mean.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Score Distribution")
        h = (f.withColumn("bin", F.floor(F.col("average_score") / 5) * 5).groupBy("bin").count()
             .orderBy("bin").toPandas())
        h["Range"] = h["bin"].astype(int).astype(str) + "-" + (h["bin"] + 5).astype(int).astype(str)
        fig = px.bar(h, x="Range", y="count", labels={"count": "Students"}, color_discrete_sequence=[PURPLE])
        fig.update_layout(height=360, margin=dict(t=10), bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Score Spread by Grade (Box Plot)")
        fig = px.box(samp.sort_values("grade"), x="Grade", y="average_score", color="Grade",
                     color_discrete_sequence=GRADE_COLORS, labels={"average_score": "Average Score"})
        fig.update_layout(height=360, margin=dict(t=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hypothesis Tests")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**ANOVA: do average scores differ across grades?**")
        gs = (f.groupBy("grade").agg(F.count("*").alias("n"), F.avg("average_score").alias("m"),
                                     F.stddev("average_score").alias("s")).toPandas().dropna())
        k, N = len(gs), int(gs["n"].sum())
        if k > 1 and N > k:
            M = (gs["n"] * gs["m"]).sum() / N
            ssb = (gs["n"] * (gs["m"] - M) ** 2).sum()
            ssw = ((gs["n"] - 1) * gs["s"] ** 2).sum()
            fval = (ssb / (k - 1)) / (ssw / (N - k))
            pval = stats.f.sf(fval, k - 1, N - k)
            eta = ssb / (ssb + ssw)
            st.metric("F-statistic", f"{fval:.2f}")
            st.write(f"p-value = **{pval:.4g}**, effect size (η²) = {eta:.3f}")
            st.success("Significant difference between grades (p < 0.05)." if pval < 0.05
                       else "No significant difference between grades (p ≥ 0.05).")
        else:
            st.info("Select at least two grades.")
    with t2:
        st.markdown("**Welch t-test: do male and female averages differ?**")
        gg = (f.groupBy("gender").agg(F.count("*").alias("n"), F.avg("average_score").alias("m"),
                                      F.stddev("average_score").alias("s")).toPandas().dropna())
        if len(gg) == 2:
            a, b = gg.iloc[0], gg.iloc[1]
            tstat, pv = stats.ttest_ind_from_stats(a["m"], a["s"], a["n"], b["m"], b["s"], b["n"], equal_var=False)
            sp = np.sqrt(((a["n"] - 1) * a["s"] ** 2 + (b["n"] - 1) * b["s"] ** 2) / (a["n"] + b["n"] - 2))
            d = (a["m"] - b["m"]) / sp
            st.metric("t-statistic", f"{tstat:.2f}")
            st.write(f"{a['gender']} mean = {a['m']:.2f}, {b['gender']} mean = {b['m']:.2f}; "
                     f"p-value = **{pv:.4g}**, Cohen's d = {d:.3f}")
            st.success("Significant gender difference (p < 0.05)." if pv < 0.05
                       else "No significant gender difference (p ≥ 0.05).")
        else:
            st.info("Select both genders to run this test.")

# =====================================================================
# TAB 3 - CORRELATION & TRENDS
# =====================================================================
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation Matrix (Spark MLlib)")
        ccols = SUBJECTS + ["attendance", "prev_average", "average_score"]
        vec = VectorAssembler(inputCols=ccols, outputCol="v").transform(f.select(*ccols)).select("v")
        cm = Correlation.corr(vec, "v").head()[0].toArray()
        labels = [c.replace("_", " ").title() for c in ccols]
        fig = px.imshow(np.round(cm, 2), x=labels, y=labels, text_auto=True, zmin=-1, zmax=1,
                        color_continuous_scale="Purples", aspect="auto")
        fig.update_layout(height=470, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Attendance vs Exam Average")
        slope, intercept = np.polyfit(samp["attendance"], samp["average_score"], 1)
        r = f.stat.corr("attendance", "average_score")
        fig = px.scatter(samp, x="attendance", y="average_score", opacity=0.45, color_discrete_sequence=[BLUE],
                         labels={"attendance": "Attendance %", "average_score": "Average Score"})
        xs = np.array([samp["attendance"].min(), samp["attendance"].max()])
        fig.add_trace(go.Scatter(x=xs, y=slope * xs + intercept, mode="lines", name="Regression line",
                                 line=dict(color=AMBER, width=3)))
        fig.update_layout(height=470, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Pearson **r = {r:.3f}**, R² = **{r * r:.3f}**; "
                 f"each +10% attendance relates to about **{slope * 10:+.1f}** marks (linear regression).")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average Marks: Grade × Subject")
        hm = (f.groupBy("grade").agg(*[F.round(F.avg(c), 1).alias(c) for c in SUBJECTS]).orderBy("grade").toPandas())
        hm.index = "Grade " + hm["grade"].astype(int).astype(str)
        hm = hm.drop(columns="grade"); hm.columns = [c.title() for c in hm.columns]
        fig = px.imshow(hm, text_auto=True, color_continuous_scale="Purples", aspect="auto")
        fig.update_layout(height=360, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Year-wise Trend")
        yt = (f.groupBy("academic_year").agg(F.round(F.avg("average_score"), 2).alias("Average Score"),
                                              F.round(F.avg("attendance"), 2).alias("Attendance"),
                                              F.round(F.avg(F.when(F.col("average_score") >= fail_below, 100.0)
                                                            .otherwise(0.0)), 2).alias("Pass Rate"))
              .orderBy("academic_year").toPandas())
        fig = px.line(yt, x="academic_year", y=["Average Score", "Attendance", "Pass Rate"], markers=True,
                      labels={"academic_year": "Academic Year", "value": "%", "variable": ""})
        fig.update_layout(height=360, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gender Comparison")
    fig = px.violin(samp, x="gender", y="average_score", color="gender", box=True,
                    color_discrete_sequence=[PURPLE, "#d03bff"], labels={"average_score": "Average Score", "gender": ""})
    fig.update_layout(height=340, margin=dict(t=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# TAB 4 - ML INSIGHTS
# =====================================================================
with tab4:
    rf, kmres = train_ml(sdf, DATA_PATH.stat().st_mtime)
    st.subheader("Early-Warning Model: Random Forest (Spark MLlib)")
    st.caption("Predicts whether a student will need review (average < 50% or attendance < 75%) using only "
               "previous-term marks and attendance, grade and gender. Trained on all students, 80/20 train/test split.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{rf['auc']:.3f}")
    m2.metric("Accuracy", f"{rf['accuracy'] * 100:.1f}%")
    m3.metric("F1 Score", f"{rf['f1']:.3f}")
    m4.metric("Train / Test rows", f"{rf['train']:,} / {rf['test']:,}")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.imshow(rf["matrix"], text_auto=True, color_continuous_scale="Purples",
                        x=["Predicted On Track", "Predicted Review"], y=["Actually On Track", "Actually Review"])
        fig.update_layout(title="Confusion Matrix (test set)", height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        imp = pd.DataFrame({"Feature": list(rf["importance"]), "Importance": list(rf["importance"].values())}
                           ).sort_values("Importance")
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[PURPLE])
        fig.update_layout(title="Feature Importance", height=340)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Student Segmentation: K-Means Clustering (Spark MLlib)")
    st.caption(f"Students grouped into 3 segments by marks and attendance (silhouette score = {kmres['silhouette']:.2f}; "
               "closer to 1 means well-separated clusters).")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = px.scatter(kmres["sample"], x="attendance", y="average_score", color="Segment", opacity=0.6,
                         color_discrete_map={"At-Risk": AMBER, "Average": BLUE, "High Achievers": PURPLE},
                         labels={"attendance": "Attendance %", "average_score": "Average Score"})
        fig.update_layout(height=400, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.dataframe(kmres["summary"], use_container_width=True, hide_index=True)
        st.info("Segment sizes and averages describe the whole dataset (not the sidebar filters).")

# =====================================================================
# TAB 5 - STUDENT LOOKUP
# =====================================================================
with tab5:
    st.subheader("Individual Student Report")
    sid = st.text_input("Enter Student ID (e.g. STU00001)", "STU00001").strip().upper()
    stu = sdf.filter(F.col("student_id") == sid).first()
    if stu is None:
        st.warning("Student ID not found.")
    else:
        total = sdf.count()
        below = sdf.filter(F.col("average_score") < stu["average_score"]).count()
        pct = below / total * 100
        grade_avg = sdf.filter(F.col("grade") == stu["grade"]).agg(*[F.avg(c).alias(c) for c in SUBJECTS]).first()
        flagged = stu["average_score"] < 50 or stu["attendance"] < 75
        a, b, c, d, e = st.columns(5)
        a.metric("Student", stu["student_name"])
        b.metric("Grade / Year", f"{int(stu['grade'])} / {stu['academic_year']}")
        c.metric("Average Score", f"{stu['average_score']}%")
        d.metric("Attendance", f"{stu['attendance']}%")
        e.metric("Percentile Rank", f"{pct:.0f}%")
        st.markdown(f'<div class="insight">{stu["student_name"]} scores higher than <b>{pct:.1f}%</b> of all students and is '
                    f'<b>{"flagged for review" if flagged else "on track"}</b>. Change since last term: '
                    f'<b>{stu["marks_gain"]:+.1f}</b> marks, <b>{stu["attendance_gain"]:+.1f}</b> attendance points.</div>',
                    unsafe_allow_html=True)
        cats = [c.title() for c in SUBJECTS]
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[stu[c] for c in SUBJECTS] + [stu[SUBJECTS[0]]], theta=cats + [cats[0]],
                                      fill="toself", name=stu["student_name"], line_color=PURPLE))
        fig.add_trace(go.Scatterpolar(r=[grade_avg[c] for c in SUBJECTS] + [grade_avg[SUBJECTS[0]]], theta=cats + [cats[0]],
                                      name=f"Grade {int(stu['grade'])} average", line_color=AMBER))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=450, title="Subject profile vs grade average")
        st.plotly_chart(fig, use_container_width=True)
