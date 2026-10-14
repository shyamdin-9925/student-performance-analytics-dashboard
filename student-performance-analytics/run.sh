#!/usr/bin/env bash
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_dataset.py
streamlit run app.py
