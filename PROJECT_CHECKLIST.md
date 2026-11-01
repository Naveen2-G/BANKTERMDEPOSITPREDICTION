# Completion checklist
- [ ] Copy `bank-full.csv` into `data/`.
- [ ] Install/start MongoDB or configure Atlas in `.env`.
- [ ] Install Python dependencies.
- [ ] Run `python src/inspect_dataset.py`.
- [ ] Run `python src/import_data.py`.
- [ ] Run `python src/query_data.py`.
- [ ] Run `python src/explore_data.py`.
- [ ] Run `python src/train_evaluate.py`.
- [ ] Review actual metrics and charts.
- [ ] Run `python src/predict_customer.py`.
- [ ] Install `imbalanced-learn` and run `python src/smote_experiment.py`.
- [ ] Review `outputs/smote_experiment/evaluation_results.json`,
      `model_comparison.csv`, reports, and charts.
- [ ] Confirm the SMOTENC model is loaded separately from
      `models/random_forest_model.joblib`.
- [ ] Document whether SMOTENC improved subscriber recall and F1-score.
- [ ] Install Streamlit and Plotly from `requirements.txt`.
- [ ] Start the web application with `streamlit run app.py`.
- [ ] Verify Dashboard, Dataset Explorer, Data Analysis, Model Performance,
      Model Comparison, Customer Prediction, Prediction History, and About Project.
- [ ] Confirm prediction history is written only to the `PredictionHistory`
      collection.
- [ ] Update report with actual metrics if required by instructor.
- [ ] Repackage submission after successful execution.
