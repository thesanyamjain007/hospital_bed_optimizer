# app.py
# Frontend: Streamlit GUI calling backend_models functions.

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io

from backend_models import (
    process_dataset,
    create_train_test_splits,
    train_all_algorithms,
    build_comparison_table,
    export_model,
    get_model_bytes,
    load_uploaded_pipeline
)

st.set_page_config(
    page_title="Hospital Bed Optimization – Structured 5-POC App",
    layout="wide"
)

st.markdown(
    """
    <style>
    body {
        background-color: #F5F9FC;
        color: #003366;
    }
    .main {
        background-color: #F5F9FC;
    }
    .stSidebar {
        background-color: #E7F1FA;
    }
    .stButton>button {
        background-color: #0093D5;
        color: white;
        border-radius: 5px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #007BB5;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Hospital Bed Optimization – 5 Algorithms (Backend + Frontend)")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Upload & Inspect",
        "Processing + EDA",
        "Train All Models",
        "POC Details",
        "Comparison Table",
        "Predict (Saved Model)"
    ]
)

# Session state
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None

if "proc" not in st.session_state:
    st.session_state.proc = None  # X, y

if "splits" not in st.session_state:
    st.session_state.splits = None

if "results" not in st.session_state:
    st.session_state.results = None

# PAGE 1: Upload
if page == "Upload & Inspect":
    st.header("Step 1 – Upload & Inspect Dataset")

    file = st.file_uploader("Upload hospital dataset (CSV/Excel)", type=["csv", "xlsx", "xls"])

    if file is not None:
        # Read directly from uploaded file object
        df = pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)

        st.session_state.raw_df = df.copy()

        # Downsample for faster training during demo
        max_rows = 3000  # adjust (e.g. 1000–3000) for speed vs. accuracy
        if len(st.session_state.raw_df) > max_rows:
            st.info(
                f"Dataset has {len(df)} rows; using a random sample of "
                f"{max_rows} rows for faster model training."
            )
            st.session_state.raw_df = st.session_state.raw_df.sample(max_rows, random_state=42)

        st.subheader("Preview")
        st.dataframe(st.session_state.raw_df.head())

        st.subheader("Shape")
        st.write(st.session_state.raw_df.shape)

        st.subheader("Data types")
        st.write(st.session_state.raw_df.dtypes)

        st.subheader("Missing values")
        st.write(st.session_state.raw_df.isnull().sum())

        st.info(
            "Dataset should contain a numeric 'lengthofstay' column. "
            "The app will automatically create 'long_stay_label' (0/1) from it in the backend."
        )
    else:
        st.warning("Upload a dataset to continue.")

# PAGE 2: Processing + EDA
elif page == "Processing + EDA":
    st.header("Step 2 – Processing + EDA (Backend logic, Frontend plots)")

    if st.session_state.raw_df is None:
        st.warning("Upload data first.")
    else:
        df = st.session_state.raw_df.copy()

        st.subheader("2.1 Drop unwanted columns")
        default_drop = [c for c in ["patient_id"] if c in df.columns]
        drop_cols = st.multiselect(
            "Columns to drop",
            options=list(df.columns),
            default=default_drop
        )
        if drop_cols:
            df = df.drop(columns=drop_cols)

        try:
            X, y, label_encoders = process_dataset(df, target_col="long_stay_label")
        except ValueError as e:
            st.error(str(e))
        else:
            st.session_state.proc = {"X": X, "y": y, "label_encoders": label_encoders}

            st.subheader("Correlation Heatmap (Reds)")
            corr = df.select_dtypes(include=[np.number]).corr()
            plt.figure(figsize=(10, 7))
            sns.heatmap(corr, cmap="Reds", annot=False)
            plt.title("Correlation Heatmap – Reds")
            st.pyplot(plt.gcf())

            st.subheader("Target Distribution")
            plt.figure()
            sns.countplot(x=y)
            plt.title("Distribution of long_stay_label")
            st.pyplot(plt.gcf())

            if "age" in df.columns:
                st.subheader("Age vs Long Stay")
                plt.figure()
                sns.histplot(data=df, x="age", hue="long_stay_label", bins=20, kde=False)
                plt.title("Age Distribution by Long Stay Label")
                st.pyplot(plt.gcf())

            st.success("Processing completed (backend_models.process_dataset).")

# PAGE 3: Train All Models
elif page == "Train All Models":
    st.header("Step 3 – Train/Test Split + Train All 5 Algorithms")

    if st.session_state.proc is None:
        st.warning("Run 'Processing + EDA' first.")
    else:
        X = st.session_state.proc["X"]
        y = st.session_state.proc["y"]

        test_size = st.selectbox("Test size", [0.2, 0.25, 0.3], index=0)  # 0.2 fastest
        rs = st.number_input("Random state", value=42, step=1)

        X_train, X_test, y_train, y_test = create_train_test_splits(
            X, y, test_size=test_size, random_state=rs
        )
        st.session_state.splits = (X_train, X_test, y_train, y_test)

        st.write(f"Train: {X_train.shape[0]} samples")
        st.write(f"Test: {X_test.shape[0]} samples")

        if st.button("Train All 5 POCs (backend_models.train_all_algorithms)"):
            results = train_all_algorithms(X_train, X_test, y_train, y_test)
            st.session_state.results = results
            st.success("All 5 algorithms trained using backend_models.py.")

# PAGE 4: POC Details
elif page == "POC Details":
    st.header("Step 4 – Per-Algorithm POC Details")

    if st.session_state.results is None or st.session_state.splits is None:
        st.warning("Train models first (Step 3).")
    else:
        results = st.session_state.results
        X_train, X_test, y_train, y_test = st.session_state.splits

        algo = st.selectbox("Select algorithm", list(results.keys()))
        res = results[algo]

        st.subheader(f"{algo} – Metrics")
        st.write(f"Train accuracy: {res['train_acc']:.4f}")
        st.write(f"Test accuracy: {res['test_acc']:.4f}")
        st.write(f"ROC AUC: {res['roc_auc']:.4f}")
        st.write(f"Optimal threshold (approx.): {res['opt_thr']:.4f}")

        st.write("Confusion matrix:")
        st.write(res["cm"])

        st.write("Classification report (dict):")
        st.json(res["report"])

        st.subheader(f"{algo} – ROC Curve")
        plt.figure()
        plt.plot(res["fpr"], res["tpr"], label=f"{algo} (AUC={res['roc_auc']:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC – {algo}")
        plt.legend()
        st.pyplot(plt.gcf())

        # Sample predictions
        pipe = res["pipeline"]
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(X_test)[:, 1]
        else:
            if hasattr(pipe, "decision_function"):
                scores = pipe.decision_function(X_test)
                y_proba = 1 / (1 + np.exp(-scores))
            else:
                y_proba = pipe.predict(X_test).astype(float)

        y_pred = pipe.predict(X_test)
        n_show = min(6, len(y_test))
        idx = np.arange(n_show)

        sample_df = pd.DataFrame({
            "Actual": y_test.iloc[idx].values,
            "Predicted": y_pred[idx],
            "Predicted_Prob_Class1": y_proba[idx]
        })
        st.subheader(f"{algo} – Sample Predictions (first {n_show})")
        st.dataframe(sample_df)

# PAGE 5: Comparison Table
elif page == "Comparison Table":
    st.header("Step 5 – Comparison Table (All 5 Algorithms)")

    if st.session_state.results is None:
        st.warning("Train the models in 'Train All Models' first.")
    else:
        comp_df = build_comparison_table(st.session_state.results)
        st.subheader("Metrics Comparison")
        st.dataframe(
            comp_df.style.format({
                "Train Accuracy": "{:.4f}",
                "Test Accuracy": "{:.4f}",
                "ROC AUC": "{:.4f}",
                "Precision (class 1)": "{:.4f}",
                "Recall (class 1)": "{:.4f}",
                "F1-score (class 1)": "{:.4f}",
            })
        )
        st.info("Use this as your comparison matrix for all 5 POCs.")

        # -----------------------------------------------------------------
        # Download any trained model
        # -----------------------------------------------------------------
        st.subheader("Download Any Trained Model ⬇️")

        algo_names = list(st.session_state.results.keys())
        selected_algo = st.selectbox(
            "Select the algorithm to download:",
            options=algo_names,
            index=0
        )

        selected_pipeline = st.session_state.results[selected_algo]['pipeline']
        model_bytes = get_model_bytes(selected_pipeline)
        download_filename = f"{selected_algo}_predictor.joblib"

        st.download_button(
            label=f"Download {selected_algo} Model (.joblib)",
            data=model_bytes,
            file_name=download_filename,
            mime="application/octet-stream",
            help="This file contains the complete scikit-learn pipeline for prediction."
        )
        st.caption(f"The downloaded file is named **{download_filename}**.")

# PAGE 6: Predict with Saved Model
elif page == "Predict (Saved Model)":
    st.header("Step 6 – Predict with Imported Model")

    # 1. Upload Model
    st.subheader("1. Load Your Model")
    model_file = st.file_uploader("Upload .joblib file", type=["joblib"], key="model_loader")

    pipeline = None
    if model_file is not None:
        try:
            pipeline = load_uploaded_pipeline(model_file)
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {e}")

    # Only show input options if model is loaded
    if pipeline is not None:
        st.divider()
        st.subheader("2. Enter Patient Data")
        input_method = st.radio("Choose input method:", ["Manual Entry ✍️", "Upload File 📂"])

        # OPTION A: MANUAL ENTRY
        if input_method == "Manual Entry ✍️":
            st.info("Enter details for a single patient below.")

            with st.form("prediction_form"):
                st.write("### Patient Details")

                c1, c2 = st.columns(2)
                with c1:
                    val1 = st.number_input("Age", min_value=0, max_value=120, value=30)
                    val2 = st.selectbox("Gender", ["Male", "Female"])
                    val3 = st.number_input("Admission Deposit", value=0.0)

                with c2:
                    val4 = st.selectbox("Department", ["gynecology", "anesthesia",
                                                       "radiotherapy", "TB & Chest disease", "surgery"])
                    val5 = st.selectbox("Type of Admission", ["Trauma", "Emergency", "Urgent"])
                    val6 = st.number_input("Visitors with Patient", min_value=0, value=2)

                submitted = st.form_submit_button("Predict Result")

            if submitted:
                input_data = {
                    "age": [val1],
                    "gender": [val2],
                    "Admission_Deposit": [val3],
                    "Department": [val4],
                    "Type of Admission": [val5],
                    "Visitors with Patient": [val6],
                    "long_stay_label": [0]  # dummy target
                }

                df_single = pd.DataFrame(input_data)
                st.write("Input Data Preview:", df_single.drop(columns=["long_stay_label"]))

                try:
                    X_single = df_single.drop(columns=["long_stay_label"])
                    prediction = pipeline.predict(X_single)[0]
                    if hasattr(pipeline, "predict_proba"):
                        probability = pipeline.predict_proba(X_single)[0][1]
                    else:
                        probability = 0.0

                    st.divider()
                    if prediction == 1:
                        st.error(f"⚠️ Prediction: Long Stay (High Risk)\nProbability: {probability:.2%}")
                    else:
                        st.success(f"✅ Prediction: Short Stay (Low Risk)\nProbability: {probability:.2%}")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
                    st.warning("Make sure the input fields above match ALL columns used in your training data.")

        # OPTION B: FILE UPLOAD
        elif input_method == "Upload File 📂":
            st.info("Upload a CSV/Excel file with multiple patients.")
            data_file = st.file_uploader("Upload new data", type=["csv", "xlsx"])

            if data_file is not None:
                try:
                    df_new = pd.read_csv(data_file) if data_file.name.lower().endswith(".csv") else pd.read_excel(data_file)
                    st.write(f"Uploaded {df_new.shape[0]} patients.")

                    if st.button("Run Batch Prediction"):
                        if "long_stay_label" not in df_new.columns:
                            df_new["long_stay_label"] = 0

                        X_new = df_new.drop(columns=["long_stay_label"])

                        preds = pipeline.predict(X_new)
                        probs = pipeline.predict_proba(X_new)[:, 1] if hasattr(pipeline, "predict_proba") else [0]*len(preds)

                        results_df = df_new.drop(columns=["long_stay_label"]).copy()
                        results_df["Predicted_Label"] = preds
                        results_df["Probability_Long_Stay"] = probs

                        st.subheader("Prediction Results")

                        def highlight_risk(val):
                            return 'background-color: #ffcccb' if val == 1 else ''

                        st.dataframe(results_df.style.applymap(highlight_risk, subset=['Predicted_Label']))

                        csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Predictions as CSV",
                            csv,
                            "patient_predictions.csv",
                            "text/csv"
                        )
                except Exception as e:
                    st.error(f"Error processing file: {e}")
