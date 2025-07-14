import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from preprocess import clean_data, encode_categoricals, split_data
from ml_engine import train_model, evaluate_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="ML Model Selector",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Styling ---
st.markdown("""
<style>
.main {background-color: #f8fafc;}
.stButton>button {background-color: #2563eb; color: white;}
.stSidebar {background-color: #f1f5f9;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Machine Learning Model Selector")
st.markdown("<span style='font-size:18px;'>Upload dataset, preview, get smart suggestions, and train models.</span>", unsafe_allow_html=True)

# --- Dataset Summary Function ---
def dataset_summary_and_suggestions(df):
    st.markdown("### 🧠 Dataset Details + AI Suggestions")
    summary_data, cols_to_drop, useful_features = [], [], []
    for col in df.columns:
        dtype = df[col].dtype
        uniques = df[col].nunique()
        null_rate = df[col].isnull().mean()
        examples = df[col].dropna().unique()[:3]
        example_vals = ", ".join(map(str, examples)) + ("..." if uniques > 3 else "")

        if dtype == 'object':
            description = f"Categorical ({uniques} categories)"
        elif pd.api.types.is_numeric_dtype(df[col]):
            description = f"Numerical ({df[col].min()} to {df[col].max()})"
        elif 'date' in col.lower():
            description = "Temporal (date format)"
        else:
            description = f"Type {dtype}"

        if uniques == 1:
            util = "🚫 Constant (to ignore)"
            cols_to_drop.append(col)
        elif null_rate > 0.5:
            util = "⚠️ Too many missing values"
        elif col.lower() in ['id', 'index'] or df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
            util = "🔁 Identifier / index"
            cols_to_drop.append(col)
        else:
            util = "✅ Potentially useful"
            useful_features.append(col)

        summary_data.append({
            "📌 Column": f"`{col}`",
            "🧠 Type": str(dtype),
            "🔢 Uniques": uniques,
            "💬 Example": example_vals,
            "✍️ Description": description,
            "📊 ML Utility": util
        })
    summary_df = pd.DataFrame(summary_data)
    for col in summary_df.columns:
        summary_df[col] = summary_df[col].astype(str)
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### ✅ AI Suggestions")
    st.markdown(f"- **Useful columns for ML :** `{', '.join(useful_features)}`")
    if cols_to_drop:
        st.markdown(f"- **Columns to ignore :** `{', '.join(cols_to_drop)}`")

# --- Target Suggestion ---
def suggest_target(df, model_choice):
    classification_models = ["Random Forest", "SVM", "XGBoost", "KNN", "Decision Tree"]
    regression_models = ["Linear Regression"]
    clustering_models = ["KMeans"]

    if model_choice in classification_models:
        # Pour la classification : privilégier les colonnes avec peu de valeurs uniques (discrètes)
        candidates = [
            col for col in df.columns
            if (df[col].nunique() <= 20 and df[col].dtype in ['object', 'int64', 'category'])
            or (df[col].dtype == 'float64' and df[col].dropna().apply(float.is_integer).all())
        ]
    elif model_choice in regression_models:
        # Pour la régression : privilégier les colonnes numériques continues
        candidates = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20
        ]
    elif model_choice in clustering_models:
        # Pour le clustering : pas de cible nécessaire, retourner None
        return None
    else:
        candidates = df.columns.tolist()

    return candidates[-1] if candidates else df.columns[-1]

# --- PDF Report Generator ---
def generate_pdf_report_reportlab(df, model_choice, params, target_column, results, figures_to_save=[]):
    from reportlab.lib import colors
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    width, height = letter
    y = height - 40

    # Palette
    blue_dark = colors.HexColor("#2563eb")
    blue_light = colors.HexColor("#60a5fa")
    gray_box = colors.HexColor("#f1f5f9")
    white = colors.white
    black = colors.black

    # Header bandeau bleu foncé
    c.setFillColor(blue_dark)
    c.rect(0, y-20, width, 50, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, y, "Automatic Report - Machine Learning")
    y -= 60

    # Dataset info (sous-titre bleu clair)
    c.setFillColor(blue_light)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Dataset : {df.shape[0]} rows, {df.shape[1]} columns")
    y -= 20
    c.drawString(50, y, f"Selected model : {model_choice}")
    y -= 20
    if target_column:
        c.drawString(50, y, f"Target column : {target_column}")
    y -= 30

    # Séparateur coloré
    c.setStrokeColor(blue_dark)
    c.setLineWidth(2)
    c.line(40, y, width-40, y)
    y -= 20

    # Encadré Hyperparameters
    c.setFillColor(gray_box)
    c.roundRect(40, y-20-15*len(params), width-80, 30+15*len(params), 8, fill=1, stroke=0)
    c.setFillColor(blue_dark)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Hyperparameters :")
    y -= 20
    c.setFillColor(black)
    c.setFont("Helvetica", 12)
    for k, v in params.items():
        c.drawString(70, y, f"- {k} : {v}")
        y -= 15
    y -= 10

    # Encadré Model results
    c.setFillColor(gray_box)
    c.roundRect(40, y-60, width-80, 60, 8, fill=1, stroke=0)
    c.setFillColor(blue_dark)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Model results :")
    y -= 20
    c.setFillColor(black)
    c.setFont("Helvetica", 12)
    if model_choice.lower() == "linear regression":
        c.drawString(70, y, f"MSE : {results.get('mse', 'N/A'):.4f}")
        y -= 15
        c.drawString(70, y, f"R2 Score : {results.get('r2', 'N/A'):.4f}")
        y -= 30
    elif model_choice.lower() == "kmeans":
        c.drawString(70, y, "Clustering completed (no accuracy/F1 for KMeans)")
        y -= 30
    else:
        c.drawString(70, y, f"Accuracy : {results.get('accuracy', 'N/A'):.2%}")
        y -= 15
        c.drawString(70, y, f"F1 Score : {results.get('f1_score', 'N/A'):.2%}")
        y -= 30

    # Séparateur coloré
    c.setStrokeColor(blue_light)
    c.setLineWidth(1.5)
    c.line(40, y, width-40, y)
    y -= 20

    # Ajouter tous les graphiques disponibles dans figures_to_save
    for fig in figures_to_save:
        if y < 200:  # Nouvelle page si l'espace est insuffisant
            c.showPage()
            y = height - 40
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.write_image(tmp_img.name, format="png", width=800, height=600, scale=2)
        tmp_img.close()
        img = ImageReader(tmp_img.name)
        img_width, img_height = img.getSize()
        aspect = img_height / img_width
        display_width = width - 100
        display_height = display_width * aspect
        if display_height > y - 50:
            display_height = y - 50
            display_width = display_height / aspect
        c.drawImage(img, 50, y - display_height, width=display_width, height=display_height)
        y -= (display_height + 20)
        os.unlink(tmp_img.name)

    c.save()
    return tmp_file.name

# --- Sidebar Navigation ---

# Initialiser la liste des figures d'exploration dans la session
if "exploration_figures" not in st.session_state:
    st.session_state.exploration_figures = []

st.sidebar.header("1️⃣ Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("2️⃣ Navigation")
page = st.sidebar.radio("Go to:", ["Data Exploration", "Training and Evaluation"])
st.session_state.page = page

exploration_figures = st.session_state.exploration_figures  # Utiliser la session pour persister
# --- Load dataset and continue ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Auto-convert datetime
    date_exclude = ['month', 'year', 'day', 'hour', 'minute', 'second']
    for col in df.select_dtypes(include=['object']).columns:
        if col.lower() in date_exclude:
            continue
        try:
            converted = pd.to_datetime(df[col], errors='coerce')
            if converted.notna().sum() / len(df) > 0.8:
                df[col] = converted
        except Exception:
            pass

    if st.session_state.page == "Data Exploration":
        st.title("📊 Data Exploration")

        # Utilise la liste globale exploration_figures pour stocker toutes les figures d'exploration

        # Dataset Preview
        st.subheader("📊 Dataset Preview")
        st.markdown("**Preview of the first few rows of your dataset to understand its structure**")
        st.dataframe(df.head(), use_container_width=True)
        st.markdown(f"**Rows:** {df.shape[0]}   **Columns:** {df.shape[1]}")

        # Statistical Summary
        st.markdown("### 📑 Statistical Summary")
        st.markdown("**Descriptive statistics of your numerical data (mean, standard deviation, min, max, etc.)**")
        st.write(df.describe(include='all').transpose().astype(str))  # Convert to string for Arrow compatibility

        # Summary Suggestions
        st.markdown("### 🧠 AI Dataset Analysis")
        st.markdown("**Automatic analysis of each column with machine learning suggestions**")
        dataset_summary_and_suggestions(df)

        # Variable Types
        st.markdown("### 🧬 Variable Types")
        st.markdown("**Data types of each column (numerical, categorical, date, etc.)**")
        dtypes_df = pd.DataFrame(df.dtypes, columns=["Type"]).reset_index().rename(columns={"index": "Column"})
        dtypes_df["Type"] = dtypes_df["Type"].astype(str)
        st.dataframe(dtypes_df, use_container_width=True)

        # Missing Values
        st.markdown("### ❌ Missing Values")
        st.markdown("**Detection and visualization of missing data in your dataset**")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing_df = pd.DataFrame(missing, columns=["Missing Values"]).reset_index().rename(columns={"index": "Column"})
            missing_df["Missing Values"] = missing_df["Missing Values"].astype(str)
            st.dataframe(missing_df, use_container_width=True)
            fig = px.bar(
                missing_df, x="Column", y="Missing Values", title="Missing Values per Column", text_auto=True,
                color="Column", color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.exploration_figures.append(fig)
        else:
            st.success("No missing values detected.")

        # Categorical Distribution
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            st.markdown("### 📊 Distribution of Categorical Variables")
            st.markdown("**Visualization of the frequency of each category in your text variables**")
            selected_cat = st.selectbox("Choose a categorical variable", cat_cols, key='eda_cat')
            vc_df = df[selected_cat].value_counts().reset_index()
            vc_df.columns = [selected_cat, 'count']
            vc_df['count'] = vc_df['count'].astype(str)
            fig = px.bar(
                vc_df, x=selected_cat, y='count', text_auto=True, title=f"Distribution of {selected_cat}",
                color=selected_cat, color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.exploration_figures.append(fig)

        # Numerical Distributions
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            st.subheader("📈 Distribution of Numerical Variables")
            st.markdown("**Histogram and box plot to understand the distribution of your numerical data**")
            selected_col = st.selectbox("Select a numerical variable for distribution", numeric_cols)
            fig = px.histogram(df, x=selected_col, nbins=30, marginal="box", title=f"Distribution of {selected_col}")
            st.plotly_chart(fig, use_container_width=True)
            st.session_state.exploration_figures.append(fig)

        # Correlation Matrix
        if len(numeric_cols) > 1:
            st.subheader("📊 Correlation Matrix")
            st.markdown("**Heatmap showing relationships between your numerical variables (-1 to +1)**")
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix, text_auto=True, color_continuous_scale='Viridis', title="Correlation Heatmap"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            st.session_state.exploration_figures.append(fig_corr)

            st.markdown("### 🔗 Scatter Matrix (Pairplot)")
            st.markdown("**Scatter plots to see relationships between pairs of variables**")
            max_pairplot_vars = 5
            pairplot_cols = st.multiselect("Select variables (max 5)", numeric_cols, default=numeric_cols[:max_pairplot_vars])
            if 1 < len(pairplot_cols) <= max_pairplot_vars:
                fig_pair = px.scatter_matrix(
                    df[pairplot_cols], dimensions=pairplot_cols, height=600,
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                st.plotly_chart(fig_pair, use_container_width=True)
                st.session_state.exploration_figures.append(fig_pair)

        # Boxplots
        if len(numeric_cols) > 0:
            st.markdown("### 🧊 Box Plots (Outlier Detection)")
            st.markdown("**Visualization of outliers and distribution of your numerical variables**")
            fig_box_multi = go.Figure()
            for i, col in enumerate(numeric_cols):
                fig_box_multi.add_trace(go.Box(y=df[col], name=col, boxpoints="outliers", marker_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]))
            fig_box_multi.update_layout(title="Boxplots of Numerical Variables", yaxis_title="Values", showlegend=False)
            st.plotly_chart(fig_box_multi, use_container_width=True)
            st.session_state.exploration_figures.append(fig_box_multi)

    elif st.session_state.page == "Training and Evaluation":
        st.title("🧪 Training and Evaluation")
        st.markdown("### 🎯 Model Selection")
        st.info("💡 **Please use the sidebar to select your machine learning model and configure its hyperparameters.**")

        # Model selection in sidebar
        st.sidebar.markdown("---")
        st.sidebar.header("3️⃣ Model Selection")
        model_choice = st.sidebar.selectbox(
            "Choose a Machine Learning Model",
            ["Random Forest", "SVM", "XGBoost", "KNN", "Decision Tree", "Linear Regression", "KMeans"],
            index=0
        )

        st.sidebar.header("4️⃣ Hyperparameters")
        params = {}
        if model_choice == "Random Forest":
            params["n_estimators"] = st.sidebar.slider("Number of Trees", 10, 300, 100, step=10)
            params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
        elif model_choice == "SVM":
            params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
            params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"], index=0)
        elif model_choice == "XGBoost":
            params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10, 3)
            params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1)
        elif model_choice == "KNN":
            params["n_neighbors"] = st.sidebar.slider("Number of Neighbors", 1, 20, 5)
            params["weights"] = st.sidebar.selectbox("Weights", ["uniform", "distance"])
        elif model_choice == "Decision Tree":
            params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
            params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
        elif model_choice == "Linear Regression":
            st.sidebar.markdown("_No hyperparameters to tune for Linear Regression_")
        elif model_choice == "KMeans":
            params["n_clusters"] = st.sidebar.slider("Number of Clusters", 2, 10, 3)
            params["init"] = st.sidebar.selectbox("Init Method", ["k-means++", "random"])
            params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 500, 300)

        # Exclude invalid target columns
        excluded_cols = [col for col in df.columns if col.lower() in ['id', 'index'] or pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col])]
        target_options = [col for col in df.columns if col not in excluded_cols]
        
        if not target_options:
            st.error("No valid column found to use as target.")
            st.stop()

        # Suggest target based on model type
        suggested_target = suggest_target(df, model_choice)
        target_column = None
        if model_choice != "KMeans":
            target_column = st.selectbox(
                "🎯 Select Target Column",
                target_options,
                index=target_options.index(suggested_target) if suggested_target in target_options else 0,
                help="For classification models, choose a column with discrete values (e.g., categories, integers). For regression, choose a column with continuous values."
            )

            # Validate target column compatibility
            if target_column:
                is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
                is_discrete = df[target_column].nunique() <= 20 or (is_numeric and df[target_column].dropna().apply(float.is_integer).all())
                
                if model_choice in ["Random Forest", "SVM", "XGBoost", "KNN", "Decision Tree"] and not is_discrete:
                    st.error(f"Error: The selected target '{target_column}' contains continuous values, but {model_choice} expects discrete classes. Please select a categorical or integer column for classification.")
                    st.stop()
                elif model_choice == "Linear Regression" and is_discrete:
                    st.warning(f"Warning: The selected target '{target_column}' contains discrete values, but Linear Regression expects continuous values. Consider choosing a numerical column with continuous values.")
                
                # Show class distribution
                st.markdown("### 🏷️ Target Class Distribution")
                st.markdown("**Distribution of your target variable to understand class balance**")
                target_counts = df[target_column].value_counts().reset_index()
                target_counts.columns = [target_column, 'Count']
                target_counts['Count'] = target_counts['Count'].astype(str)
                fig_target = px.bar(
                    target_counts,
                    x=target_column,
                    y="Count",
                    title=f"Class Distribution of Target: {target_column}",
                    text_auto=True,
                    color=target_column
                )
                st.plotly_chart(fig_target, use_container_width=True)

        # Configuration display
        st.subheader("⚙️ Model Configuration")
        st.markdown("**Parameters chosen for your machine learning model**")
        st.json(params)

        # Training trigger
        run = st.button("🚀 Start Training", use_container_width=True)
        if run:
            try:
                df_clean = clean_data(df)

                # Prepare features
                feature_cols = [
                    col for col in df_clean.columns
                    if (model_choice != "KMeans" and col != target_column)
                    and col.lower() not in ['id', 'index']
                    and not pd.api.types.is_datetime64_any_dtype(df_clean[col])
                    and not pd.api.types.is_timedelta64_dtype(df_clean[col])
                ]
                features = df_clean[feature_cols]
                features_encoded, _ = encode_categoricals(features)

                # Encode target (if not KMeans)
                if model_choice != "KMeans":
                    target_series = df_clean[target_column]
                    if target_series.dtype == 'object' or str(target_series.dtype).startswith('category'):
                        target_encoded, _ = encode_categoricals(target_series.to_frame())
                        df_encoded = features_encoded.copy()
                        df_encoded[target_column] = target_encoded[target_column]
                    else:
                        df_encoded = features_encoded.copy()
                        df_encoded[target_column] = target_series
                else:
                    df_encoded = features_encoded

                # Split & train
                X_train, X_test, y_train, y_test = split_data(df_encoded, target_column if model_choice != "KMeans" else None)
                model = train_model(model_choice, X_train, y_train if model_choice != "KMeans" else None, params)
                results = evaluate_model(model, X_test, y_test if model_choice != "KMeans" else None)

                figures_to_save = []

                if model_choice == "KMeans":
                    st.subheader("📌 Clustering Results")
                    st.markdown("**Groups created by the K-Means algorithm to segment your data**")
                    st.write(results["labels"].astype(str))
                    pca = PCA(n_components=2)
                    X_test_2d = pca.fit_transform(X_test)
                    pca_df = pd.DataFrame(X_test_2d, columns=["PC1", "PC2"])
                    pca_df["Cluster"] = results["labels"].astype(str)
                    fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="PCA Projection of Clusters", color_discrete_sequence=px.colors.qualitative.Set2)
                    st.subheader("📊 2D Visualization of Clusters")
                    st.markdown("**2D visualization of clusters created by K-Means using PCA dimensionality reduction**")
                    st.plotly_chart(fig_pca, use_container_width=True)
                    figures_to_save.append(fig_pca)

                elif model_choice == "Linear Regression":
                    st.markdown("**Evaluation metrics for linear regression**")
                    st.metric("MSE", f"{results['mse']:.4f}")
                    st.metric("R²", f"{results['r2']:.4f}")

                    fig = px.scatter(x=y_test, y=results["y_pred"], 
                                     labels={'x': 'True Values', 'y': 'Predicted Values'},
                                     title="True vs Predicted",
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                                  line=dict(color='black', dash='dash'))
                    st.markdown("**Comparison of predicted vs actual values (closer to the line = better model)**")
                    st.plotly_chart(fig, use_container_width=True)
                    figures_to_save.append(fig)

                    st.subheader("Error Distribution")
                    st.markdown("**Distribution of prediction errors to evaluate model quality**")
                    errors = y_test - results["y_pred"]
                    fig_err = px.histogram(errors, nbins=30, marginal="box", title="Distribution of Prediction Errors", color_discrete_sequence=px.colors.qualitative.Set2)
                    st.plotly_chart(fig_err, use_container_width=True)
                    figures_to_save.append(fig_err)

                else:
                    st.markdown("**Evaluation metrics for classification**")
                    st.metric("🎯 Accuracy", f"{results['accuracy']:.2%}")
                    st.metric("📈 F1 Score", f"{results['f1_score']:.2%}")

                    st.subheader("📉 Confusion Matrix")
                    st.markdown("**Table showing correct vs incorrect predictions for each class**")
                    fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix")
                    st.plotly_chart(fig_cm, use_container_width=True)
                    figures_to_save.append(fig_cm)

                    if results["y_proba"] is not None and len(np.unique(y_test)) == 2:
                        fpr, tpr, _ = roc_curve(y_test, results["y_proba"])
                        roc_auc = auc(fpr, tpr)
                        fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.2f})",
                                          labels=dict(x="False Positive Rate", y="True Positive Rate"),
                                          color_discrete_sequence=px.colors.qualitative.Set2)
                        fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
                        st.markdown("**ROC curve showing the model's ability to distinguish classes (AUC > 0.5 = good model)**")
                        st.plotly_chart(fig_roc, use_container_width=True)
                        figures_to_save.append(fig_roc)

                    if model_choice in ["Random Forest", "XGBoost", "Decision Tree"] and hasattr(model, "feature_importances_"):
                        st.subheader("🌟 Feature Importance")
                        st.markdown("**Most important variables for the model's predictions**")
                        importances = model.feature_importances_
                        features = X_train.columns
                        imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                        imp_df["Importance"] = imp_df["Importance"].astype(str)
                        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Feature Importance", color="Importance", color_continuous_scale='Viridis')
                        st.plotly_chart(fig_imp, use_container_width=True)
                        figures_to_save.append(fig_imp)

                    elif model_choice == "SVM" and params.get("kernel") == "linear":
                        st.subheader("🌟 SVM Coefficients")
                        st.markdown("**Coefficients of the linear SVM model showing the importance of each variable**")
                        coefs = model.coef_.flatten()
                        coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": coefs}).sort_values(by="Coefficient", ascending=False)
                        coef_df["Coefficient"] = coef_df["Coefficient"].astype(str)
                        fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="SVM Coefficients", color="Coefficient", color_continuous_scale='RdBu')
                        st.plotly_chart(fig_coef, use_container_width=True)
                        figures_to_save.append(fig_coef)

                # Inclure aussi les figures d'exploration dans le PDF
                figures_to_save = st.session_state.exploration_figures + figures_to_save

                # PDF Report Download
                st.markdown("**📥 Download a complete PDF report with all results**")
                pdf_path = generate_pdf_report_reportlab(df, model_choice, params, target_column, results, figures_to_save)
                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download PDF Report", data=f.read(), file_name="ml_report.pdf", mime="application/pdf")
                os.unlink(pdf_path)

            except Exception as e:
                st.error(f"Error during model training: {str(e)}")

else:
    st.info("👈 Load a CSV file to get started.")

st.markdown("---")
st.caption("Développé avec ❤️ et Streamlit | [GitHub](https://github.com/)")
