import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, silhouette_score
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Customer Behavior Analysis & Prediction System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Modern Styling & Custom CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #7928ca 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.12);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    
    .main-header p {
        color: #e2e8f0;
        font-size: 1.05rem;
        margin: 0;
    }
    
    .stat-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 18px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.09);
    }
    
    .stat-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stat-value {
        color: #0f172a;
        font-size: 1.7rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .insight-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 16px 18px;
        border-radius: 0 12px 12px 0;
        margin: 12px 0;
    }
    
    .persona-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 6px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 18px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🛍️ AI-Based Customer Behavior Analysis & Prediction System</h1>
    <p>Empowering retail intelligence through Preprocessing, Exploratory Data Analysis, PCA Dimensionality Reduction, K-Means Clustering, and Decision Tree Classification.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions: Data Loading & Preprocessing
# ---------------------------------------------------------
@st.cache_data
def get_default_dataset():
    default_path = os.path.join(os.path.dirname(__file__), "data", "Mall_Customers.csv")
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    else:
        # Fallback inline generation if file missing
        np.random.seed(42)
        n = 200
        genders = np.random.choice(['Male', 'Female'], n)
        ages = np.random.randint(18, 70, n)
        incomes = np.random.randint(15, 138, n)
        scores = np.random.randint(1, 100, n)
        return pd.DataFrame({
            'CustomerID': range(1, n + 1),
            'Genre': genders,
            'Age': ages,
            'Annual Income (k$)': incomes,
            'Spending Score (1-100)': scores
        })

def standardize_columns(df):
    """Clean and normalize column headers for easy access"""
    rename_dict = {}
    for col in df.columns:
        clean = col.strip()
        if 'customer' in clean.lower() and 'id' in clean.lower():
            rename_dict[col] = 'CustomerID'
        elif 'gen' in clean.lower() or 'sex' in clean.lower():
            rename_dict[col] = 'Gender'
        elif 'age' in clean.lower():
            rename_dict[col] = 'Age'
        elif 'income' in clean.lower():
            rename_dict[col] = 'Annual Income (k$)'
        elif 'spend' in clean.lower() or 'score' in clean.lower():
            rename_dict[col] = 'Spending Score (1-100)'
    return df.rename(columns=rename_dict)

def preprocess_dataframe(df):
    """Handle missing values, encode categoricals, extract numeric features"""
    df_clean = df.copy()
    
    # Missing values check & handle
    missing_info = df_clean.isnull().sum()
    if df_clean.isnull().values.any():
        for col in df_clean.columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
                
    # Encode Gender
    le_gender = LabelEncoder()
    if 'Gender' in df_clean.columns:
        df_clean['Gender_Encoded'] = le_gender.fit_transform(df_clean['Gender'].astype(str))
    else:
        df_clean['Gender_Encoded'] = 0
        
    return df_clean, missing_info, le_gender

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/shop.png", width=110)
    st.title("⚙️ Control Panel")
    
    st.subheader("📂 1. Dataset Selection")
    data_source = st.radio(
        "Choose Dataset Source:",
        ["Mall Customers (Default)", "Upload Custom CSV"],
        index=0
    )
    
    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        
    st.markdown("---")
    st.subheader("🎯 2. Clustering Settings")
    k_clusters = st.slider("Number of Clusters (K)", min_value=2, max_value=10, value=5, step=1)
    
    st.markdown("---")
    st.subheader("🌲 3. Decision Tree Settings")
    dt_criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"], index=0)
    dt_max_depth = st.slider("Max Depth", min_value=2, max_value=15, value=5, step=1)
    test_ratio = st.slider("Test Split Ratio", min_value=0.1, max_value=0.4, value=0.2, step=0.05)
    
    st.markdown("---")
    st.info("💡 **Tip**: Use the tabs on the right to navigate between Data Preprocessing, EDA, PCA, K-Means Clustering, Decision Tree, and Live Predictions!")

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
if data_source == "Upload Custom CSV" and uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.sidebar.success("Custom dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading custom CSV: {e}")
        raw_df = get_default_dataset()
else:
    raw_df = get_default_dataset()

# Standardize and preprocess
df_std = standardize_columns(raw_df)
df, missing_summary, le_gender = preprocess_dataframe(df_std)

# Essential feature verification
required_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
if not all(col in df.columns for col in required_cols):
    st.error(f"The dataset must contain the following columns (or similar): {required_cols}")
    st.stop()

# Primary Clustering Features (Annual Income & Spending Score)
X_cluster = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Full Feature Matrix for PCA & Decision Tree
feature_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']
X_full = df[feature_cols].values

# ---------------------------------------------------------
# Machine Learning Pipeline Execution
# ---------------------------------------------------------
# 1. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

# 2. PCA
pca = PCA(n_components=2, random_state=42)
pca_transformed = pca.fit_transform(X_scaled)
df['PCA_1'] = pca_transformed[:, 0]
df['PCA_2'] = pca_transformed[:, 1]
var_explained = pca.explained_variance_ratio_

# 3. K-Means Clustering
kmeans = KMeans(n_clusters=k_clusters, init='k-means++', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_cluster)
centroids = kmeans.cluster_centers_
sil_score = silhouette_score(X_cluster, df['Cluster'])

# Cluster Persona Definition Helper
def get_cluster_persona(cluster_id, cluster_df):
    mean_inc = cluster_df[cluster_df['Cluster'] == cluster_id]['Annual Income (k$)'].mean()
    mean_spend = cluster_df[cluster_df['Cluster'] == cluster_id]['Spending Score (1-100)'].mean()
    
    if mean_inc >= 65 and mean_spend >= 60:
        return "⭐ VIP Target Customers", "#10b981", "High Income, High Spending", "Offer premium loyalty rewards, exclusive product previews, VIP concierge."
    elif mean_inc >= 65 and mean_spend < 40:
        return "💎 Conservative Wealth", "#3b82f6", "High Income, Low Spending", "Target with high-value investments, quality-driven promos, institutional trust."
    elif mean_inc < 45 and mean_spend >= 60:
        return "🔥 Trendsetters / Impulse", "#f59e0b", "Low Income, High Spending", "Offer flash sales, trendy fast fashion, installment & buy-now-pay-later deals."
    elif mean_inc < 45 and mean_spend < 40:
        return "🛡️ Budget Savers", "#ef4444", "Low Income, Low Spending", "Promote essential discounts, value packs, clearance deals, coupon incentives."
    else:
        return "⚖️ Balanced Shoppers", "#8b5cf6", "Moderate Income, Moderate Spending", "Standard marketing campaigns, seasonal holiday discounts, general memberships."

# 4. Decision Tree Classifier
X_dt = df[feature_cols]
y_dt = df['Cluster']
X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=test_ratio, random_state=42, stratify=y_dt)

dt_model = DecisionTreeClassifier(criterion=dt_criterion, max_depth=dt_max_depth, random_state=42)
dt_model.fit(X_train, y_train)

y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

# ---------------------------------------------------------
# UI Layout: Main Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 1. Overview & Preprocessing",
    "📈 2. Exploratory Data Analysis (EDA)",
    "🔬 3. PCA Dimensionality Reduction",
    "🎯 4. K-Means Customer Clustering",
    "🌲 5. Decision Tree Classification",
    "🔮 6. Live Customer Predictor"
])

# =========================================================
# TAB 1: Dataset Overview & Preprocessing
# =========================================================
with tabs[0]:
    st.subheader("📋 Dataset Overview & Preprocessing Summary")
    
    # Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Total Customers</div>
            <div class="stat-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Dataset Features</div>
            <div class="stat-value">{df_std.shape[1]}</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        missing_cnt = missing_summary.sum()
        status_color = "#10b981" if missing_cnt == 0 else "#ef4444"
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Missing Values</div>
            <div class="stat-value" style="color: {status_color}">{missing_cnt}</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Configured Clusters (K)</div>
            <div class="stat-value">{k_clusters}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Two Column Layout for Table & Checks
    col_l, col_r = st.columns([3, 2])
    
    with col_l:
        st.markdown("##### 🔍 Cleaned Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        csv_download = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Processed Dataset (CSV)",
            data=csv_download,
            file_name="processed_mall_customers.csv",
            mime="text/csv"
        )
        
    with col_r:
        st.markdown("##### 🛠️ Data Preprocessing & Missing Values Check")
        missing_df = pd.DataFrame({
            'Column': missing_summary.index,
            'Missing Count': missing_summary.values,
            'Data Type': [df_std[c].dtype for c in missing_summary.index]
        })
        st.dataframe(missing_df, use_container_width=True)
        
        st.markdown("""
        <div class="insight-card">
            <strong>Preprocessing Actions Executed:</strong><br>
            • Standardized variable naming conventions.<br>
            • Verified null & missing value status (handled via median/mode imputation).<br>
            • Transformed categorical feature <code>Gender</code> to numerical encoding (<code>Gender_Encoded</code>).<br>
            • Applied feature scaling for dimensional transformations.
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("##### 📊 Descriptive Statistics")
    st.dataframe(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].describe().T, use_container_width=True)

# =========================================================
# TAB 2: Exploratory Data Analysis (EDA)
# =========================================================
with tabs[1]:
    st.subheader("📈 Exploratory Data Analysis (EDA)")
    
    # Row 1: Feature Distributions
    st.markdown("##### 1. Distribution of Key Numerical Attributes")
    d1, d2, d3 = st.columns(3)
    
    with d1:
        fig_age = px.histogram(
            df, x="Age", nbins=20, marginal="box",
            title="Age Distribution",
            color_discrete_sequence=["#3b82f6"]
        )
        fig_age.update_layout(template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_age, use_container_width=True)
        
    with d2:
        fig_inc = px.histogram(
            df, x="Annual Income (k$)", nbins=20, marginal="box",
            title="Annual Income Distribution (k$)",
            color_discrete_sequence=["#10b981"]
        )
        fig_inc.update_layout(template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_inc, use_container_width=True)
        
    with d3:
        fig_spend = px.histogram(
            df, x="Spending Score (1-100)", nbins=20, marginal="box",
            title="Spending Score Distribution (1-100)",
            color_discrete_sequence=["#f59e0b"]
        )
        fig_spend.update_layout(template="plotly_white", margin=dict(t=40, b=20, l=20, r=20))
        st.plotly_chart(fig_spend, use_container_width=True)
        
    st.markdown("---")
    
    # Row 2: Annual Income vs Spending Score & Gender Breakdown
    st.markdown("##### 2. Core Feature Correlations & Demographics")
    eda_col1, eda_col2 = st.columns([3, 2])
    
    with eda_col1:
        fig_scatter = px.scatter(
            df, x="Annual Income (k$)", y="Spending Score (1-100)",
            color="Gender" if "Gender" in df.columns else None,
            size="Age",
            hover_data=["CustomerID", "Age"],
            title="Annual Income vs Spending Score (Bubble Size = Age)",
            color_discrete_map={"Male": "#3b82f6", "Female": "#ec4899"}
        )
        fig_scatter.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with eda_col2:
        if 'Gender' in df.columns:
            fig_gender = px.pie(
                df, names='Gender',
                title="Gender Composition Ratio",
                color='Gender',
                color_discrete_map={"Male": "#3b82f6", "Female": "#ec4899"},
                hole=0.45
            )
            fig_gender.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.info("Gender column not available for breakdown.")

    st.markdown("---")
    
    # Row 3: Correlation Heatmap
    st.markdown("##### 3. Correlation Heatmap")
    heat_col1, heat_col2 = st.columns([2, 2])
    
    with heat_col1:
        corr_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Encoded']
        corr_matrix = df[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5, ax=ax, cbar=True)
        plt.title("Pearson Correlation Heatmap")
        st.pyplot(fig)
        
    with heat_col2:
        st.markdown("""
        <div class="insight-card">
            <h4>💡 Key Exploratory Takeaways:</h4>
            <ul>
                <li><strong>No direct linear correlation:</strong> Income vs Spending Score shows distinct non-linear clusters rather than a simple diagonal trend.</li>
                <li><strong>Age Dynamics:</strong> Younger demographics (18-35) tend to have higher spending variance, frequently occupying the high-spending bands.</li>
                <li><strong>Natural Clusters:</strong> Five intuitive customer groups visually stand out in the Income vs Spending Score space.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 3: PCA (Principal Component Analysis)
# =========================================================
with tabs[2]:
    st.subheader("🔬 Principal Component Analysis (PCA)")
    
    st.markdown("""
    <div class="insight-card">
        <strong>What is PCA?</strong> Principal Component Analysis projects multidimensional features 
        (<em>Age, Annual Income, Spending Score, Gender</em>) into orthogonal principal axes that maximize variance, 
        enabling clear 2D visualization while retaining core structural relationships.
    </div>
    """, unsafe_allow_html=True)
    
    pca_c1, pca_c2 = st.columns([3, 2])
    
    with pca_c1:
        fig_pca = px.scatter(
            df, x='PCA_1', y='PCA_2',
            color='Spending Score (1-100)',
            hover_data=['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
            title=f"2D PCA Projection (Explained Variance: PC1={var_explained[0]*100:.1f}%, PC2={var_explained[1]*100:.1f}%)",
            color_continuous_scale="Plasma"
        )
        fig_pca.update_layout(template="plotly_white", height=450)
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with pca_c2:
        st.markdown("##### 📊 Variance Explained")
        pca_var_df = pd.DataFrame({
            'Component': ['Principal Component 1 (PC1)', 'Principal Component 2 (PC2)'],
            'Explained Variance (%)': [f"{var_explained[0]*100:.2f}%", f"{var_explained[1]*100:.2f}%"],
            'Cumulative Variance (%)': [f"{var_explained[0]*100:.2f}%", f"{sum(var_explained[:2])*100:.2f}%"]
        })
        st.dataframe(pca_var_df, use_container_width=True)
        
        # Loadings Heatmap
        st.markdown("##### 🧬 Feature Loadings on Components")
        loadings_df = pd.DataFrame(
            pca.components_,
            columns=feature_cols,
            index=['PC1', 'PC2']
        )
        fig_load, ax_load = plt.subplots(figsize=(6, 2.5))
        sns.heatmap(loadings_df, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax_load)
        st.pyplot(fig_load)

# =========================================================
# TAB 4: K-Means Customer Clustering
# =========================================================
with tabs[3]:
    st.subheader("🎯 K-Means Customer Segmentation")
    
    # Row 1: Elbow Method & Silhouette Analysis
    st.markdown("##### 1. Optimal Cluster Detection: Elbow Method & Silhouette Curve")
    k_eval_col1, k_eval_col2 = st.columns(2)
    
    wcss = []
    sil_scores_list = []
    k_range = range(2, 11)
    
    # Calculate for elbow
    for k_val in range(1, 11):
        km = KMeans(n_clusters=k_val, init='k-means++', random_state=42, n_init=10)
        km.fit(X_cluster)
        wcss.append(km.inertia_)
        
    for k_val in k_range:
        km = KMeans(n_clusters=k_val, init='k-means++', random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster)
        sil_scores_list.append(silhouette_score(X_cluster, labels))
        
    with k_eval_col1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(x=list(range(1, 11)), y=wcss, mode='lines+markers', name='WCSS', line=dict(color='#3b82f6', width=3)))
        # Highlight selected K
        fig_elbow.add_trace(go.Scatter(x=[k_clusters], y=[wcss[k_clusters-1]], mode='markers', marker=dict(size=14, color='#ef4444'), name=f'Current K={k_clusters}'))
        fig_elbow.update_layout(title="Elbow Method (WCSS vs K)", xaxis_title="Number of Clusters K", yaxis_title="WCSS (Inertia)", template="plotly_white", height=350)
        st.plotly_chart(fig_elbow, use_container_width=True)
        
    with k_eval_col2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(x=list(k_range), y=sil_scores_list, mode='lines+markers', name='Silhouette Score', line=dict(color='#10b981', width=3)))
        if k_clusters >= 2:
            fig_sil.add_trace(go.Scatter(x=[k_clusters], y=[sil_scores_list[k_clusters-2]], mode='markers', marker=dict(size=14, color='#ef4444'), name=f'Current K={k_clusters}'))
        fig_sil.update_layout(title="Silhouette Score Analysis vs K", xaxis_title="Number of Clusters K", yaxis_title="Silhouette Score", template="plotly_white", height=350)
        st.plotly_chart(fig_sil, use_container_width=True)
        
    st.markdown("---")
    
    # Row 2: 2D Cluster Visualizations
    st.markdown(f"##### 2. Customer Clusters Visualization (K = {k_clusters})")
    clust_col1, clust_col2 = st.columns([3, 2])
    
    with clust_col1:
        # Create interactive Plotly Cluster Scatter
        df['Cluster_Label'] = df['Cluster'].apply(lambda x: f"Cluster {x}")
        fig_clusters = px.scatter(
            df, x="Annual Income (k$)", y="Spending Score (1-100)",
            color="Cluster_Label",
            hover_data=["CustomerID", "Age", "Gender" if "Gender" in df.columns else "Cluster"],
            title="Interactive Customer Clusters (Annual Income vs Spending Score)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        # Add Centroids
        fig_clusters.add_trace(go.Scatter(
            x=centroids[:, 0],
            y=centroids[:, 1],
            mode='markers+text',
            text=[f"C{i}" for i in range(len(centroids))],
            textposition="top center",
            marker=dict(size=16, color='black', symbol='diamond-cross', line=dict(width=2, color='white')),
            name='Centroids'
        ))
        fig_clusters.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_clusters, use_container_width=True)
        
    with clust_col2:
        # Matplotlib / Seaborn publication-ready chart
        st.markdown("##### 🖼️ Matplotlib Cluster Plot")
        fig_mat, ax_mat = plt.subplots(figsize=(6, 4.8))
        palette = sns.color_palette("bright", k_clusters)
        
        for i in range(k_clusters):
            cluster_data = df[df['Cluster'] == i]
            ax_mat.scatter(
                cluster_data['Annual Income (k$)'], 
                cluster_data['Spending Score (1-100)'],
                s=50, 
                color=palette[i], 
                label=f'Cluster {i}'
            )
            
        ax_mat.scatter(
            centroids[:, 0], centroids[:, 1], 
            s=180, c='black', marker='X', edgecolors='white', linewidths=1.5, label='Centroids'
        )
        ax_mat.set_title(f'K-Means Clusters (K={k_clusters})', fontsize=12, fontweight='bold')
        ax_mat.set_xlabel('Annual Income (k$)')
        ax_mat.set_ylabel('Spending Score (1-100)')
        ax_mat.legend(loc='upper right', fontsize=8)
        ax_mat.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig_mat)

    # 3D Cluster Visualization & PCA Cluster Space
    st.markdown("##### 3. 3D Cluster Space & PCA Cluster Mapping")
    c3d_1, c3d_2 = st.columns(2)
    
    with c3d_1:
        fig_3d = px.scatter_3d(
            df, x='Age', y='Annual Income (k$)', z='Spending Score (1-100)',
            color='Cluster_Label',
            hover_data=['CustomerID', 'Gender'] if 'Gender' in df.columns else ['CustomerID'],
            title="3D Customer Clusters (Age × Income × Spending)",
            color_discrete_sequence=px.colors.qualitative.Bold,
            opacity=0.85
        )
        fig_3d.update_layout(template="plotly_white", height=430, margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with c3d_2:
        fig_pca_clust = px.scatter(
            df, x='PCA_1', y='PCA_2',
            color='Cluster_Label',
            hover_data=['CustomerID', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'],
            title="K-Means Clusters Projected on 2D PCA Space",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pca_clust.update_layout(template="plotly_white", height=430)
        st.plotly_chart(fig_pca_clust, use_container_width=True)

    st.markdown("---")
    
    # Row 4: Cluster Personas & Targeted Marketing Profiles
    st.markdown("##### 4. Cluster Profiling & Strategic Business Personas")
    
    persona_cards = []
    for c_id in range(k_clusters):
        c_sub = df[df['Cluster'] == c_id]
        p_title, p_color, p_desc, p_action = get_cluster_persona(c_id, df)
        count = len(c_sub)
        pct = (count / len(df)) * 100
        avg_age = c_sub['Age'].mean()
        avg_inc = c_sub['Annual Income (k$)'].mean()
        avg_spn = c_sub['Spending Score (1-100)'].mean()
        
        persona_cards.append({
            "Cluster": f"Cluster {c_id}",
            "Persona": p_title,
            "Behavior": p_desc,
            "Customer Count": f"{count} ({pct:.1f}%)",
            "Avg Age": f"{avg_age:.1f} yrs",
            "Avg Income": f"${avg_inc:.1f}k",
            "Avg Spending": f"{avg_spn:.1f}/100",
            "Recommended Strategy": p_action
        })
        
    persona_df = pd.DataFrame(persona_cards)
    st.dataframe(persona_df, use_container_width=True)

# =========================================================
# TAB 5: Decision Tree Classification
# =========================================================
with tabs[4]:
    st.subheader("🌲 Decision Tree Classification")
    
    st.markdown("""
    <div class="insight-card">
        <strong>Supervised Classification:</strong> A Decision Tree is trained using the customer features 
        (<em>Age, Annual Income, Spending Score, Gender</em>) to predict customer cluster membership.
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Training Accuracy</div>
            <div class="stat-value" style="color: #10b981">{train_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Test Accuracy</div>
            <div class="stat-value" style="color: #3b82f6">{test_acc*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Criterion</div>
            <div class="stat-value" style="font-size: 1.3rem">{dt_criterion.capitalize()}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-title">Max Tree Depth</div>
            <div class="stat-value">{dt_max_depth}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    dt_col1, dt_col2 = st.columns([3, 2])
    
    with dt_col1:
        st.markdown("##### 🔲 Confusion Matrix (Test Set)")
        cm = confusion_matrix(y_test, y_pred_test)
        
        fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                    xticklabels=[f"C{i}" for i in range(k_clusters)],
                    yticklabels=[f"C{i}" for i in range(k_clusters)])
        ax_cm.set_xlabel("Predicted Cluster Label")
        ax_cm.set_ylabel("True Cluster Label")
        ax_cm.set_title("Test Confusion Matrix Heatmap")
        st.pyplot(fig_cm)
        
    with dt_col2:
        st.markdown("##### 📊 Feature Importance")
        importances = dt_model.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)
        
        fig_fi = px.bar(
            fi_df, x='Importance', y='Feature', orientation='h',
            title="Feature Importance in Cluster Prediction",
            color='Importance', color_continuous_scale='Viridis'
        )
        fig_fi.update_layout(template="plotly_white", height=320)
        st.plotly_chart(fig_fi, use_container_width=True)
        
    st.markdown("---")
    
    # Classification Report
    st.markdown("##### 📑 Detailed Classification Report")
    report_dict = classification_report(y_test, y_pred_test, output_dict=True)
    report_df = pd.DataFrame(report_dict).T
    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
    
    with st.expander("🔍 View Decision Tree Decision Logic (Text Rules)"):
        tree_rules = export_text(dt_model, feature_names=feature_cols)
        st.code(tree_rules, language="text")

# =========================================================
# TAB 6: Live Customer Segment Predictor
# =========================================================
with tabs[5]:
    st.subheader("🔮 Real-Time Customer Segment Predictor")
    
    st.markdown("""
    <div class="insight-card">
        <strong>Instant AI Profiling:</strong> Enter customer attributes below to predict their behavioral cluster segment, 
        view their business persona, and see their real-time position mapped against existing mall customers!
    </div>
    """, unsafe_allow_html=True)
    
    input_c1, input_c2 = st.columns([2, 3])
    
    with input_c1:
        st.markdown("##### 📝 Input Customer Parameters")
        with st.form("customer_prediction_form"):
            in_gender = st.selectbox("Gender", ["Female", "Male"], index=0)
            in_age = st.slider("Customer Age", min_value=18, max_value=85, value=30, step=1)
            in_income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=70, step=1)
            in_spend = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=75, step=1)
            
            predict_btn = st.form_submit_button("🚀 Predict Customer Segment", use_container_width=True)
            
    with input_c2:
        if predict_btn or 'prediction_done' in st.session_state:
            st.session_state['prediction_done'] = True
            
            # Encode input
            gender_encoded_val = 1 if in_gender == "Male" else 0
            input_df = pd.DataFrame([[in_age, in_income, in_spend, gender_encoded_val]], columns=feature_cols)
            
            # Model prediction
            pred_cluster = dt_model.predict(input_df)[0]
            persona_title, persona_color, persona_desc, persona_action = get_cluster_persona(pred_cluster, df)
            
            st.markdown(f"""
            <div style="background: white; border: 2px solid {persona_color}; border-radius: 16px; padding: 22px; box-shadow: 0 10px 25px rgba(0,0,0,0.08); margin-bottom: 20px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.3rem; font-weight: 700; color: #0f172a;">Assigned: Cluster {pred_cluster}</span>
                    <span style="background: {persona_color}; color: white; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.9rem;">
                        {persona_title}
                    </span>
                </div>
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #e2e8f0;">
                <p style="margin: 6px 0; color: #334155;"><strong>Behavioral Profile:</strong> {persona_desc}</p>
                <p style="margin: 6px 0; color: #334155;"><strong>Input Summary:</strong> {in_gender}, {in_age} yrs old | Income: ${in_income}k | Spending Score: {in_spend}/100</p>
                <div style="background: #f8fafc; border-left: 4px solid {persona_color}; padding: 10px 14px; margin-top: 10px; border-radius: 0 8px 8px 0;">
                    <strong style="color: #0f172a;">🎯 Recommended Action:</strong> {persona_action}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Scatter Plot highlighting the new customer point
            st.markdown("##### 📍 Real-Time Spatial Positioning")
            fig_live = px.scatter(
                df, x="Annual Income (k$)", y="Spending Score (1-100)",
                color="Cluster_Label",
                title="Customer Segment Space (⭐ = New Predicted Customer)",
                color_discrete_sequence=px.colors.qualitative.Bold,
                opacity=0.65
            )
            
            # Highlight New Customer
            fig_live.add_trace(go.Scatter(
                x=[in_income],
                y=[in_spend],
                mode='markers+text',
                marker=dict(size=24, color='#fbbf24', symbol='star', line=dict(width=2.5, color='black')),
                text=["⭐ NEW CUSTOMER"],
                textposition="top center",
                name="New Customer"
            ))
            
            fig_live.update_layout(template="plotly_white", height=420)
            st.plotly_chart(fig_live, use_container_width=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; border: 2px dashed #cbd5e1; border-radius: 16px; padding: 40px; text-align: center; color: #64748b;">
                <img src="https://img.icons8.com/clouds/100/000000/user-male-circle.png" width="80" style="margin-bottom: 12px;"/>
                <h4>Awaiting Customer Parameters</h4>
                <p>Adjust the sliders on the left and click <strong>"🚀 Predict Customer Segment"</strong> to view real-time behavioral insights.</p>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 10px 0;">
    AI-Based Customer Behavior Analysis & Prediction System | Built with Streamlit, Scikit-Learn, Pandas & Plotly
</div>
""", unsafe_allow_html=True)
