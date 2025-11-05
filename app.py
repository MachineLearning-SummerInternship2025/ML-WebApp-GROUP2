import os
import time
import base64
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import mimetypes
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc, confusion_matrix, mean_squared_error, r2_score, precision_recall_curve
from sklearn.decomposition import PCA
from preprocess import clean_data, encode_categoricals, split_data, detect_task_type, select_features, handle_imbalance, simple_automl
try:
    from preprocess import advanced_preprocessing_pipeline, impute_missing, remove_outliers_zscore, winsorize_outliers, scale_data, encode_categoricals_advanced, detect_useless_columns
except ImportError:
    def advanced_preprocessing_pipeline(df, **kwargs):
        return df
from ml_engine import train_model, evaluate_model
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile

# --- Page Config (MUST be first Streamlit command) ---
st.set_page_config(
    page_title="ML Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)



# --- helper: build a data URI for any image type ---
def file_to_data_uri(path: Path) -> str:
    if not path.exists():
        st.error(f"Image not found: {path}")
        return ""
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"  # fallback
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

# --- resolve image path robustly ---
app_dir = Path(__file__).parent if "__file__" in globals() else Path.cwd()
upload_image_path = app_dir / "images" / "Upload.png"
graph_image_path = app_dir /"images" / "Graph.png"
train_image_path = app_dir /"images" / "Train.png"
Kaggle_image_path = app_dir / "images" / "Kaggle.png"
UCI_image_path = app_dir /"images" / "UCI.png"
HuggingFace_image_path = app_dir /"images" / "HuggingFace.png"

# quick sanity check (remove after testing)
# st.write("Using image path:", upload_image_path)

data_uri = file_to_data_uri(upload_image_path)
graph_image_uri = file_to_data_uri(graph_image_path)
train_image_uri = file_to_data_uri(train_image_path)
Kaggle_image_uri = file_to_data_uri(Kaggle_image_path)
UCI_image_uri = file_to_data_uri(UCI_image_path)
HuggingFace_image_uri = file_to_data_uri(HuggingFace_image_path)






def img_data_uri(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""  # will render as empty if file missing
    ext = p.suffix.lower()
    if ext in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif ext == ".svg":
        mime = "image/svg+xml"
    else:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"



#font injection 
def inject_local_font(font_path: str, font_family: str, fmt: str = "truetype"):
    font_bytes = Path(font_path).read_bytes()
    b64 = base64.b64encode(font_bytes).decode("utf-8")
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: '{font_family}';
            src: url(data:font/{'woff2' if fmt=='woff2' else 'ttf'};base64,{b64}) format('{fmt}');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def load_css(path: str):
    with open(path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# 👇 inject your font + load your stylesheet
inject_local_font("fonts/Buka Bird.ttf", "BukaBird", fmt="truetype")
# Inject Cocogoose Pro font variants
inject_local_font("fonts/Cocogoose-Pro-Bold-trial.ttf", "Cocogoose-Bold", fmt="truetype")
inject_local_font("fonts/Cocogoose-Pro-Regular-trial.ttf", "Cocogoose-Regular", fmt="truetype")
inject_local_font("fonts/Cocogoose-Pro-Light-trial.ttf", "Cocogoose-Light", fmt="truetype")
inject_local_font("fonts/Cocogoose-Pro-Semilight-trial.ttf", "Cocogoose-Semilight", fmt="truetype")
inject_local_font("fonts/Cocogoose-Pro-Thin-trial.ttf", "Cocogoose-Thin", fmt="truetype")
inject_local_font("fonts/Cocogoose-Pro-Ultralight-trial.ttf", "Cocogoose-Ultralight", fmt="truetype")
inject_local_font("fonts/Cocogoose-Pro-Italic-trial.ttf", "Cocogoose-Italic", fmt="truetype")
inject_local_font("fonts/Cocogoose-Pro-Darkmode-trial.ttf", "Cocogoose-Darkmode", fmt="truetype")
inject_local_font("fonts/MADE TOMMY Regular_PERSONAL USE.otf", "Tommy-Regular", fmt="opentype")
inject_local_font("fonts/MADE TOMMY Thin_PERSONAL USE.otf", "Tommy-Thin", fmt="opentype")
inject_local_font("fonts/MADE TOMMY Black_PERSONAL USE.otf", "Tommy-Black", fmt="opentype")
inject_local_font("fonts/MADE TOMMY Bold_PERSONAL USE.otf", "Tommy-Bold", fmt="opentype")
inject_local_font("fonts/MADE TOMMY ExtraBold_PERSONAL USE.otf", "Tommy-ExtraBold", fmt="opentype")
inject_local_font("fonts/MADE TOMMY Light_PERSONAL USE.otf", "Tommy-Light", fmt="opentype")
inject_local_font("fonts/MADE TOMMY Medium_PERSONAL USE.otf", "Tommy-Medium", fmt="opentype")
inject_local_font("fonts/Playfull Rocket.ttf", "Playfull-Rocket", fmt="opentype")
inject_local_font("fonts/Happy School.ttf", "Happy-School", fmt="opentype")

load_css("styles.css")

# --- NOUVELLES IMPORTS SÉCURITÉ ET CLOUD ---
try:
    from security import get_security_manager, secure_session_state, check_session_timeout, require_authentication
    from config import get_config
    SECURITY_AVAILABLE = True
    print("✅ Module de sécurité chargé avec succès")
except ImportError as e:
    print(f"⚠️ Module de sécurité non disponible: {e}")
    SECURITY_AVAILABLE = False

# --- IMPORT DROPBOX CLOUD ---
try:
    from cloud_dropbox import DropboxCloudStorage
    CLOUD_AVAILABLE = True
    print("✅ Module Dropbox chargé avec succès")
except ImportError as e:
    print(f"⚠️ Module Dropbox non disponible: {e}")
    CLOUD_AVAILABLE = False

# --- SQLAlchemy imports et config ---
from sqlalchemy import create_engine, Column, String, Integer, LargeBinary, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
engine = create_engine('sqlite:///users.db', connect_args={"check_same_thread": False})
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# --- Modèles ORM ---
class User(Base):
    __tablename__ = 'users'
    username = Column(String, primary_key=True)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    datasets = relationship('Dataset', back_populates='user', cascade="all, delete-orphan")
    user_data = relationship('UserData', back_populates='user', cascade="all, delete-orphan")

class Dataset(Base):
    __tablename__ = 'datasets'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, ForeignKey('users.username'))
    name = Column(String, nullable=False)
    content = Column(LargeBinary, nullable=False)  # Peut contenir un lien S3 (bytes)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship('User', back_populates='datasets')
    user_data = relationship('UserData', back_populates='dataset', cascade="all, delete-orphan")

class UserData(Base):
    __tablename__ = 'user_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, ForeignKey('users.username'))
    dataset_id = Column(Integer, ForeignKey('datasets.id'))
    data_type = Column(String)
    data_content = Column(LargeBinary)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    user = relationship('User', back_populates='user_data')
    dataset = relationship('Dataset', back_populates='user_data')

# Créer les tables si besoin
Base.metadata.create_all(engine)
# --- Fin modèles ORM ---
import sqlite3
import bcrypt
import textwrap
import uuid
import requests
import joblib
from sklearn.preprocessing import LabelEncoder

# --- Page Config moved to top of file ---

# --- Styling ---
st.markdown("""
<style>
.main {background-color: #f8fafc;}
.stButton>button {background-color: #2563eb; color: white;}
.stSidebar {background-color: #f1f5f9;}
</style>
""", unsafe_allow_html=True)

# --- Load External CSS ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Initialize Session State ---
# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'about'

# Initialize authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.name = None

# Initialize signup state
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False

# Initialize exploration figures
if "exploration_figures" not in st.session_state:
    st.session_state.exploration_figures = []

# Hide sidebar until the user is authenticated
if (st.session_state.current_page == 'about') and (not st.session_state.get('authenticated', False)):
    st.markdown(
        """
        <style>
        section[data-testid='stSidebar'],
        div[data-testid='stSidebar'],
        [data-testid='stSidebarNav'],
        div[data-testid='collapsedControl'] { display: none !important; visibility: hidden !important; width: 0 !important; min-width: 0 !important; }
        /* Remove reserved sidebar space so content uses full width */
        div.block-container { margin-left: 0 !important; }
        /* In some Streamlit versions, the main container uses a grid; ensure full width */
        [data-testid='stAppViewContainer'] > .main { width: 100% !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# --- Authentication Function (must be defined before use) ---
def authenticate_user(username, password):
    if SECURITY_AVAILABLE:
        security = get_security_manager()
        
        # Vérifier les tentatives de connexion
        can_login, attempts, rate_limit = security.check_login_attempts(username)
        if not can_login:
            if rate_limit > 0:
                return False, f"Rate limit dépassé. Réessayez plus tard."
            else:
                remaining_lock = security.get_remaining_lock_time(username)
                wait_secs = remaining_lock if remaining_lock > 0 else security.login_timeout
                return False, f"Trop de tentatives. Réessayez dans {wait_secs} secondes."
        
        # Vérifier le mot de passe
        user = session.query(User).filter_by(username=username).first()
        if user and security.verify_password(password, user.password):
            # Connexion réussie
            security.record_login_attempt(username, True)
            security.record_request(username)
            return True, user.name
        else:
            # Connexion échouée
            security.record_login_attempt(username, False)
            remaining_lock = security.get_remaining_lock_time(username)
            remaining_attempts = security.get_remaining_attempts(username)
            if remaining_lock > 0:
                return False, f"Trop de tentatives. Réessayez dans {remaining_lock} secondes."
            return False, f"Nom d'utilisateur ou mot de passe incorrect. Tentatives restantes: {remaining_attempts}"
    else:
        # Mode sécurité standard
        user = session.query(User).filter_by(username=username).first()
        if user and bcrypt.checkpw(password.encode('utf-8'), user.password.encode('utf-8')):
            return True, user.name
        return False, None



# Navigation bar - Show different buttons based on authentication status
if st.session_state.authenticated:
    # After login: Show About, Dashboard, and Logout
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 5])
    with col1:
        about_active = st.session_state.current_page == 'about'
        if st.button("About", key="nav_about_auth", help="Go to About page", 
                     type="primary" if about_active else "secondary"):
            st.session_state.current_page = 'about'
            st.rerun()
    with col2:
        dashboard_active = st.session_state.current_page == 'dashboard'
        if st.button("Dashboard", key="nav_dashboard_auth", help="Go to Dashboard",
                     type="primary" if dashboard_active else "secondary"):
            st.session_state.current_page = 'dashboard'
            st.rerun()
    with col3:
        if st.button("Logout", key="nav_logout", help="Logout from account", type="secondary"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.name = None
            st.rerun()
else:
    # Before login: Show About and Get Started
    col1, col2, col3, col4 = st.columns([1, 1, 1, 6])
    with col1:
        about_active = st.session_state.current_page == 'about'
        if st.button("About", key="nav_about", help="Go to About page", 
                     type="primary" if about_active else "secondary"):
            st.session_state.current_page = 'about'
            st.rerun()
    with col2:
        get_started_active = st.session_state.current_page == 'get_started'
        if st.button("Get Started", key="nav_get_started", help="Go to Get Started page",
                     type="primary" if get_started_active else "secondary"):
            st.session_state.current_page = 'get_started'
            st.rerun()

# Add minimal spacing after navbar
st.markdown("<br>", unsafe_allow_html=True)

# --- Page Content Based on Navigation ---
if st.session_state.current_page == 'about':
    # --- About Page (Landing Page) ---
    # Add left + right margins
    left_space, content, right_space = st.columns([0.2, 1.6, 0.2])

    with content:
        col1, col2 = st.columns([1, 1])  # equal space for text + image

        with col1:
            st.markdown(
                """
                <div style="padding-left:40px; max-width:600px;">
                    <div style="font-size: 70px; font-family:'Happy-School'; color:#764f72; margin-bottom: 5px;">
                        ML Explorer
                    </div>
                    <div style="font-size: 26px; font-family:'Tommy-Medium'; margin-bottom: 18px; color:#000000;">
                        Learn Machine Learning by Doing.
                    </div>
                    <div style="font-size: 17px; font-family:'Tommy-Medium'; line-height: 1.5; margin-bottom: 12px; color:#000000;">
                        Our website is an interactive platform where you can upload datasets, explore data, and train machine learning models—all without writing a single line of code.
                    </div>
                    <div style="font-size: 17px; font-family:'Tommy-Medium'; line-height: 1.5; margin-bottom: 20px; color:#000000;">
                        It combines hands-on practice with simple explanations, visualizations, and
                        makes Machine Learning easy and fun to learn.
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                "<div style='display:flex; align-items:center; height:100%;'>",
                unsafe_allow_html=True,
            )
            st.image("images/myphoto.png", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- "How do I use it?" + Steps Section ---
    st.markdown(
        """
<div style="margin-top:80px; text-align:center; font-size:40px; font-family:'Tommy-Bold'; color:#111731; margin-bottom:20px;">
    How do I use it?
</div>
<div style="text-align:center; font-size:18px; font-family:'Tommy-Light'; color:#000000; margin-bottom:50px; max-width:600px; margin-left:auto; margin-right:auto; line-height:1.5;">
    Get started with ML Explorer in just three simple steps. No coding required, 
    just upload your data and let our intelligent platform guide you through the process.
</div>
""",
        unsafe_allow_html=True,
    )

    import os

    # Get the absolute path to the images
    current_dir = os.path.dirname(os.path.abspath(__file__))
    upload_image_path = os.path.join(current_dir, "images/Upload.png")
    graph_image_path = os.path.join(current_dir, "images/Graph.png")
    train_image_path = os.path.join(current_dir, "images/Train.png")




    # Step 1: Upload Dataset
    spacer1, col1, col2, col3, spacer2 = st.columns([0.5, 2, 2, 2, 0.5])  # keeps everything centered

    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:white; border-radius:16px; box-shadow:0 8px 32px rgba(17,23,49,0.08); border:1px solid rgba(17,23,49,0.05); transition:all 0.3s ease;">
                <img src="{data_uri}" alt="Upload Dataset"
                 style="width:200px; height:auto; border-radius:12px;
                         margin-bottom:15px;" />
                <div style="font-family:'Tommy-Bold'; font-size:18px; color:#111731; margin-bottom:8px;">
                    Upload Dataset
                </div>
                <div style="font-family:'Tommy-Medium'; font-size:16px; color:#292B4C; line-height:1.4;">
                    Drag & drop your data and start instantly - no setup needed.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:white; border-radius:16px; box-shadow:0 8px 32px rgba(17,23,49,0.08); border:1px solid rgba(17,23,49,0.05); transition:all 0.3s ease;">
                <img src="{graph_image_uri}" alt="Explore Data" 
                 style="width:200px; height:auto; border-radius:12px; margin-bottom:15px;" />
                <div style="font-family:'Tommy-Bold'; font-size:18px; color:#111731; margin-bottom:8px;">
                    Data Exploration
                </div>
                <div style="font-family:'Tommy-Medium'; font-size:16px; color:#292B4C; line-height:1.4;">
                    Understand your dataset with interactive graphs and visual insights.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:white; border-radius:16px; box-shadow:0 8px 32px rgba(17,23,49,0.08); border:1px solid rgba(17,23,49,0.05); transition:all 0.3s ease;">
                <img src="{train_image_uri}" alt="Train Model" 
                 style="width:200px; height:auto; border-radius:12px; margin-bottom:15px;" />
                <div style="font-family:'Tommy-Bold'; font-size:18px; color:#111731; margin-bottom:8px;">
                    Model Training
                </div>
                <div style="font-family:'Tommy-Medium'; font-size:16px; color:#292B4C; line-height:1.4;">
                    Train ML models and see performance results with clear visualizations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



    # --- "From where can I download free datasets ?"---
    st.markdown(
        """
<div style="margin-top:80px; text-align:center; font-size:40px; font-family:'Tommy-Bold'; color:#111731; margin-bottom:20px;">
    From where can I download free datasets ?
</div>
<div style="text-align:center; font-size:18px; font-family:'Tommy-Light'; color:#000000; margin-bottom:50px; max-width:600px; margin-left:auto; margin-right:auto; line-height:1.5;">
    Finding quality datasets is the first step in any Machine Learning project. 
    Here are some of the best free resources where you can explore and download datasets to start practicing right away.
</div>
""",
        unsafe_allow_html=True,
    )




    # Step 1: Upload Dataset
    spacer1, col1, col2, col3, spacer2 = st.columns([0.5, 2, 2, 2, 0.5])  # keeps everything centered

    with col1:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:white; border-radius:16px; box-shadow:0 8px 32px rgba(17,23,49,0.08); border:1px solid rgba(17,23,49,0.05); transition:all 0.3s ease;">
                <img src="{Kaggle_image_uri}" alt="Kaggle"
                 style="width:200px; height:125px; border-radius:12px;
                         margin-bottom:15px;" />
                <div style="margin-bottom:8px;">
    <a href="https://www.kaggle.com/datasets" target="_blank" 
       style="font-family:'Tommy-Bold'; font-size:18px; color:#111731; text-decoration:none; border-bottom:2px solid #111731; padding-bottom:1px;">
        Kaggle
    </a>
</div>
                
            
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:white; border-radius:16px; box-shadow:0 8px 32px rgba(17,23,49,0.08); border:1px solid rgba(17,23,49,0.05); transition:all 0.3s ease;">
                <img src="{UCI_image_uri}" alt="UCI ML Repository" 
                 style="width:200px; height:125px; border-radius:12px; margin-bottom:15px;" />
                <div style="margin-bottom:8px;">
    <a href="https://archive.ics.uci.edu/ml/index.php" target="_blank" 
       style="font-family:'Tommy-Bold'; font-size:18px; color:#111731; 
              text-decoration:none; border-bottom:2px solid #111731; padding-bottom:2px;">
        UCI ML Repository
    </a>
</div>

                
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="text-align:center; padding:20px; background:white; border-radius:16px; box-shadow:0 8px 32px rgba(17,23,49,0.08); border:1px solid rgba(17,23,49,0.05); transition:all 0.3s ease;">
                <img src="{HuggingFace_image_uri}" alt="Hugging Face" 
                 style="width:200px; height:125px; border-radius:12px; margin-bottom:15px;" />
                <div style="margin-bottom:8px;">
    <a href="https://huggingface.co/datasets" target="_blank" 
       style="font-family:'Tommy-Bold'; font-size:18px; color:#111731; 
              text-decoration:none; border-bottom:2px solid #111731; padding-bottom:2px;">
        Hugging Face
    </a>
</div>

            """,
            unsafe_allow_html=True,
        )

elif st.session_state.current_page == 'get_started':
    # --- Get Started Page (Login/Signup) ---
    st.markdown('<h1 class="get-started-title">Get Started</h1>', unsafe_allow_html=True)
    st.markdown('<p class="get-started-subtitle"style="color:#000000;">Welcome! Please login or create an account to start using ML Explorer.</p>', unsafe_allow_html=True)
    
    # Login and Sign-Up Forms
    if not st.session_state.show_signup:
        with st.form(key='login_form'):
            st.markdown('<h2 class="get-started-form-title">Login</h2>', unsafe_allow_html=True)
            username = st.text_input("Username", key='login_username')
            password = st.text_input("Password", type="password", key='login_password')
            # CAPTCHA
            if SECURITY_AVAILABLE:
                security = get_security_manager()
                captcha_q = security.ensure_captcha()
                st.caption("Veuillez résoudre le CAPTCHA pour continuer")
                captcha_answer = st.text_input(captcha_q, key='login_captcha_answer')
            submit = st.form_submit_button("Login")
            if submit:
                if SECURITY_AVAILABLE:
                    security = get_security_manager()
                    username = security.sanitize_input(username)
                    # Vérifier CAPTCHA avant toute authentification
                    provided_captcha = (captcha_answer if 'captcha_answer' in locals() else st.session_state.get('login_captcha_answer', ''))
                    if not security.verify_captcha(str(provided_captcha).strip()):
                        st.error("CAPTCHA invalide ou expiré. Réessayez.")
                        security.rotate_captcha()
                        st.rerun()
                authenticated, name = authenticate_user(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.name = name
                    st.session_state.current_page = 'dashboard'  # Redirect to dashboard after login
                    st.success(f"Welcome back, {name}!")
                    st.rerun()
                else:
                    # Afficher des informations dynamiques de sécurité
                    if SECURITY_AVAILABLE:
                        security = get_security_manager()
                        # Sauvegarder le dernier username tenté pour mise à jour sidebar
                        st.session_state['last_login_username'] = username
                        remaining_lock = security.get_remaining_lock_time(username)
                        remaining_attempts = security.get_remaining_attempts(username)
                        if remaining_lock > 0:
                            st.error(f"Trop de tentatives. Réessayez dans {remaining_lock}s.")
                        else:
                            st.error(f"Identifiants incorrects. Tentatives restantes: {remaining_attempts}")
                        # Petit délai anti brute-force (progressif)
                        current_fail_count = security.max_login_attempts - remaining_attempts
                        backoff = min(max(0.0, current_fail_count) * 0.4, 3.0)
                        if backoff > 0:
                            time.sleep(backoff)
                        # Rafraîchir le CAPTCHA après un échec
                        security.rotate_captcha()
                    else:
                        st.error("Invalid username or password")
            
            st.markdown("---")
            st.markdown("Don't have an account?")
            if st.form_submit_button("Create Account"):
                st.session_state.show_signup = True
                st.rerun()
    else:
        with st.form(key='signup_form'):
            st.markdown('<h2 class="get-started-form-title">📝 Create Account</h2>', unsafe_allow_html=True)
            name = st.text_input("Full Name")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Sign Up")
            if submit:
                if not name or not username or not password:
                    st.error("All fields are required.")
                else:
                    # Validation du mot de passe si la sécurité est disponible
                    if SECURITY_AVAILABLE:
                        security = get_security_manager()
                        is_valid, message = security.validate_password_strength(password)
                        if not is_valid:
                            st.error(f"Password too weak: {message}")
                            st.stop()
                    
                    existing_user = session.query(User).filter_by(username=username).first()
                    if existing_user:
                        st.error("This username already exists.")
                    else:
                        # Hash du mot de passe avec la méthode appropriée
                        if SECURITY_AVAILABLE:
                            hashed_pw = security.hash_password(password)
                        else:
                            hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                        
                        new_user = User(username=username, password=hashed_pw, name=name)
                        session.add(new_user)
                        session.commit()
                        st.success("Account created successfully! Please login.")
                        st.session_state.show_signup = False
                        st.rerun()
            
            st.markdown("---")
            st.markdown("Already have an account?")
            if st.form_submit_button("Login"):
                st.session_state.show_signup = False
                st.rerun()

# --- Initialisation de la sécurité ---
if SECURITY_AVAILABLE:
    secure_session_state()
    st.sidebar.success("🔒 Sécurité renforcée activée")
else:
    st.sidebar.warning("⚠️ Mode sécurité standard")

# --- Affichage des informations cloud ---
if CLOUD_AVAILABLE:
    try:
        # Fonction pour obtenir l'instance Dropbox
        def get_dropbox_storage():
            """Retourne l'instance Dropbox si configurée"""
            access_token = os.getenv('DROPBOX_ACCESS_TOKEN')
            if (access_token):
                try:
                    return DropboxCloudStorage(access_token)
                except Exception as e:
                    return None
            return None
        
        dropbox = get_dropbox_storage()
        if dropbox:
            try:
                storage_info = dropbox.get_storage_info()
                if storage_info.get("storage_type") == "dropbox":
                    st.sidebar.success(f"☁️ Dropbox: {storage_info.get('total_files', 0)} fichiers")
                else:
                    st.sidebar.info("💾 Stockage local")
            except Exception as e:
                st.sidebar.info("💾 Stockage local")
        else:
            st.sidebar.info("💾 Stockage local")
    except Exception as e:
        st.sidebar.info("💾 Stockage local")
else:
    st.sidebar.info("💾 Stockage local uniquement")

# --- Database Setup ---
def init_db():
    # Crée les tables si besoin (déjà fait par Base.metadata.create_all(engine))
    # Ajoute des utilisateurs par défaut si la table est vide
    if session.query(User).count() == 0:
        import bcrypt
        users = [
            User(username="user1", password=bcrypt.hashpw("password123".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'), name="User One"),
            User(username="user2", password=bcrypt.hashpw("welcome2023".encode('utf-8'), bcrypt.gensalt()).decode('utf-8'), name="User Two")
        ]
        session.add_all(users)
        session.commit()

init_db()

# Ajout de la colonne dataset_id si elle n'existe pas déjà
# (inutile avec SQLAlchemy, la structure est gérée par les modèles)





# --- Main Application Logic ---
if not st.session_state.authenticated:
    # Show configuration info in sidebar for non-authenticated users
    with st.sidebar.expander("⚙️ Configuration & État", expanded=False):
        if SECURITY_AVAILABLE:
            st.success("🔒 Sécurité renforcée")
            security = get_security_manager()
            # Valeurs de configuration
            st.info(f"Timeout: {security.login_timeout}s")
            st.info(f"Mot de passe min: {security.password_min_length} caractères")
            # Infos dynamiques selon l'utilisateur saisi ou dernière tentative
            typed_username = st.session_state.get('login_username', '').strip()
            if not typed_username:
                typed_username = st.session_state.get('last_login_username', '')
            if typed_username:
                remaining_attempts = security.get_remaining_attempts(typed_username)
                remaining_lock = security.get_remaining_lock_time(typed_username)
                st.info(f"Tentatives restantes: {remaining_attempts}/{security.max_login_attempts}")
                if remaining_lock > 0:
                    st.error(f"Compte temporairement bloqué: {remaining_lock}s restants")
            else:
                st.info(f"Tentatives max: {security.max_login_attempts}")
        else:
            st.warning("⚠️ Sécurité standard")
else:
    # User is authenticated - show main dashboard
    st.sidebar.success(f"Welcome, {st.session_state.name}!")

    
    # Show dashboard content
    if st.session_state.current_page == 'dashboard':
        # Welcome message before dashboard title
        st.markdown('<h1 class="dashboard-title">Main Dashboard</h1>', unsafe_allow_html=True)
        st.markdown('<p class="dashboard-subtitle">Welcome to your ML Explorer dashboard! Use the sidebar to navigate through different features.</p>', unsafe_allow_html=True)
        
        # Quick stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Datasets", len(session.query(Dataset).filter_by(username=st.session_state.username).all()))
        with col2:
            st.metric("User", st.session_state.username)
        with col3:
            st.metric("Status", "Active")
        
        st.markdown("---")
        st.markdown('<h3 class="dashboard-quick-actions">Quick Actions</h3>', unsafe_allow_html=True)
        st.markdown("""
        • Upload a new dataset using the sidebar
        • Explore your existing datasets
        • Train machine learning models
        • Generate reports and visualizations
        """)

def call_ai_analysis_api(df):
    """Appelle l'API d'analyse AI si disponible, sinon utilise l'analyse locale"""
    try:
        import io
        csv_bytes = df.to_csv(index=False).encode()
        response = requests.post(
            "http://localhost:5001/analyze-dataset",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
            timeout=10  # Timeout de 10 secondes
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API returned status {response.status_code}")
    except Exception as e:
        # Si l'API n'est pas disponible, utiliser l'analyse locale
        return generate_local_analysis(df)

def generate_local_analysis(df):
    """Génère une analyse locale du dataset sans API externe"""
    summary_data = []
    useful_features = []
    cols_to_drop = []
    
    for col in df.columns:
        dtype = df[col].dtype
        uniques = df[col].nunique()
        null_rate = df[col].isnull().mean()
        examples = df[col].dropna().unique()[:3]
        example_vals = ", ".join(map(str, examples)) + ("..." if uniques > 3 else "")
        
        # Déterminer l'utilité ML
        if uniques == 1:
            ml_utility = "🚫 Constant (to ignore)"
            cols_to_drop.append(col)
        elif null_rate > 0.5:
            ml_utility = "⚠️ Too many missing values"
        elif col.lower() in ['id', 'index'] or df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
            ml_utility = "🔁 Identifier / index"
            cols_to_drop.append(col)
        else:
            ml_utility = "✅ Potentially useful"
            useful_features.append(col)
        
        # Déterminer le type
        if dtype == 'object':
            description = f"Categorical ({uniques} categories)"
        elif pd.api.types.is_numeric_dtype(df[col]):
            description = f"Numerical ({df[col].min()} to {df[col].max()})"
        elif 'date' in col.lower():
            description = "Temporal (date format)"
        else:
            description = f"Type {dtype}"
        
        summary_data.append({
            "📌 Column": f"`{col}`",
            "🧠 Type": str(dtype),
            "🔢 Uniques": uniques,
            "💬 Example": example_vals,
            "✍️ Description": description,
            "📊 ML Utility": ml_utility
        })
    
    return {
        "summary": summary_data,
        "useful_features": useful_features,
        "cols_to_drop": cols_to_drop,
        "analysis_type": "local"
    }

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
    classification_models = ["Random Forest", "SVM", "XGBoost", "KNN", "Decision Tree", "Logistic Regression"]
    regression_models = ["Linear Regression", "XGBoost Regressor"]
    clustering_models = ["KMeans", "DBSCAN"]

    if model_choice in classification_models:
        candidates = [
            col for col in df.columns
            if (df[col].nunique() <= 20 and df[col].dtype in ['object', 'int64', 'category'])
            or (df[col].dtype == 'float64' and df[col].dropna().apply(float.is_integer).all())
        ]
    elif model_choice in regression_models:
        candidates = [
            col for col in df.columns
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20
        ]
    elif model_choice in clustering_models:
        return None
    else:
        candidates = df.columns.tolist()

    return candidates[-1] if candidates else df.columns[-1]

# --- Générateur de description pour chaque figure ---
def describe_figure(fig_title):
    fig_title = fig_title or ""
    if "Distribution" in fig_title:
        return "Ce graphique montre la répartition des valeurs d'une variable. Il permet de repérer les tendances, les valeurs extrêmes et les déséquilibres éventuels dans les données."
    elif "Boxplots" in fig_title or "Box Plots" in fig_title:
        return "Le boxplot visualise la dispersion, la médiane et les valeurs atypiques (outliers) d'une variable numérique. Il aide à détecter les anomalies et à comparer les distributions."
    elif "Correlation" in fig_title or "Heatmap" in fig_title:
        return "La heatmap de corrélation met en évidence les relations linéaires entre variables numériques. Les couleurs indiquent la force et le sens de la corrélation."
    elif "Scatter Matrix" in fig_title or "Pairplot" in fig_title:
        return "Le scatter matrix (pairplot) permet de visualiser les relations croisées entre plusieurs variables numériques sous forme de nuages de points."
    elif "Feature Importance" in fig_title:
        return "Ce graphique présente l'importance de chaque variable pour le modèle. Les variables les plus importantes influencent le plus la prédiction."
    elif "Confusion Matrix" in fig_title:
        return "La matrice de confusion compare les prédictions du modèle aux valeurs réelles. Elle permet d'évaluer la qualité de la classification."
    elif "ROC Curve" in fig_title:
        return "La courbe ROC illustre la capacité du modèle à distinguer les classes. Plus la courbe est proche du coin supérieur gauche, meilleure est la performance. L'AUC résume cette performance."
    elif "PCA Projection" in fig_title or "2D Visualization of Clusters" in fig_title:
        return "Cette projection 2D (PCA) permet de visualiser la séparation des groupes (clusters) obtenus par l'algorithme de clustering."
    elif "Precision-Recall" in fig_title or "PR Curve" in fig_title:
        return "La courbe Precision-Recall (PR) montre le compromis entre la précision et le rappel pour différents seuils. Elle est particulièrement utile pour évaluer les modèles sur des jeux de données déséquilibrés. Plus la courbe est proche du coin supérieur droit, meilleure est la performance."
    elif "Residual Plot" in fig_title:
        return "Le graphique des résidus montre la différence entre les valeurs réelles et prédites en fonction des valeurs prédites. Un bon modèle doit présenter des résidus répartis aléatoirement autour de zéro, sans motif particulier, ce qui indique que les erreurs sont aléatoires et non systématiques."
    else:
        return "Graphique généré automatiquement pour l'analyse exploratoire ou l'évaluation du modèle."

# --- PDF Report Generator ---
def generate_pdf_report_reportlab(df, model_choice, params, target_column, results, figures_to_save=[]):
    from reportlab.lib import colors
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    width, height = letter
    y = height - 40

    blue_dark = colors.HexColor("#2563eb")
    blue_light = colors.HexColor("#60a5fa")
    gray_box = colors.HexColor("#f1f5f9")
    white = colors.white
    black = colors.black

    c.setFillColor(blue_dark)
    c.rect(0, y-20, width, 50, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, y, "Automatic Report - Machine Learning")
    y -= 60

    c.setFillColor(blue_light)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, f"Dataset : {df.shape[0]} rows, {df.shape[1]} columns")
    y -= 20
    c.drawString(50, y, f"Selected model : {model_choice}")
    y -= 20
    if target_column:
        c.drawString(50, y, f"Target column : {target_column}")
    y -= 30

    c.setStrokeColor(blue_dark)
    c.setLineWidth(2)
    c.line(40, y, width-40, y)
    y -= 20

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

    c.setFillColor(gray_box)
    c.roundRect(40, y-60, width-80, 60, 8, fill=1, stroke=0)
    c.setFillColor(blue_dark)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "Model results :")
    y -= 20
    c.setFillColor(black)
    c.setFont("Helvetica", 12)
    if model_choice.lower() == "linear regression":
        mse_val = results.get('mse')
        r2_val = results.get('r2')
        c.drawString(70, y, f"MSE : {mse_val:.4f}" if mse_val is not None else "MSE : N/A")
        y -= 15
        c.drawString(70, y, f"R2 Score : {r2_val:.4f}" if r2_val is not None else "R2 Score : N/A")
        y -= 30
    elif model_choice.lower() == "kmeans":
        c.drawString(70, y, "Clustering completed (no accuracy/F1 for KMeans)")
        y -= 30
    elif model_choice.lower() == "dbscan":
        c.drawString(70, y, "Clustering completed (no accuracy/F1 for DBSCAN)")
        y -= 30
    else:
        # Handle None values for classification metrics
        accuracy_val = results.get('accuracy')
        f1_val = results.get('f1_score')
        recall_val = results.get('recall')
        precision_val = results.get('precision')
        auc_val = results.get('auc_roc')
        
        c.drawString(70, y, f"Accuracy : {accuracy_val:.2%}" if accuracy_val is not None else "Accuracy : N/A")
        y -= 15
        c.drawString(70, y, f"F1 Score : {f1_val:.2%}" if f1_val is not None else "F1 Score : N/A")
        y -= 15
        c.drawString(70, y, f"Recall : {recall_val:.2%}" if recall_val is not None else "Recall : N/A")
        y -= 15
        c.drawString(70, y, f"Precision : {precision_val:.2%}" if precision_val is not None else "Precision : N/A")
        y -= 15
        if auc_val is not None:
            c.drawString(70, y, f"AUC-ROC : {auc_val:.4f}")
            y -= 15
        y -= 15

    c.setStrokeColor(blue_light)
    c.setLineWidth(1.5)
    c.line(40, y, width-40, y)
    y -= 20

    for fig in figures_to_save:
        # Titre du graphe (si disponible)
        fig_title = getattr(fig.layout, 'title', None)
        fig_title_str = str((fig_title.text if fig_title and hasattr(fig_title, 'text') else "Graphique") or "Graphique")
        desc = describe_figure(fig_title_str)
        # Découpage automatique des lignes longues (max 90 caractères)
        desc_lines = []
        for line in desc.split("\n"):
            desc_lines.extend(textwrap.wrap(line, width=90))
        desc_height = 18 + 16 * len(desc_lines) + 10
        # Prévoir la hauteur de l'image (max 300px)
        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        fig.write_image(tmp_img.name, format="png", width=800, height=600, scale=2)
        tmp_img.close()
        img = ImageReader(tmp_img.name)
        img_width, img_height = img.getSize()
        aspect = img_height / img_width
        display_width = width - 100
        display_height = min(display_width * aspect, 300)
        block_height = desc_height + display_height + 40
        # Saut de page si besoin
        if y - block_height < 60:
            c.showPage()
            y = height - 40
        # Titre
        c.setFont("Helvetica-Bold", 12)
        c.setFillColor(blue_dark)
        c.drawString(50, y, fig_title_str)
        y -= 18
        # Description (multiligne, drawText)
        c.setFont("Helvetica", 11)
        c.setFillColor(black)
        text_obj = c.beginText(60, y)
        for line in desc_lines:
            text_obj.textLine(line)
        c.drawText(text_obj)
        y -= 16 * len(desc_lines) + 10
        # Image
        c.drawImage(img, 50, y - display_height, width=display_width, height=display_height)
        y -= (display_height + 30)
        os.unlink(tmp_img.name)

    c.save()
    return tmp_file.name

# --- Sidebar Navigation ---
if "exploration_figures" not in st.session_state:
    st.session_state.exploration_figures = []

# --- Upload et enregistrement du dataset ---
# Remise en local : upload et stockage dans la base, sans S3 ni cloud
if st.session_state.get('authenticated', False):
    st.sidebar.header("1️⃣ Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key=st.session_state.get("file_uploader_key", "default"))
    dataset_id = None
    if uploaded_file is not None:
        if not st.session_state.get('username'):
            st.error("Erreur : utilisateur non authentifié. Veuillez vous reconnecter.")
        else:
            dataset_name = uploaded_file.name
            file_bytes = uploaded_file.read()
            # Recherche d'un doublon (par nom et contenu)
            existing = session.query(Dataset).filter_by(username=st.session_state.username, name=dataset_name, content=file_bytes).first()
            if existing:
                st.warning("Ce dataset existe déjà dans votre espace. Aucun doublon n'a été ajouté.")
                dataset_id = existing.id
                df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
                st.session_state["current_dataset_id"] = dataset_id
                st.session_state["loaded_dataset_name"] = dataset_name
                st.session_state["loaded_df"] = df
                st.session_state["file_uploader_key"] = str(uuid.uuid4())
                st.session_state["last_uploaded_name"] = dataset_name
                st.session_state["last_uploaded_user"] = st.session_state['username']
                st.rerun()
            else:
                new_dataset = Dataset(username=st.session_state.username, name=dataset_name, content=file_bytes)
                session.add(new_dataset)
                session.commit()
                dataset_id = new_dataset.id
                df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
                st.session_state["current_dataset_id"] = dataset_id
                st.session_state["loaded_dataset_name"] = dataset_name
                st.session_state["loaded_df"] = df
                st.success(f"Dataset '{dataset_name}' uploadé et chargé avec succès !")
                st.session_state["file_uploader_key"] = str(uuid.uuid4())
                st.session_state["last_uploaded_name"] = dataset_name
                st.session_state["last_uploaded_user"] = st.session_state['username']
                st.rerun()
    elif "current_dataset_id" in st.session_state:
        dataset_id = st.session_state["current_dataset_id"]
else:
    uploaded_file = None
    dataset_id = None

# --- Section Mes datasets dans la sidebar ---
if st.session_state.get('authenticated', False):
    user_datasets = session.query(Dataset).filter_by(username=st.session_state.username).order_by(Dataset.timestamp.desc()).all()
    if user_datasets:
        st.sidebar.markdown('---')
        st.sidebar.header('📂 Mes datasets')
        search_query = st.sidebar.text_input("🔍 Recherche par nom", value=st.session_state.get('search_dataset', ''), key="search_dataset", placeholder="Tapez pour filtrer...", label_visibility="visible")
        with st.sidebar.expander("⚙️ Filtres avancés", expanded=False):
            min_date = min([ds.timestamp.date() for ds in user_datasets])
            max_date = max([ds.timestamp.date() for ds in user_datasets])
            min_date_obj = pd.to_datetime(min_date)
            max_date_obj = pd.to_datetime(max_date)
            date_range = st.date_input(
                "Date (entre)",
                value=(min_date_obj, max_date_obj),
                min_value=min_date_obj,
                max_value=max_date_obj,
                key="date_filter",
                format="YYYY-MM-DD"
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                st.caption(f"Sélection : {date_range[0].strftime('%d/%m/%Y')} → {date_range[1].strftime('%d/%m/%Y')}")
            min_size = min([len(ds.content) for ds in user_datasets])
            max_size = max([len(ds.content) for ds in user_datasets])
            size_range = st.slider(
                "Taille (Ko)",
                min_value=0,
                max_value=max(1, int(max_size/1024)+1),
                value=(0, max(1, int(max_size/1024)+1)),
                key="size_filter",
                format="%d Ko"
            )
        filtered_datasets = []
        for ds in user_datasets:
            # Filtre nom (dynamique)
            if search_query.lower() not in ds.name.lower():
                continue
            # Filtre date
            date_ok = True
            if isinstance(date_range, tuple) and len(date_range) == 2:
                date_val = pd.to_datetime(ds.timestamp.date())
                date_ok = (date_val >= pd.to_datetime(date_range[0])) and (date_val <= pd.to_datetime(date_range[1]))
            # Filtre taille
            size_ok = (len(ds.content)/1024 >= size_range[0]) and (len(ds.content)/1024 <= size_range[1])
            if date_ok and size_ok:
                filtered_datasets.append(ds)
        if not filtered_datasets:
            st.sidebar.info("Aucun dataset ne correspond à la recherche ou aux filtres avancés.")
        for ds in filtered_datasets:
            date_str = pd.to_datetime(ds.timestamp.date()).strftime('%d/%m/%Y')
            size_str = f"{round(len(ds.content)/1024,1)} Ko"
            st.sidebar.markdown(f"""
                <div style='background:#f1f5f9;border-radius:8px;margin-bottom:10px;padding:10px 8px;'>
                    <b>📄 {ds.name}</b><br>
                    <span style='font-size:12px;color:#888;'>Ajouté le {date_str} | {size_str}</span>
                </div>
            """, unsafe_allow_html=True)
            col1, col2, col3 = st.sidebar.columns([1,1,1])
            with col1:
                if st.button("▶️", key=f"load_{ds.id}", help="Charger ce dataset"):
                    df = pd.read_csv(pd.io.common.BytesIO(ds.content))
                    st.session_state["current_dataset_id"] = ds.id
                    st.session_state["loaded_dataset_name"] = ds.name
                    st.session_state["loaded_df"] = df
                    st.success(f"Dataset '{ds.name}' chargé avec succès !")
                    st.rerun()
            with col2:
                st.download_button("⬇️", data=ds.content, file_name=ds.name, mime="text/csv", key=f"dl_{ds.id}", help="Télécharger ce dataset")
            with col3:
                if st.button("🗑️", key=f"delete_{ds.id}", help="Supprimer ce dataset"):
                    session.query(UserData).filter_by(dataset_id=ds.id).delete()
                    session.delete(ds)
                    session.commit()
                    if st.session_state.get("current_dataset_id") == ds.id:
                        st.session_state.pop("current_dataset_id", None)
                        st.session_state.pop("loaded_dataset_name", None)
                        st.session_state.pop("loaded_df", None)
                    st.success(f"Dataset '{ds.name}' supprimé !")
                    st.rerun()

# --- Chargement du dataset pour l'app principale ---
# Priorité : dataset chargé depuis l'historique, sinon upload
if st.session_state.get("loaded_df") is not None:
    df = st.session_state["loaded_df"]
    dataset_id = st.session_state.get("current_dataset_id")
    uploaded_file = None  # Pour éviter de recharger un nouveau fichier
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # dataset_id déjà mis à jour lors de l'upload
else:
    df = None
    dataset_id = None

# --- Section Dropbox & Sauvegarde ---
if CLOUD_AVAILABLE and st.session_state.get('authenticated', False):
    st.sidebar.markdown("---")
    st.sidebar.header("☁️ Dropbox & Sauvegarde")
    
    # Fonction pour obtenir l'instance Dropbox
    def get_dropbox_storage():
        """Retourne l'instance Dropbox si configurée"""
        access_token = os.getenv('DROPBOX_ACCESS_TOKEN')
        if access_token:
            try:
                return DropboxCloudStorage(access_token)
            except Exception as e:
                st.error(f"Erreur Dropbox: {e}")
                return None
        return None
    
    dropbox = get_dropbox_storage()
    if dropbox:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("💾 Sauvegarder DB"):
                try:
                    success, message = dropbox.backup_database("users.db")
                    if success:
                        st.success("✅ Base sauvegardée sur Dropbox!")
                    else:
                        st.error(f"❌ Erreur: {message}")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
        
        with col2:
            if st.button("🧹 Nettoyer backups"):
                try:
                    success, message = dropbox.cleanup_old_backups(days_to_keep=30)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ Erreur: {message}")
                except Exception as e:
                    st.error(f"❌ Erreur: {e}")
        
        # Afficher les informations Dropbox
        try:
            storage_info = dropbox.get_storage_info()
            st.sidebar.info(f"📁 {storage_info.get('total_files', 0)} fichiers")
            st.sidebar.info(f"💾 {storage_info.get('total_size_mb', 0)} MB")
            st.sidebar.info(f"👤 {storage_info.get('account_name', 'N/A')}")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Erreur info: {e}")
    else:
        st.sidebar.warning("⚠️ Dropbox non configuré")
        st.sidebar.info("💡 Configurez DROPBOX_ACCESS_TOKEN dans votre fichier .env")
        with st.sidebar.expander("🔧 Configuration Dropbox", expanded=False):
            st.code("""
DROPBOX_ACCESS_TOKEN=votre_access_token_ici
DROPBOX_APP_KEY=votre_app_key_ici
DROPBOX_APP_SECRET=votre_app_secret_ici
DROPBOX_FOLDER=/ml-selector
            """)

if st.session_state.get('authenticated', False):
    st.sidebar.markdown("---")
    st.sidebar.header("2️⃣ Navigation")
    page = st.sidebar.radio("Go to:", ["Data Exploration", "Training and Evaluation"])
    st.session_state.page = page

exploration_figures = st.session_state.exploration_figures

# --- Main Application Content (only show on dashboard) ---
if st.session_state.authenticated and st.session_state.current_page == 'dashboard':
    # --- Affichage principal selon dataset chargé ---
    if st.session_state.get("loaded_df") is not None:
        df = st.session_state["loaded_df"]

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
            st.title("Data Exploration")

            # Dataset Preview (dataset original)
            st.subheader("📊 Dataset Preview")
            st.markdown("**Preview of the first few rows of your dataset to understand its structure**")
            st.dataframe(df.head(), use_container_width=True)
            st.markdown(f"**Rows:** {df.shape[0]}   **Columns:** {df.shape[1]}")

            # Statistical Summary
            st.markdown("### 📑 Statistical Summary")
            st.markdown("**Descriptive statistics of your numerical data (mean, standard deviation, min, max, etc.)**")
            st.write(df.describe(include='all').transpose().astype(str))

            # Summary Suggestions
            st.markdown("### 🧠 AI Dataset Analysis")
            st.markdown("**Automatic analysis of each column with machine learning suggestions **")
            if df is not None:
                try:
                    api_result = call_ai_analysis_api(df)
                    summary_df = pd.DataFrame(api_result["summary"])
                    st.dataframe(summary_df, use_container_width=True)
                    useful_cols = summary_df[summary_df["ml_utility"] == "✅ Potentially useful"]["column"].tolist()
                    ignore_cols = summary_df[summary_df["ml_utility"] == "🚫 Constant (to ignore)"]["column"].tolist()
                    st.markdown(f"- **Useful columns for ML :** `{', '.join(useful_cols)}`")
                    if ignore_cols:
                        st.markdown(f"- **Columns to ignore :** `{', '.join(ignore_cols)}`")
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API d'analyse : {e}")

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
                fig = px.bar(missing_df, x="Column", y="Missing Values", title="Missing Values per Column", text_auto=True, color="Column", color_discrete_sequence=px.colors.qualitative.Set2)
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
                fig = px.bar(vc_df, x=selected_cat, y='count', text_auto=True, title=f"Distribution of {selected_cat}", color=selected_cat, color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.exploration_figures.append(fig)

            # Numerical Distributions
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                st.subheader("📈 Distribution of Numerical Variables")
                st.markdown("**Histogram and box plot to understand the distribution of your numerical data**")
                selected_col = st.selectbox("Select a numerical variable for distribution", numeric_cols)
                fig = px.histogram(df, x=selected_col, nbins=30, marginal="box", title=f"Distribution of {selected_col}", color_discrete_sequence=px.colors.qualitative.Set2)
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.exploration_figures.append(fig)

            # Correlation Matrix
            if len(numeric_cols) > 1:
                st.subheader("📊 Correlation Matrix")
                st.markdown("**Heatmap showing relationships between your numerical variables (-1 to +1)**")
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='Viridis', title="Correlation Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)
                st.session_state.exploration_figures.append(fig_corr)

                st.markdown("### 🔗 Scatter Matrix (Pairplot)")
                st.markdown("**Scatter plots to see relationships between pairs of variables**")
                max_pairplot_vars = 5
                pairplot_cols = st.multiselect("Select variables (max 5)", numeric_cols, default=numeric_cols[:max_pairplot_vars])
                if 1 < len(pairplot_cols) <= max_pairplot_vars:
                    fig_pair = px.scatter_matrix(df[pairplot_cols], dimensions=pairplot_cols, height=600, color_discrete_sequence=px.colors.qualitative.Set2)
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

            # --- Prétraitement avancé automatique (moved here, after EDA/graphs) ---
            if df is not None and len(df) > 0:
                st.markdown("---")
                st.info("🔧 **Prétraitement automatique appliqué**")
                with st.spinner("Prétraitement en cours..."):
                    df_processed, _ = advanced_preprocessing_pipeline(
                        df, 
                        impute=True, 
                        outlier_method="winsorize", 
                        scale_method="standard", 
                        encode_method="ordinal", 
                        drop_useless=True
                    )
                # Résumé des traitements appliqués
                col1, col2 = st.columns(2)
                with col1:
                    st.success("✅ **Traitements appliqués :**")
                    st.markdown("• Valeurs manquantes imputées")
                    st.markdown("• Outliers winsorisés")
                    st.markdown("• Variables normalisées (StandardScaler)")
                    st.markdown("• Variables catégorielles encodées")
                    st.markdown("• Colonnes inutiles supprimées")
                with col2:
                    st.info("📊 **Résumé du dataset :**")
                    st.markdown(f"• **Avant :** {df.shape[0]} lignes, {df.shape[1]} colonnes")
                    st.markdown(f"• **Après :** {df_processed.shape[0]} lignes, {df_processed.shape[1]} colonnes")
                    if df.shape[1] != df_processed.shape[1]:
                        dropped_cols = set(df.columns) - set(df_processed.columns)
                        st.markdown(f"• **Colonnes supprimées :** {len(dropped_cols)}")
                        if len(dropped_cols) > 0:
                            st.markdown(f"  - {', '.join(list(dropped_cols)[:5])}{'...' if len(dropped_cols) > 5 else ''}")
                # Dataset Preview (après prétraitement)
                st.subheader("📊 Dataset Preview (after preprocessing)")
                st.markdown("**Preview of the dataset after applying automatic processes**")
                st.dataframe(df_processed.head(), use_container_width=True)
                st.markdown(f"**Rows:** {df_processed.shape[0]}   **Columns:** {df_processed.shape[1]}")

        elif st.session_state.page == "Training and Evaluation":
            st.title("Training and Evaluation")
            st.markdown("### 🎯 Model Selection")
            st.info("💡 **Please use the sidebar to select your machine learning model and configure its hyperparameters.**")

            # Model selection in sidebar
            st.sidebar.markdown("---")
            st.sidebar.header("3️⃣ Model Selection")
            model_choice = st.sidebar.selectbox(
                "Choose a Machine Learning Model",
                ["Random Forest", "SVM", "XGBoost", "KNN", "Decision Tree", "Linear Regression", "Logistic Regression", "XGBoost Regressor", "KMeans", "DBSCAN"],
                index=0
            )
            st.session_state["model_choice"] = model_choice

            st.sidebar.header("4️⃣ Hyperparameters")
            params = {}
            if model_choice == "Random Forest":
                params["n_estimators"] = st.sidebar.slider("Number of Trees", 10, 300, 100, step=10, key="rf_n_estimators")
                params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5, key="rf_max_depth")
            elif model_choice == "SVM":
                params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, key="svm_C")
                params["kernel"] = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly"], index=0, key="svm_kernel")
            elif model_choice == "XGBoost":
                params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10, 3, key="xgb_max_depth")
                params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, key="xgb_learning_rate")
            elif model_choice == "KNN":
                params["n_neighbors"] = st.sidebar.slider("Number of Neighbors", 1, 20, 5, key="knn_n_neighbors")
                params["weights"] = st.sidebar.selectbox("Weights", ["uniform", "distance"], key="knn_weights")
            elif model_choice == "Decision Tree":
                params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5, key="dt_max_depth")
                params["criterion"] = st.sidebar.selectbox("Criterion", ["gini", "entropy"], key="dt_criterion")
            elif model_choice == "Linear Regression":
                st.sidebar.markdown("_No hyperparameters to tune for Linear Regression_")
            elif model_choice == "Logistic Regression":
                params["C"] = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0, key="logr_C")
                params["penalty"] = st.sidebar.selectbox("Penalty", ["l1", "l2"], index=1, key="logr_penalty")
                params["solver"] = st.sidebar.selectbox("Solver", ["liblinear", "saga"], index=0, key="logr_solver")
            elif model_choice == "XGBoost Regressor":
                params["max_depth"] = st.sidebar.slider("Max Depth", 1, 10, 3, key="xgbreg_max_depth")
                params["learning_rate"] = st.sidebar.slider("Learning Rate", 0.01, 0.5, 0.1, key="xgbreg_learning_rate")
        elif model_choice == "KMeans":
            params["n_clusters"] = st.sidebar.slider("Number of Clusters", 2, 10, 3, key="kmeans_n_clusters")
            params["init"] = st.sidebar.selectbox("Init Method", ["k-means++", "random"], key="kmeans_init")
            params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 500, 300, key="kmeans_max_iter")
        elif model_choice == "DBSCAN":
            params["eps"] = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, step=0.1, key="dbscan_eps")
            params["min_samples"] = st.sidebar.slider("Min Samples", 2, 10, 5, key="dbscan_min_samples")

        # --- Move Sidebar Advanced Options here ---
        st.sidebar.markdown("---")
        st.sidebar.header("5️⃣ Advanced ML Options")
        # Feature Selection
        use_feature_selection = st.sidebar.checkbox("Feature Selection (SelectKBest/RFE)", value=False, key="feature_selection_train")
        if use_feature_selection:
            fs_method = st.sidebar.selectbox("Method", ["kbest", "rfe"], index=0, key="fs_method_train")
            fs_k = st.sidebar.slider("Number of features (k)", 1, 20, 10, key="fs_k_train")
        # Imbalance Handling
        use_imbalance = st.sidebar.checkbox("Handle Class Imbalance (SMOTE/class_weight)", value=False, key="imbalance_train")
        if use_imbalance:
            imb_method = st.sidebar.selectbox("Imbalance Method", ["smote", "class_weight"], index=0, key="imb_method_train")
        # AutoML
        use_automl = st.sidebar.checkbox("AutoML: Test & Select Best Model", value=False, key="automl_train")
        automl_models = {}
        if use_automl:
            st.sidebar.markdown("**Select models to include in AutoML:**")
            automl_models["Random Forest"] = st.sidebar.checkbox("Random Forest", value=True, key="rf_train")
            automl_models["SVM"] = st.sidebar.checkbox("SVM", value=True, key="svm_train")
            automl_models["XGBoost"] = st.sidebar.checkbox("XGBoost", value=True, key="xgb_train")
            automl_models["KNN"] = st.sidebar.checkbox("KNN", value=True, key="knn_train")
            automl_models["Decision Tree"] = st.sidebar.checkbox("Decision Tree", value=True, key="dt_train")
            automl_models["Linear Regression"] = st.sidebar.checkbox("Linear Regression", value=True, key="lr_train")
            automl_models["Logistic Regression"] = st.sidebar.checkbox("Logistic Regression", value=True, key="logr_train")
            automl_models["XGBoost Regressor"] = st.sidebar.checkbox("XGBoost Regressor", value=True, key="xgb_reg_train")
            automl_models["DBSCAN"] = st.sidebar.checkbox("DBSCAN", value=True, key="dbscan_train")
        # AutoML + GridSearchCV
        use_automl_grid = st.sidebar.checkbox("AutoML + GridSearchCV (slow)", value=False, key="automl_grid_train")

        # Ensure model_choice is defined (in case of unexpected branch execution)
        model_choice = locals().get("model_choice", st.session_state.get("model_choice", "Random Forest"))

        # Exclude invalid target columns
        excluded_cols = [col for col in df.columns if col.lower() in ['id', 'index'] or pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_timedelta64_dtype(df[col])]
        target_options = [col for col in df.columns if col not in excluded_cols]
        
        if not target_options:
            st.error("No valid column found to use as target.")
            st.stop()

        suggested_target = suggest_target(df, model_choice)
        target_column = None
        if model_choice != "KMeans":
            target_column = st.selectbox(
                "🎯 Select Target Column",
                target_options,
                index=target_options.index(suggested_target) if suggested_target in target_options else 0,
                help="For classification models, choose a column with discrete values (e.g., categories, integers). For regression, choose a column with continuous values."
            )

            if target_column:
                is_numeric = pd.api.types.is_numeric_dtype(df[target_column])
                is_discrete = df[target_column].nunique() <= 20 or (is_numeric and df[target_column].dropna().apply(float.is_integer).all())
                
                if model_choice in ["Random Forest", "SVM", "XGBoost", "KNN", "Decision Tree", "Logistic Regression"] and not is_discrete:
                    st.error(f"Error: The selected target '{target_column}' contains continuous values, but {model_choice} expects discrete classes. Please select a categorical or integer column for classification.")
                    st.stop()
                elif model_choice == "Linear Regression" and is_discrete:
                    st.warning(f"Warning: The selected target '{target_column}' contains discrete values, but Linear Regression expects continuous values. Consider choosing a numerical column with continuous values.")
                
                st.markdown("### 🏷️ Target Class Distribution")
                st.markdown("**Distribution of your target variable to understand class balance**")
                target_counts = df[target_column].value_counts().reset_index()
                target_counts.columns = [target_column, 'Count']
                target_counts['Count'] = target_counts['Count'].astype(str)
                fig_target = px.bar(target_counts, x=target_column, y="Count", title=f"Class Distribution of Target: {target_column}", text_auto=True, color=target_column)
                st.plotly_chart(fig_target, use_container_width=True)

        # --- Prétraitement avancé automatique pour l'entraînement ---
        if df is not None and len(df) > 0:
            st.info("💡 **Give the advanced ML options in the sidebar a try to explore and discover which model works best for your data!**")
            with st.spinner("Preprocessing in progress..."):
                df_processed, feature_encoder = advanced_preprocessing_pipeline(
                    df, 
                    impute=True, 
                    outlier_method="winsorize", 
                    scale_method="standard", 
                    encode_method="ordinal", 
                    drop_useless=True,
                    encoder=None,
                    fit_encoder=True,
                    target_column=target_column
                )
                st.session_state["feature_encoder"] = feature_encoder
            df = df_processed

        st.subheader("⚙️ Model Configuration")
        st.markdown("**Parameters chosen for your machine learning model**")
        if 'params' not in locals():
            params = {}
        st.json(params)

        # --- AutoML automatique ---
        automl_ready = use_automl and model_choice != "KMeans" and target_column and len(df) > 0 and any(automl_models.values())
        if automl_ready:
            try:
                df_clean = clean_data(df)
                feature_cols = [
                    col for col in df_clean.columns
                    if (model_choice != "KMeans" and col != target_column)
                    and col.lower() not in ['id', 'index']
                    and not pd.api.types.is_datetime64_any_dtype(df_clean[col])
                    and not pd.api.types.is_timedelta64_dtype(df_clean[col])
                ]
                features = df_clean[feature_cols]
                features_encoded, _ = encode_categoricals(features)
                target_series = df_clean[target_column]
                # Always encode target for classification tasks
                if detect_task_type(target_series) == 'classification':
                    le = LabelEncoder()
                    y_encoded = le.fit_transform(target_series.astype(str))
                    df_encoded = features_encoded.copy()
                    df_encoded[target_column] = y_encoded
                    st.session_state["target_label_encoder"] = le
                else:
                    df_encoded = features_encoded.copy()
                    df_encoded[target_column] = target_series
                    st.session_state["target_label_encoder"] = None
                # Feature selection
                if use_feature_selection:
                    st.info(f"Feature selection: {fs_method} (top {fs_k})")
                    X_fs, selected_features = select_features(
                        df_encoded.drop(columns=[target_column]),
                        df_encoded[target_column],
                        method=fs_method,
                        k=fs_k
                    )
                    df_encoded = X_fs.copy()
                    df_encoded[target_column] = target_series.loc[X_fs.index]
                    st.success(f"Selected features: {', '.join(selected_features)}")
                # Imbalance handling
                if use_imbalance:
                    st.info(f"Imbalance handling: {imb_method}")
                    X_bal, y_bal, imb_info = handle_imbalance(
                        df_encoded.drop(columns=[target_column]),
                        df_encoded[target_column],
                        method=imb_method
                    )
                    df_encoded = X_bal.copy()
                    df_encoded[target_column] = y_bal
                    st.success(f"Class distribution before: {imb_info.get('before')}, after: {imb_info.get('after')}")
                X_train, X_test, y_train, y_test = split_data(df_encoded, target_column)
                # Préparer la liste des modèles sélectionnés
                selected_models = [name for name, checked in automl_models.items() if checked]
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.svm import SVC
                from xgboost import XGBClassifier
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.linear_model import LinearRegression
                from sklearn.linear_model import LogisticRegression # Added Logistic Regression import
                from sklearn.model_selection import GridSearchCV
                models_dict = {}
                param_grids = {}
                task_type = detect_task_type(y_train)
                if task_type == 'classification':
                    if "Random Forest" in selected_models:
                        models_dict["Random Forest"] = RandomForestClassifier(random_state=42)
                        param_grids["Random Forest"] = {"n_estimators": [50, 100], "max_depth": [5, 10]}
                    if "SVM" in selected_models:
                        models_dict["SVM"] = SVC(probability=True, random_state=42)
                        param_grids["SVM"] = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
                    if "XGBoost" in selected_models:
                        models_dict["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
                        param_grids["XGBoost"] = {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
                    if "KNN" in selected_models:
                        models_dict["KNN"] = KNeighborsClassifier()
                        param_grids["KNN"] = {"n_neighbors": [3, 5, 7]}
                    if "Decision Tree" in selected_models:
                        models_dict["Decision Tree"] = DecisionTreeClassifier(random_state=42)
                        param_grids["Decision Tree"] = {"max_depth": [5, 10], "criterion": ["gini", "entropy"]}
                    if "Logistic Regression" in selected_models:
                        models_dict["Logistic Regression"] = LogisticRegression(random_state=42)
                        param_grids["Logistic Regression"] = {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]}
                else:
                    if "Linear Regression" in selected_models:
                        models_dict["Linear Regression"] = LinearRegression()
                        param_grids["Linear Regression"] = {}  # Pas d'hyperparamètres pertinents
                if not models_dict:
                    st.warning("No compatible models selected for this task type.")
                    st.stop()
                if use_automl_grid:
                    st.warning("AutoML + GridSearchCV can be slow. Please wait...")
                    best_score = -float('inf') if task_type == 'regression' else 0
                    best_model = None
                    best_model_name = None
                    best_params = None
                    all_scores = {}
                    import pandas as pd
                    with st.spinner("AutoML + GridSearchCV in progress..."):
                        for name, model in models_dict.items():
                            grid = GridSearchCV(model, param_grids[name], cv=3, scoring='r2' if task_type=='regression' else 'accuracy', n_jobs=-1)
                            try:
                                grid.fit(X_train, y_train)
                                score = grid.best_score_
                                all_scores[name] = f"{score:.4f} (params: {grid.best_params_})"
                                if (task_type == 'classification' and score > best_score) or (task_type == 'regression' and score > best_score):
                                    best_score = score
                                    best_model = grid.best_estimator_
                                    best_model_name = name
                                    best_params = grid.best_params_
                            except Exception as e:
                                all_scores[name] = f"Error: {e}"
                    scores_df = pd.DataFrame(list(all_scores.items()), columns=["Model", "Score (CV)"])
                    st.markdown("### 🔍 AutoML + GridSearchCV Model Scores")
                    st.dataframe(scores_df, use_container_width=True)
                    if best_model_name is not None:
                        st.success(f"Best model: **{best_model_name}** (CV score: **{best_score:.4f}**)\nBest params: {best_params}")
                    else:
                        st.error("No model could be successfully trained. Please check your data or model selection.")
                else:
                    with st.spinner("AutoML in progress..."):
                        best_model_name, best_model, best_score, all_scores = simple_automl(X_train, y_train, X_test, y_test, task_type=task_type, models=models_dict)
                    import pandas as pd
                    scores_df = pd.DataFrame(list(all_scores.items()), columns=["Model", "Score"])
                    st.markdown("### 🔍 AutoML Model Scores")
                    st.dataframe(scores_df, use_container_width=True)
                    if best_model_name is not None:
                        st.success(f"Best model: **{best_model_name}** (score: **{best_score:.4f}**) 🏆")
                    else:
                        st.error("No model could be successfully trained. Please check your data or model selection.")
            except Exception as e:
                st.error(f"Error during AutoML: {e}")
                import traceback
                st.text(traceback.format_exc())
        else:
            run = st.button("🚀 Start Training", use_container_width=True, disabled=use_automl)
            if run:
                try:
                    df_clean = clean_data(df)
                    feature_cols = [
                        col for col in df_clean.columns
                        if (model_choice != "KMeans" and col != target_column)
                        and col.lower() not in ['id', 'index']
                        and not pd.api.types.is_datetime64_any_dtype(df_clean[col])
                        and not pd.api.types.is_timedelta64_dtype(df_clean[col])
                    ]
                    features = df_clean[feature_cols]
                    features_encoded, _ = encode_categoricals(features)
                    target_series = df_clean[target_column]
                    # Always encode target for classification tasks
                    if detect_task_type(target_series) == 'classification':
                        le = LabelEncoder()
                        y_encoded = le.fit_transform(target_series.astype(str))
                        df_encoded = features_encoded.copy()
                        df_encoded[target_column] = y_encoded
                        st.session_state["target_label_encoder"] = le
                    else:
                        df_encoded = features_encoded.copy()
                        df_encoded[target_column] = target_series
                        st.session_state["target_label_encoder"] = None
                    # Feature selection
                    if use_feature_selection:
                        st.info(f"Feature selection: {fs_method} (top {fs_k})")
                        X_fs, selected_features = select_features(
                            df_encoded.drop(columns=[target_column]),
                            df_encoded[target_column],
                            method=fs_method,
                            k=fs_k
                        )
                        df_encoded = X_fs.copy()
                        df_encoded[target_column] = target_series.loc[X_fs.index]
                        st.success(f"Selected features: {', '.join(selected_features)}")
                    # Imbalance handling
                    if use_imbalance:
                        st.info(f"Imbalance handling: {imb_method}")
                        X_bal, y_bal, imb_info = handle_imbalance(
                            df_encoded.drop(columns=[target_column]),
                            df_encoded[target_column],
                            method=imb_method
                        )
                        df_encoded = X_bal.copy()
                        df_encoded[target_column] = y_bal
                        st.success(f"Class distribution before: {imb_info.get('before')}, after: {imb_info.get('after')}")
                    X_train, X_test, y_train, y_test = split_data(df_encoded, target_column)
                    model = train_model(model_choice, X_train, y_train if model_choice != "KMeans" else None, params)
                    results = evaluate_model(model, X_test, y_test if model_choice != "KMeans" else None)

                    figures_to_save = []

                    # Debug: Print the model choice
                    st.write(f"Debug: Processing model: {model_choice}")

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

                    elif model_choice == "DBSCAN":
                        st.subheader("📌 Clustering Results")
                        st.markdown("**Groups created by the DBSCAN algorithm to segment your data**")
                        st.write(results["labels"].astype(str))
                        pca = PCA(n_components=2)
                        X_test_2d = pca.fit_transform(X_test)
                        pca_df = pd.DataFrame(X_test_2d, columns=["PC1", "PC2"])
                        pca_df["Cluster"] = results["labels"].astype(str)
                        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster", title="PCA Projection of DBSCAN Clusters", color_discrete_sequence=px.colors.qualitative.Set2)
                        st.subheader("📊 2D Visualization of Clusters")
                        st.markdown("**2D visualization of clusters created by DBSCAN using PCA dimensionality reduction**")
                        st.plotly_chart(fig_pca, use_container_width=True)
                        figures_to_save.append(fig_pca)

                    elif model_choice == "Linear Regression":
                        st.markdown("**Evaluation metrics for linear regression**")
                        st.metric("MSE", f"{results['mse']:.4f}")
                        st.metric("R²", f"{results['r2']:.4f}")

                        fig = px.scatter(x=y_test, y=results["y_pred"], labels={'x': 'True Values', 'y': 'Predicted Values'}, title="True vs Predicted", color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='black', dash='dash'))
                        st.markdown("**Comparison of predicted vs actual values (closer to the line = better model)**")
                        st.plotly_chart(fig, use_container_width=True)
                        figures_to_save.append(fig)

                        st.subheader("Error Distribution")
                        st.markdown("**Distribution of prediction errors to evaluate model quality**")
                        errors = y_test - results["y_pred"]
                        fig_err = px.histogram(errors, nbins=30, marginal="box", title="Distribution of Prediction Errors", color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig_err, use_container_width=True)
                        figures_to_save.append(fig_err)

                        # Residual plot
                        st.subheader("Residual Plot")
                        st.markdown("**Residuals (errors) vs predicted values. A good model should have residuals randomly scattered around zero.**")
                        fig_resid = px.scatter(x=results["y_pred"], y=errors, labels={'x': 'Predicted Values', 'y': 'Residuals (y_true - y_pred)'}, title="Residual Plot", color_discrete_sequence=px.colors.qualitative.Set2)
                        fig_resid.add_shape(type='line', x0=min(results["y_pred"]), y0=0, x1=max(results["y_pred"]), y1=0, line=dict(color='black', dash='dash'))
                        st.plotly_chart(fig_resid, use_container_width=True)
                        figures_to_save.append(fig_resid)

                    elif model_choice == "XGBoost Regressor":
                        st.markdown("**Evaluation metrics for XGBoost regression**")
                        st.metric("MSE", f"{results['mse']:.4f}")
                        st.metric("R²", f"{results['r2']:.4f}")

                        fig = px.scatter(x=y_test, y=results["y_pred"], labels={'x': 'True Values', 'y': 'Predicted Values'}, title="True vs Predicted (XGBoost)", color_discrete_sequence=px.colors.qualitative.Set2)
                        fig.add_shape(type='line', x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color='black', dash='dash'))
                        st.markdown("**Comparison of predicted vs actual values (closer to the line = better model)**")
                        st.plotly_chart(fig, use_container_width=True)
                        figures_to_save.append(fig)

                        st.subheader("Error Distribution")
                        st.markdown("**Distribution of prediction errors to evaluate model quality**")
                        errors = y_test - results["y_pred"]
                        fig_err = px.histogram(errors, nbins=30, marginal="box", title="Distribution of Prediction Errors (XGBoost)", color_discrete_sequence=px.colors.qualitative.Set2)
                        st.plotly_chart(fig_err, use_container_width=True)
                        figures_to_save.append(fig_err)

                        # Residual plot
                        st.subheader("Residual Plot")
                        st.markdown("**Residuals (errors) vs predicted values. A good model should have residuals randomly scattered around zero.**")
                        fig_resid = px.scatter(x=results["y_pred"], y=errors, labels={'x': 'Predicted Values', 'y': 'Residuals (y_true - y_pred)'}, title="Residual Plot (XGBoost)", color_discrete_sequence=px.colors.qualitative.Set2)
                        fig_resid.add_shape(type='line', x0=min(results["y_pred"]), y0=0, x1=max(results["y_pred"]), y1=0, line=dict(color='black', dash='dash'))
                        st.plotly_chart(fig_resid, use_container_width=True)
                        figures_to_save.append(fig_resid)

                    elif model_choice == "Logistic Regression":
                        st.markdown("**Evaluation metrics for logistic regression**")
                        st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        st.metric("F1 Score", f"{results.get('f1_score', 'N/A'):.2%}")
                        st.metric("Recall", f"{results.get('recall', 'N/A'):.2%}")
                        st.metric("Precision", f"{results.get('precision', 'N/A'):.2%}")
                        if results.get('auc_roc') is not None:
                            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")

                        st.subheader("📉 Confusion Matrix")
                        st.markdown("**Table showing correct vs incorrect predictions for each class**")
                        fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        figures_to_save.append(fig_cm)

                        y_proba = results.get("y_proba")
                        # Correction robuste : vérifier que y_proba n'est pas None et est bien itérable
                        if y_proba is not None:
                            try:
                                _ = iter(y_proba)
                                if len(np.unique(y_test)) == 2:
                                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                                    roc_auc = auc(fpr, tpr)
                                    fig_roc = px.area(x=fpr, y=tpr, title=f"ROC Curve (AUC = {roc_auc:.2f})", labels=dict(x="False Positive Rate", y="True Positive Rate"), color_discrete_sequence=px.colors.qualitative.Set2)
                                    fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash'))
                                    st.markdown("**ROC curve showing the model's ability to distinguish classes (AUC > 0.5 = good model)**")
                                    st.plotly_chart(fig_roc, use_container_width=True)
                                    figures_to_save.append(fig_roc)
                                    # PR Curve
                                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                                    fig_pr = px.area(x=recall, y=precision, title="Precision-Recall (PR) Curve", labels=dict(x="Recall", y="Precision"), color_discrete_sequence=px.colors.qualitative.Set2)
                                    fig_pr.add_shape(type='line', x0=0, y0=1, x1=1, y1=0, line=dict(dash='dash'))
                                    st.markdown("**Precision-Recall (PR) curve: Useful for imbalanced datasets. The higher the area under the curve, the better.**")
                                    st.plotly_chart(fig_pr, use_container_width=True)
                                    figures_to_save.append(fig_pr)
                            except Exception as e:
                                st.warning(f"Impossible d'afficher la courbe ROC/PR : {e}")

                        if model_choice in ["Random Forest", "XGBoost", "Decision Tree"] and hasattr(model, "feature_importances_") and hasattr(X_train, 'columns'):
                            st.subheader("🌟 Feature Importance")
                            st.markdown("**Most important variables for the model's predictions**")
                            importances = model.feature_importances_
                            features = X_train.columns
                            if len(features) == len(importances):
                                imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                                imp_df["Importance"] = imp_df["Importance"].astype(str)
                                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Feature Importance", color="Importance", color_continuous_scale='Viridis')
                                st.plotly_chart(fig_imp, use_container_width=True)
                                figures_to_save.append(fig_imp)
                            else:
                                st.warning("Mismatch between number of features and importances. Cannot display feature importances.")

                        elif model_choice == "SVM" and params.get("kernel") == "linear" and hasattr(model, "coef_"):
                            st.subheader("🌟 SVM Coefficients")
                            st.markdown("**Coefficients of the linear SVM model showing the importance of each variable**")
                            coefs = model.coef_.flatten()
                            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"f{i}" for i in range(len(coefs))]
                            if len(feature_names) == len(coefs):
                                coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs}).sort_values(by="Coefficient", ascending=False)
                                coef_df["Coefficient"] = coef_df["Coefficient"].astype(str)
                                fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="SVM Coefficients", color="Coefficient", color_continuous_scale='RdBu')
                                st.plotly_chart(fig_coef, use_container_width=True)
                                figures_to_save.append(fig_coef)
                            else:
                                st.warning("Mismatch between number of features and coefficients. Cannot display SVM coefficients.")

                    elif model_choice == "XGBoost":
                        st.markdown("**Evaluation metrics for XGBoost classification**")
                        st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        st.metric("F1 Score", f"{results.get('f1_score', 'N/A'):.2%}")
                        st.metric("Recall", f"{results.get('recall', 'N/A'):.2%}")
                        st.metric("Precision", f"{results.get('precision', 'N/A'):.2%}")
                        if results.get('auc_roc') is not None:
                            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")

                        st.subheader("📉 Confusion Matrix")
                        st.markdown("**Table showing correct vs incorrect predictions for each class**")
                        fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix (XGBoost)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        figures_to_save.append(fig_cm)

                        # 1. Feature Importance Plot for XGBoost
                        if hasattr(model, "feature_importances_") and hasattr(X_train, 'columns'):
                            st.subheader("🌟 Feature Importance Plot")
                            st.markdown("**Most important variables for the XGBoost model's predictions**")
                            importances = model.feature_importances_
                            features = X_train.columns
                            if len(features) == len(importances):
                                imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                                imp_df["Importance"] = imp_df["Importance"].astype(str)
                                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="XGBoost Feature Importance", color="Importance", color_continuous_scale='Viridis')
                                st.plotly_chart(fig_imp, use_container_width=True)
                                figures_to_save.append(fig_imp)
                            else:
                                st.warning("Mismatch between number of features and importances. Cannot display feature importances.")

                        # 2. Precision-Recall Curve for XGBoost
                        y_proba = results.get("y_proba")
                        if y_proba is not None:
                            try:
                                _ = iter(y_proba)
                                if len(np.unique(y_test)) == 2:
                                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                                    fig_pr = px.area(x=recall, y=precision, title="Precision-Recall (PR) Curve (XGBoost)", labels=dict(x="Recall", y="Precision"), color_discrete_sequence=px.colors.qualitative.Set2)
                                    fig_pr.add_shape(type='line', x0=0, y0=1, x1=1, y1=0, line=dict(dash='dash'))
                                    st.markdown("**Precision-Recall (PR) curve: Useful for imbalanced datasets. The higher the area under the curve, the better.**")
                                    st.plotly_chart(fig_pr, use_container_width=True)
                                    figures_to_save.append(fig_pr)
                            except Exception as e:
                                st.warning(f"Impossible d'afficher la courbe Precision-Recall : {e}")

                        # 3. SHAP Summary Plot for XGBoost
                        try:
                            import shap
                            st.subheader("🔍 SHAP Summary Plot (Model Explainability)")
                            st.markdown("**SHAP values show how each feature contributes to the model's predictions. Red indicates higher feature values, blue indicates lower values.**")
                            
                            # Create SHAP explainer
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test)
                            
                            # Create SHAP summary plot
                            fig_shap = go.Figure()
                            
                            # Handle different SHAP value shapes more robustly
                            if len(shap_values.shape) > 1:
                                # For multi-class, use the first class or average across classes
                                if shap_values.shape[1] > 1:
                                    # Use the first class for binary classification or average for multi-class
                                    if shap_values.shape[1] == 2:
                                        shap_values_plot = shap_values[:, 1]  # Positive class for binary
                                    else:
                                        shap_values_plot = np.mean(shap_values, axis=1)  # Average for multi-class
                                else:
                                    shap_values_plot = shap_values[:, 0]
                            else:
                                shap_values_plot = shap_values
                            
                            # Get feature names safely
                            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f"Feature_{i}" for i in range(X_test.shape[1])]
                            
                            # Ensure we don't exceed the number of features
                            num_features = min(len(feature_names), shap_values_plot.shape[1] if len(shap_values_plot.shape) > 1 else 1)
                            
                            # Create summary plot data
                            for i in range(num_features):
                                if len(shap_values_plot.shape) > 1:
                                    shap_vals = shap_values_plot[:, i]
                                else:
                                    shap_vals = shap_values_plot
                                
                                feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                                
                                # Get feature values safely
                                if hasattr(X_test, 'iloc') and i < X_test.shape[1]:
                                    feature_vals = X_test.iloc[:, i]
                                else:
                                    feature_vals = np.zeros(len(shap_vals))
                                
                                fig_shap.add_trace(go.Scatter(
                                    x=shap_vals,
                                    y=[feature_name] * len(shap_vals),
                                    mode='markers',
                                    marker=dict(
                                        color=feature_vals,
                                        colorscale='RdBu',
                                        showscale=True,
                                        colorbar=dict(title="Feature Value")
                                    ),
                                    name=feature_name,
                                    hovertemplate=f'{feature_name}<br>SHAP: %{{x}}<br>Value: %{{marker.color}}<extra></extra>'
                                ))
                            
                            fig_shap.update_layout(
                                title="SHAP Summary Plot (XGBoost)",
                                xaxis_title="SHAP Value",
                                yaxis_title="Features",
                                showlegend=False,
                                height=600
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                            figures_to_save.append(fig_shap)
                            
                        except ImportError:
                            st.warning("SHAP library not available. Install with: pip install shap")
                        except Exception as e:
                            st.warning(f"Could not generate SHAP plot: {e}")
                            st.info("This might be due to the dataset size or model complexity. The other evaluation graphs are still available.")

                    elif model_choice == "Random Forest":
                        st.markdown("**Evaluation metrics for Random Forest classification**")
                        st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        st.metric("F1 Score", f"{results.get('f1_score', 'N/A'):.2%}")
                        st.metric("Recall", f"{results.get('recall', 'N/A'):.2%}")
                        st.metric("Precision", f"{results.get('precision', 'N/A'):.2%}")
                        if results.get('auc_roc') is not None:
                            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")

                        st.subheader("📉 Confusion Matrix")
                        st.markdown("**Table showing correct vs incorrect predictions for each class**")
                        fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix (Random Forest)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        figures_to_save.append(fig_cm)

                        # Feature Importance for Random Forest
                        if hasattr(model, "feature_importances_") and hasattr(X_train, 'columns'):
                            st.subheader("🌟 Feature Importance")
                            st.markdown("**Most important variables for the Random Forest model's predictions**")
                            importances = model.feature_importances_
                            features = X_train.columns
                            if len(features) == len(importances):
                                imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                                imp_df["Importance"] = imp_df["Importance"].astype(str)
                                # Utilisation d'une palette qualitative pour garantir la couleur
                                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Random Forest Feature Importance", color="Feature", color_discrete_sequence=px.colors.qualitative.Set2)
                                st.plotly_chart(fig_imp, use_container_width=True)
                                figures_to_save.append(fig_imp)
                            else:
                                st.warning("Mismatch between number of features and importances. Cannot display feature importances.")

                    elif model_choice == "SVM":
                        st.markdown("**Evaluation metrics for SVM classification**")
                        st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        st.metric("F1 Score", f"{results.get('f1_score', 'N/A'):.2%}")
                        st.metric("Recall", f"{results.get('recall', 'N/A'):.2%}")
                        st.metric("Precision", f"{results.get('precision', 'N/A'):.2%}")
                        if results.get('auc_roc') is not None:
                            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")

                        st.subheader("📉 Confusion Matrix")
                        st.markdown("**Table showing correct vs incorrect predictions for each class**")
                        fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix (SVM)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        figures_to_save.append(fig_cm)

                        # SVM Coefficients for linear kernel
                        if params.get("kernel") == "linear" and hasattr(model, "coef_"):
                            st.subheader("🌟 SVM Coefficients")
                            st.markdown("**Coefficients of the linear SVM model showing the importance of each variable**")
                            coefs = model.coef_.flatten()
                            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"f{i}" for i in range(len(coefs))]
                            if len(feature_names) == len(coefs):
                                coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs}).sort_values(by="Coefficient", ascending=False)
                                coef_df["Coefficient"] = coef_df["Coefficient"].astype(str)
                                fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="SVM Coefficients", color="Coefficient", color_continuous_scale='RdBu')
                                st.plotly_chart(fig_coef, use_container_width=True)
                                figures_to_save.append(fig_coef)
                            else:
                                st.warning("Mismatch between number of features and coefficients. Cannot display SVM coefficients.")

                    elif model_choice == "XGBoost":
                        st.markdown("**Evaluation metrics for XGBoost classification**")
                        st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        st.metric("F1 Score", f"{results.get('f1_score', 'N/A'):.2%}")
                        st.metric("Recall", f"{results.get('recall', 'N/A'):.2%}")
                        st.metric("Precision", f"{results.get('precision', 'N/A'):.2%}")
                        if results.get('auc_roc') is not None:
                            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")

                        st.subheader("📉 Confusion Matrix")
                        st.markdown("**Table showing correct vs incorrect predictions for each class**")
                        fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix (XGBoost)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        figures_to_save.append(fig_cm)

                        # 1. Feature Importance Plot for XGBoost
                        if hasattr(model, "feature_importances_") and hasattr(X_train, 'columns'):
                            st.subheader("🌟 Feature Importance Plot")
                            st.markdown("**Most important variables for the XGBoost model's predictions**")
                            importances = model.feature_importances_
                            features = X_train.columns
                            if len(features) == len(importances):
                                imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                                imp_df["Importance"] = imp_df["Importance"].astype(str)
                                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="XGBoost Feature Importance", color="Importance", color_continuous_scale='Viridis')
                                st.plotly_chart(fig_imp, use_container_width=True)
                                figures_to_save.append(fig_imp)
                            else:
                                st.warning("Mismatch between number of features and importances. Cannot display feature importances.")

                        # 2. Precision-Recall Curve for XGBoost
                        y_proba = results.get("y_proba")
                        if y_proba is not None:
                            try:
                                _ = iter(y_proba)
                                if len(np.unique(y_test)) == 2:
                                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                                    fig_pr = px.area(x=recall, y=precision, title="Precision-Recall (PR) Curve (XGBoost)", labels=dict(x="Recall", y="Precision"), color_discrete_sequence=px.colors.qualitative.Set2)
                                    fig_pr.add_shape(type='line', x0=0, y0=1, x1=1, y1=0, line=dict(dash='dash'))
                                    st.markdown("**Precision-Recall (PR) curve: Useful for imbalanced datasets. The higher the area under the curve, the better.**")
                                    st.plotly_chart(fig_pr, use_container_width=True)
                                    figures_to_save.append(fig_pr)
                            except Exception as e:
                                st.warning(f"Impossible d'afficher la courbe Precision-Recall : {e}")

                        # 3. SHAP Summary Plot for XGBoost
                        try:
                            import shap
                            st.subheader("🔍 SHAP Summary Plot (Model Explainability)")
                            st.markdown("**SHAP values show how each feature contributes to the model's predictions. Red indicates higher feature values, blue indicates lower values.**")
                            
                            # Create SHAP explainer
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_test)
                            
                            # Create SHAP summary plot
                            fig_shap = go.Figure()
                            
                            # Handle different SHAP value shapes more robustly
                            if len(shap_values.shape) > 1:
                                # For multi-class, use the first class or average across classes
                                if shap_values.shape[1] > 1:
                                    # Use the first class for binary classification or average for multi-class
                                    if shap_values.shape[1] == 2:
                                        shap_values_plot = shap_values[:, 1]  # Positive class for binary
                                    else:
                                        shap_values_plot = np.mean(shap_values, axis=1)  # Average for multi-class
                                else:
                                    shap_values_plot = shap_values[:, 0]
                            else:
                                shap_values_plot = shap_values
                            
                            # Get feature names safely
                            feature_names = X_test.columns if hasattr(X_test, 'columns') else [f"Feature_{i}" for i in range(X_test.shape[1])]
                            
                            # Ensure we don't exceed the number of features
                            num_features = min(len(feature_names), shap_values_plot.shape[1] if len(shap_values_plot.shape) > 1 else 1)
                            
                            # Create summary plot data
                            for i in range(num_features):
                                if len(shap_values_plot.shape) > 1:
                                    shap_vals = shap_values_plot[:, i]
                                else:
                                    shap_vals = shap_values_plot
                                
                                feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                                
                                # Get feature values safely
                                if hasattr(X_test, 'iloc') and i < X_test.shape[1]:
                                    feature_vals = X_test.iloc[:, i]
                                else:
                                    feature_vals = np.zeros(len(shap_vals))
                                
                                fig_shap.add_trace(go.Scatter(
                                    x=shap_vals,
                                    y=[feature_name] * len(shap_vals),
                                    mode='markers',
                                    marker=dict(
                                        color=feature_vals,
                                        colorscale='RdBu',
                                        showscale=True,
                                        colorbar=dict(title="Feature Value")
                                    ),
                                    name=feature_name,
                                    hovertemplate=f'{feature_name}<br>SHAP: %{{x}}<br>Value: %{{marker.color}}<extra></extra>'
                                ))
                            
                            fig_shap.update_layout(
                                title="SHAP Summary Plot (XGBoost)",
                                xaxis_title="SHAP Value",
                                yaxis_title="Features",
                                showlegend=False,
                                height=600
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                            figures_to_save.append(fig_shap)
                            
                        except ImportError:
                            st.warning("SHAP library not available. Install with: pip install shap")
                        except Exception as e:
                            st.warning(f"Could not generate SHAP plot: {e}")
                            st.info("This might be due to the dataset size or model complexity. The other evaluation graphs are still available.")

                    elif model_choice == "Random Forest":
                        st.markdown("**Evaluation metrics for Random Forest classification**")
                        st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        st.metric("F1 Score", f"{results.get('f1_score', 'N/A'):.2%}")
                        st.metric("Recall", f"{results.get('recall', 'N/A'):.2%}")
                        st.metric("Precision", f"{results.get('precision', 'N/A'):.2%}")
                        if results.get('auc_roc') is not None:
                            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")

                        st.subheader("📉 Confusion Matrix")
                        st.markdown("**Table showing correct vs incorrect predictions for each class**")
                        fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix (Random Forest)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        figures_to_save.append(fig_cm)

                        # Feature Importance for Random Forest
                        if hasattr(model, "feature_importances_") and hasattr(X_train, 'columns'):
                            st.subheader("🌟 Feature Importance")
                            st.markdown("**Most important variables for the Random Forest model's predictions**")
                            importances = model.feature_importances_
                            features = X_train.columns
                            if len(features) == len(importances):
                                imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=False)
                                imp_df["Importance"] = imp_df["Importance"].astype(str)
                                # Utilisation d'une palette qualitative pour garantir la couleur
                                fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h", title="Random Forest Feature Importance", color="Feature", color_discrete_sequence=px.colors.qualitative.Set2)
                                st.plotly_chart(fig_imp, use_container_width=True)
                                figures_to_save.append(fig_imp)
                            else:
                                st.warning("Mismatch between number of features and importances. Cannot display feature importances.")

                    elif model_choice == "SVM":
                        st.markdown("**Evaluation metrics for SVM classification**")
                        st.metric("Accuracy", f"{results['accuracy']:.2%}")
                        st.metric("F1 Score", f"{results.get('f1_score', 'N/A'):.2%}")
                        st.metric("Recall", f"{results.get('recall', 'N/A'):.2%}")
                        st.metric("Precision", f"{results.get('precision', 'N/A'):.2%}")
                        if results.get('auc_roc') is not None:
                            st.metric("AUC-ROC", f"{results['auc_roc']:.4f}")

                        st.subheader("📉 Confusion Matrix")
                        st.markdown("**Table showing correct vs incorrect predictions for each class**")
                        fig_cm = px.imshow(results["confusion_matrix"], text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix (SVM)")
                        st.plotly_chart(fig_cm, use_container_width=True)
                        figures_to_save.append(fig_cm)

                        # SVM Coefficients for linear kernel
                        if params.get("kernel") == "linear" and hasattr(model, "coef_"):
                            st.subheader("🌟 SVM Coefficients")
                            st.markdown("**Coefficients of the linear SVM model showing the importance of each variable**")
                            coefs = model.coef_.flatten()
                            feature_names = X_train.columns if hasattr(X_train, 'columns') else [f"f{i}" for i in range(len(coefs))]
                            if len(feature_names) == len(coefs):
                                coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs}).sort_values(by="Coefficient", ascending=False)
                                coef_df["Coefficient"] = coef_df["Coefficient"].astype(str)
                                fig_coef = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h", title="SVM Coefficients", color="Coefficient", color_continuous_scale='RdBu')
                                st.plotly_chart(fig_coef, use_container_width=True)
                                figures_to_save.append(fig_coef)
                            else:
                                st.warning("Mismatch between number of features and coefficients. Cannot display SVM coefficients.")

                    elif model_choice == "Linear Regression":
                        st.sidebar.markdown("_No hyperparameters to tune for Linear Regression_")

                    # figures_to_save = st.session_state.exploration_figures + figures_to_save
                    # Correction : on ne concatène plus les figures d'exploration, on garde uniquement celles du modèle courant
                    
                    # Inclure toutes les figures (exploration + training) dans le PDF
                    all_figures = st.session_state.exploration_figures + figures_to_save

                    # Sauvegarder les figures dans la base de données
                    with sqlite3.connect('users.db', timeout=30) as conn:
                        c = conn.cursor()
                        for fig in figures_to_save:
                            tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            fig.write_image(tmp_img.name, format="png", width=800, height=600, scale=2)
                            tmp_img_path = tmp_img.name
                            tmp_img.close()  # Ensure file is closed before reading
                            with open(tmp_img_path, 'rb') as f:
                                img_data = f.read()
                            c.execute("INSERT INTO user_data (username, dataset_id, data_type, data_content) VALUES (?, ?, ?, ?)", (st.session_state.username, dataset_id, "figure", sqlite3.Binary(img_data)))
                            os.unlink(tmp_img_path)  # Delete after file is closed
                        conn.commit()

                    # --- Sauvegarde des figures dans la base de données (user_data) ---
                    for fig in figures_to_save:
                        tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                        fig.write_image(tmp_img.name, format="png", width=800, height=600, scale=2)
                        tmp_img_path = tmp_img.name
                        tmp_img.close()
                        with open(tmp_img_path, 'rb') as f:
                            img_data = f.read()
                        new_ud = UserData(username=st.session_state.username, dataset_id=dataset_id, data_type="figure", data_content=img_data)
                        session.add(new_ud)
                        os.unlink(tmp_img_path)
                    session.commit()

                    st.markdown("**📥 Download a complete PDF report with all results**")
                    pdf_path = generate_pdf_report_reportlab(df, model_choice, params, target_column, results, all_figures)
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF Report", data=f.read(), file_name=f"ml_report_{st.session_state.username}.pdf", mime="application/pdf")
                    os.unlink(pdf_path)

                    # After model training, store model, features, target, and preprocessing in session_state ---
                    st.session_state["trained_model"] = model
                    st.session_state["trained_model_features"] = [col for col in X_train.columns if col != target_column]  # Exclude target
                    st.session_state["trained_model_target"] = target_column
                    st.session_state["preprocess_func"] = lambda df: advanced_preprocessing_pipeline(
                        df, impute=True, outlier_method="winsorize", scale_method="standard", encode_method="ordinal", drop_useless=False,
                        encoder=st.session_state.get("feature_encoder"), fit_encoder=False
                    )[[col for col in X_train.columns if col != target_column]]  # Exclude target
                    st.success("✅ Model trained and saved! You can now go to the Prediction tab to make predictions on new data.")

                except TypeError as e:
                    st.error(f"TypeError: {e}")
                    import traceback
                    st.text(traceback.format_exc())
                except Exception as e:
                    st.error(f"Error during model training: {e}")
                    import traceback
                    st.text(traceback.format_exc())


    
    # End of main application content
    # (This content only shows when user is authenticated and on dashboard page)
else:
    if not st.session_state.get('authenticated', False):
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("Please log in to your account to get started.")

# Modern Footer
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem 0; background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); border-top: 1px solid #e2e8f0; text-align: center;">
    <div style="margin-top: 0.5rem; font-family: 'Tommy-Light', Arial, sans-serif; color: #94a3b8; font-size: 0.75rem;">
        © 2025 ML Explorer - Making Machine Learning Accessible to Everyone
    </div>
</div>

""", unsafe_allow_html=True)

import os
os.environ["PATH"] += os.pathsep + os.path.dirname(os.__file__) + r"\site-packages\kaleido\executable"
