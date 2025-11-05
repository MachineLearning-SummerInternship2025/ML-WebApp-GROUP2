import os
import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import bcrypt
from cryptography.fernet import Fernet
from passlib.context import CryptContext
import streamlit as st

# Configuration de sécurité
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SecurityManager:
    def __init__(self):
        self.secret_key = os.getenv('SECRET_KEY', 'default-secret-key-change-in-production')
        self.max_login_attempts = int(os.getenv('MAX_LOGIN_ATTEMPTS', 5))
        self.login_timeout = int(os.getenv('LOGIN_TIMEOUT', 300))  # 5 minutes
        self.password_min_length = int(os.getenv('PASSWORD_MIN_LENGTH', 8))
        self.rate_limit_requests = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
        self.rate_limit_window = int(os.getenv('RATE_LIMIT_WINDOW', 3600))  # 1 heure
        
        # Initialiser le chiffrement
        self.cipher_key = Fernet.generate_key()
        self.cipher = Fernet(self.cipher_key)
        
        # Stockage des tentatives de connexion
        # Structure: { username: { 'count': int, 'last_attempt': float, 'blocked_until': Optional[float] } }
        self.login_attempts: Dict[str, Dict] = {}
        self.rate_limit: Dict[str, list] = {}
        # CAPTCHA configuration
        self.captcha_ttl_seconds = int(os.getenv('CAPTCHA_TTL_SECONDS', 180))
    
    def validate_password_strength(self, password: str) -> Tuple[bool, str]:
        """Valide la force du mot de passe"""
        if len(password) < self.password_min_length:
            return False, f"Le mot de passe doit contenir au moins {self.password_min_length} caractères"
        
        if not any(c.isupper() for c in password):
            return False, "Le mot de passe doit contenir au moins une majuscule"
        
        if not any(c.islower() for c in password):
            return False, "Le mot de passe doit contenir au moins une minuscule"
        
        if not any(c.isdigit() for c in password):
            return False, "Le mot de passe doit contenir au moins un chiffre"
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Le mot de passe doit contenir au moins un caractère spécial"
        
        return True, "Mot de passe valide"
    
    def hash_password(self, password: str) -> str:
        """Hash le mot de passe avec bcrypt"""
        return pwd_context.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Vérifie le mot de passe hashé"""
        return pwd_context.verify(password, hashed)
    
    def check_login_attempts(self, username: str) -> Tuple[bool, int, int]:
        """Vérifie les tentatives de connexion et applique le rate limiting.

        Retourne (can_login, attempt_count, rate_limit_hit)
        - can_login: False si verrouillé (trop de tentatives) ou rate limit atteint
        - attempt_count: nombre de tentatives déjà effectuées dans la fenêtre
        - rate_limit_hit: >0 si rate limit atteint, sinon 0
        """
        current_time = time.time()
        
        # Charger ou initialiser l'état des tentatives
        attempts = self.login_attempts.get(username, {
            'count': 0,
            'last_attempt': 0.0,
            'blocked_until': None
        })

        # Vérifier si le compte est verrouillé
        blocked_until = attempts.get('blocked_until')
        if blocked_until and current_time < blocked_until:
            # Toujours verrouillé
            self.login_attempts[username] = attempts
            return False, attempts.get('count', 0), 0

        # Déverrouiller si la période est passée
        if blocked_until and current_time >= blocked_until:
            attempts = {'count': 0, 'last_attempt': 0.0, 'blocked_until': None}
            self.login_attempts[username] = attempts

        # Nettoyer si inactif depuis longtemps (hors verrouillage)
        if attempts.get('last_attempt', 0.0) and (current_time - attempts['last_attempt'] > self.login_timeout):
            attempts['count'] = 0
            attempts['last_attempt'] = 0.0
            attempts['blocked_until'] = None
            self.login_attempts[username] = attempts
        
        # Vérifier le rate limiting
        if username in self.rate_limit:
            user_requests = self.rate_limit[username]
            # Nettoyer les anciennes requêtes
            user_requests = [req_time for req_time in user_requests 
                           if current_time - req_time < self.rate_limit_window]
            
            if len(user_requests) >= self.rate_limit_requests:
                return False, attempts['count'], self.rate_limit_requests
            
            self.rate_limit[username] = user_requests
        
        return True, attempts['count'], 0
    
    def record_login_attempt(self, username: str, success: bool):
        """Enregistre une tentative de connexion et applique le verrouillage si nécessaire"""
        current_time = time.time()
        
        if username not in self.login_attempts:
            self.login_attempts[username] = {'count': 0, 'last_attempt': 0.0, 'blocked_until': None}
        
        if not success:
            self.login_attempts[username]['count'] += 1
            self.login_attempts[username]['last_attempt'] = current_time
            # Verrouiller si le max est atteint
            if self.login_attempts[username]['count'] >= self.max_login_attempts:
                self.login_attempts[username]['blocked_until'] = current_time + self.login_timeout
        else:
            # Réinitialiser les tentatives en cas de succès
            self.login_attempts[username] = {'count': 0, 'last_attempt': current_time, 'blocked_until': None}
    
    def record_request(self, username: str):
        """Enregistre une requête pour le rate limiting"""
        current_time = time.time()
        
        if username not in self.rate_limit:
            self.rate_limit[username] = []
        
        self.rate_limit[username].append(current_time)
    
    def generate_session_token(self, username: str) -> str:
        """Génère un token de session sécurisé"""
        timestamp = str(int(time.time()))
        random_part = secrets.token_urlsafe(32)
        data = f"{username}:{timestamp}:{random_part}"
        return hashlib.sha256(data.encode()).hexdigest()

    def get_remaining_lock_time(self, username: str) -> int:
        """Retourne le temps restant (en secondes) si le compte est verrouillé, sinon 0"""
        current_time = time.time()
        attempts = self.login_attempts.get(username)
        if not attempts:
            return 0
        blocked_until = attempts.get('blocked_until')
        if blocked_until and current_time < blocked_until:
            return int(blocked_until - current_time)
        return 0

    def get_remaining_attempts(self, username: str) -> int:
        """Retourne le nombre de tentatives restantes avant verrouillage"""
        attempts = self.login_attempts.get(username, {'count': 0})
        remaining = self.max_login_attempts - attempts.get('count', 0)
        return max(0, remaining)

    # --- CAPTCHA ---
    def _generate_math_captcha(self) -> str:
        """Génère une question CAPTCHA mathématique simple et stocke la réponse attendue en session"""
        a = secrets.randbelow(9) + 1  # 1..9
        b = secrets.randbelow(9) + 1  # 1..9
        question = f"Combien font {a} + {b} ?"
        answer = str(a + b)
        expires_at = time.time() + self.captcha_ttl_seconds
        st.session_state['captcha_question'] = question
        st.session_state['captcha_expected_answer'] = answer
        st.session_state['captcha_expires_at'] = expires_at
        return question

    def ensure_captcha(self) -> str:
        """S'assure qu'un CAPTCHA valide existe et retourne la question"""
        question = st.session_state.get('captcha_question')
        expected = st.session_state.get('captcha_expected_answer')
        expires_at = st.session_state.get('captcha_expires_at', 0)
        if (not question) or (not expected) or time.time() >= float(expires_at):
            return self._generate_math_captcha()
        return question

    def rotate_captcha(self) -> str:
        """Force la génération d'un nouveau CAPTCHA et retourne la nouvelle question"""
        return self._generate_math_captcha()

    def verify_captcha(self, user_answer: str) -> bool:
        """Vérifie le CAPTCHA courant"""
        if not user_answer:
            return False
        expected = st.session_state.get('captcha_expected_answer')
        expires_at = st.session_state.get('captcha_expires_at', 0)
        question = st.session_state.get('captcha_question', '')

        # Fallback: tenter d'extraire la réponse depuis la question
        if (not expected) and question:
            try:
                import re
                m = re.search(r"(\d+)\s*\+\s*(\d+)", question)
                if m:
                    expected = str(int(m.group(1)) + int(m.group(2)))
            except Exception:
                expected = None

        # Vérifier expiration
        if not expected or time.time() >= float(expires_at):
            return False

        is_ok = str(user_answer).strip() == str(expected).strip()
        # Invalider le CAPTCHA utilisé (empêche réutilisation)
        if is_ok:
            st.session_state['captcha_question'] = None
            st.session_state['captcha_expected_answer'] = None
            st.session_state['captcha_expires_at'] = 0
        return is_ok
    
    def encrypt_data(self, data: str) -> bytes:
        """Chiffre des données sensibles"""
        return self.cipher.encrypt(data.encode())
    
    def decrypt_data(self, encrypted_data: bytes) -> str:
        """Déchiffre des données sensibles"""
        return self.cipher.decrypt(encrypted_data).decode()
    
    def sanitize_input(self, text: str) -> str:
        """Nettoie les entrées utilisateur pour éviter les injections"""
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}']
        for char in dangerous_chars:
            text = text.replace(char, '')
        return text.strip()
    
    def validate_email_format(self, email: str) -> bool:
        """Valide le format d'un email"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

# Instance globale du gestionnaire de sécurité
security_manager = SecurityManager()

def get_security_manager() -> SecurityManager:
    """Retourne l'instance du gestionnaire de sécurité"""
    return security_manager

# Fonctions utilitaires pour Streamlit
def secure_session_state():
    """Configure la session Streamlit de manière sécurisée"""
    if 'security_token' not in st.session_state:
        st.session_state.security_token = None
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = time.time()
    if 'user_permissions' not in st.session_state:
        st.session_state.user_permissions = []

def check_session_timeout(timeout_minutes: int = 30) -> bool:
    """Vérifie si la session a expiré"""
    if 'last_activity' not in st.session_state:
        return True
    
    current_time = time.time()
    last_activity = st.session_state.last_activity
    
    if current_time - last_activity > (timeout_minutes * 60):
        return True
    
    # Mettre à jour l'activité
    st.session_state.last_activity = current_time
    return False

def require_authentication():
    """Décorateur pour exiger l'authentification"""
    if not st.session_state.get('authenticated', False):
        st.error("🔒 Accès refusé. Veuillez vous connecter.")
        st.stop()
    return True

def require_permission(permission: str):
    """Vérifie si l'utilisateur a une permission spécifique"""
    if not st.session_state.get('authenticated', False):
        return False
    
    user_permissions = st.session_state.get('user_permissions', [])
    return permission in user_permissions 