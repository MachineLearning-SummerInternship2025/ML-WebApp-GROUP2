import os
from dotenv import load_dotenv
from typing import Dict, Any

# Charger les variables d'environnement depuis .env
load_dotenv()

class Config:
    """Configuration centralisée de l'application"""
    
    # Configuration de base
    APP_NAME = "ML Model Selector"
    APP_VERSION = "2.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Configuration de sécurité
    SECRET_KEY = os.getenv('SECRET_KEY', 'default-secret-key-change-in-production')
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'default-jwt-secret-change-in-production')
    SESSION_TIMEOUT = int(os.getenv('SESSION_TIMEOUT', 3600))  # 1 heure
    
    # Configuration de la base de données
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///users.db')
    
    # Configuration de sécurité avancée
    MAX_LOGIN_ATTEMPTS = int(os.getenv('MAX_LOGIN_ATTEMPTS', 5))
    LOGIN_TIMEOUT = int(os.getenv('LOGIN_TIMEOUT', 300))  # 5 minutes
    PASSWORD_MIN_LENGTH = int(os.getenv('PASSWORD_MIN_LENGTH', 8))
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 3600))  # 1 heure
    
    # Configuration AWS S3
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')
    
    # Configuration du serveur
    HOST = os.getenv('HOST', '0.0.0.0')
    PORT = int(os.getenv('PORT', 8501))
    
    # Configuration des modèles ML
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
    SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
    MAX_ROWS_PREVIEW = 1000
    
    # Configuration des sauvegardes
    AUTO_BACKUP_ENABLED = os.getenv('AUTO_BACKUP_ENABLED', 'True').lower() == 'true'
    BACKUP_INTERVAL_HOURS = int(os.getenv('BACKUP_INTERVAL_HOURS', 24))
    BACKUP_RETENTION_DAYS = int(os.getenv('BACKUP_RETENTION_DAYS', 30))
    
    @classmethod
    def get_all_config(cls) -> Dict[str, Any]:
        """Retourne toute la configuration sous forme de dictionnaire"""
        return {
            'app_name': cls.APP_NAME,
            'app_version': cls.APP_VERSION,
            'debug': cls.DEBUG,
            'database_url': cls.DATABASE_URL,
            'security': {
                'max_login_attempts': cls.MAX_LOGIN_ATTEMPTS,
                'login_timeout': cls.LOGIN_TIMEOUT,
                'password_min_length': cls.PASSWORD_MIN_LENGTH,
                'rate_limit_requests': cls.RATE_LIMIT_REQUESTS,
                'rate_limit_window': cls.RATE_LIMIT_WINDOW
            },
            'aws': {
                'access_key_id': cls.AWS_ACCESS_KEY_ID,
                'secret_access_key': '***HIDDEN***' if cls.AWS_SECRET_ACCESS_KEY else None,
                'region': cls.AWS_DEFAULT_REGION,
                's3_bucket': cls.AWS_S3_BUCKET
            },
            'server': {
                'host': cls.HOST,
                'port': cls.PORT
            },
            'backup': {
                'auto_backup_enabled': cls.AUTO_BACKUP_ENABLED,
                'backup_interval_hours': cls.BACKUP_INTERVAL_HOURS,
                'backup_retention_days': cls.BACKUP_RETENTION_DAYS
            }
        }
    
    @classmethod
    def is_production(cls) -> bool:
        """Vérifie si l'application est en production"""
        return not cls.DEBUG and cls.SECRET_KEY != 'default-secret-key-change-in-production'
    
    @classmethod
    def is_cloud_configured(cls) -> bool:
        """Vérifie si la configuration cloud est disponible"""
        return all([
            cls.AWS_ACCESS_KEY_ID,
            cls.AWS_SECRET_ACCESS_KEY,
            cls.AWS_S3_BUCKET
        ])
    
    @classmethod
    def validate_config(cls) -> Dict[str, str]:
        """Valide la configuration et retourne les erreurs"""
        errors = {}
        
        # Vérifications de sécurité
        if cls.SECRET_KEY == 'default-secret-key-change-in-production':
            errors['secret_key'] = 'SECRET_KEY doit être changé en production'
        
        if cls.JWT_SECRET_KEY == 'default-jwt-secret-change-in-production':
            errors['jwt_secret_key'] = 'JWT_SECRET_KEY doit être changé en production'
        
        if cls.PASSWORD_MIN_LENGTH < 8:
            errors['password_min_length'] = 'PASSWORD_MIN_LENGTH doit être >= 8'
        
        # Vérifications de la base de données
        if not cls.DATABASE_URL:
            errors['database_url'] = 'DATABASE_URL est requis'
        
        # Vérifications AWS (optionnelles)
        if cls.AWS_ACCESS_KEY_ID and not cls.AWS_SECRET_ACCESS_KEY:
            errors['aws'] = 'AWS_SECRET_ACCESS_KEY requis si AWS_ACCESS_KEY_ID est défini'
        
        if cls.AWS_SECRET_ACCESS_KEY and not cls.AWS_ACCESS_KEY_ID:
            errors['aws'] = 'AWS_ACCESS_KEY_ID requis si AWS_SECRET_ACCESS_KEY est défini'
        
        if (cls.AWS_ACCESS_KEY_ID and cls.AWS_SECRET_ACCESS_KEY) and not cls.AWS_S3_BUCKET:
            errors['aws'] = 'AWS_S3_BUCKET requis si AWS est configuré'
        
        return errors

# Instance globale de configuration
config = Config()

def get_config() -> Config:
    """Retourne l'instance de configuration"""
    return config

def validate_environment() -> bool:
    """Valide l'environnement et affiche les erreurs"""
    errors = config.validate_config()
    
    if errors:
        print("❌ Erreurs de configuration détectées:")
        for key, message in errors.items():
            print(f"   - {key}: {message}")
        return False
    
    print("✅ Configuration valide")
    return True

if __name__ == "__main__":
    # Test de la configuration
    print("🔧 Configuration de l'application:")
    print(f"   Nom: {config.APP_NAME}")
    print(f"   Version: {config.APP_VERSION}")
    print(f"   Mode debug: {config.DEBUG}")
    print(f"   Production: {config.is_production()}")
    print(f"   Cloud configuré: {config.is_cloud_configured()}")
    
    print("\n📋 Configuration complète:")
    import json
    print(json.dumps(config.get_all_config(), indent=2, default=str))
    
    print("\n🔍 Validation de l'environnement:")
    validate_environment() 