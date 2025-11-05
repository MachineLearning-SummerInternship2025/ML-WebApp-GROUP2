#!/usr/bin/env python3
"""
Script de démarrage rapide pour ML Model Selector v2.0
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Configure l'environnement pour le démarrage"""
    print("🔧 Configuration de l'environnement...")
    
    # Vérifier si .env existe, sinon le créer
    env_file = Path('.env')
    env_default = Path('env_default.txt')
    
    if not env_file.exists() and env_default.exists():
        print("📄 Création du fichier .env...")
        with open(env_default, 'r') as src:
            content = src.read()
        with open('.env', 'w') as dst:
            dst.write(content)
        print("✅ Fichier .env créé")
    elif env_file.exists():
        print("✅ Fichier .env déjà présent")
    else:
        print("⚠️ Aucun fichier de configuration trouvé")
    
    # Vérifier les dépendances
    print("📦 Vérification des dépendances...")
    try:
        import streamlit
        print("✅ Streamlit disponible")
    except ImportError:
        print("❌ Streamlit non installé. Installation...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'streamlit'], check=True)
    
    try:
        import bcrypt
        print("✅ bcrypt disponible")
    except ImportError:
        print("❌ bcrypt non installé. Installation...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'bcrypt'], check=True)
    
    try:
        import cryptography
        print("✅ cryptography disponible")
    except ImportError:
        print("❌ cryptography non installé. Installation...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'cryptography'], check=True)
    
    try:
        import passlib
        print("✅ passlib disponible")
    except ImportError:
        print("❌ passlib non installé. Installation...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'passlib'], check=True)
    
    try:
        import dotenv
        print("✅ python-dotenv disponible")
    except ImportError:
        print("❌ python-dotenv non installé. Installation...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-dotenv'], check=True)

def test_modules():
    """Teste les nouveaux modules"""
    print("\n🧪 Test des modules de sécurité et cloud...")
    
    try:
        from security import get_security_manager
        security = get_security_manager()
        print("✅ Module de sécurité chargé")
        
        # Test rapide
        is_valid, message = security.validate_password_strength("Test123!")
        if is_valid:
            print("✅ Validation des mots de passe fonctionne")
        else:
            print("❌ Problème avec la validation des mots de passe")
            
    except Exception as e:
        print(f"❌ Erreur module sécurité: {e}")
    
    try:
        from cloud_storage import get_cloud_storage
        cloud = get_cloud_storage()
        print("✅ Module de stockage cloud chargé")
        
        if cloud.is_configured:
            print("☁️ Configuration AWS S3 détectée")
        else:
            print("💾 Mode stockage local")
            
    except Exception as e:
        print(f"❌ Erreur module cloud: {e}")
    
    try:
        from config import get_config
        config = get_config()
        print("✅ Module de configuration chargé")
        print(f"   Version: {config.APP_VERSION}")
        print(f"   Production: {config.is_production()}")
        
    except Exception as e:
        print(f"❌ Erreur module configuration: {e}")

def start_application():
    """Démarre l'application"""
    print("\n🚀 Démarrage de ML Model Selector v2.0...")
    print("=" * 50)
    print("🔒 Sécurité renforcée activée")
    print("☁️ Support cloud disponible")
    print("📊 Interface ML avancée")
    print("=" * 50)
    
    # Démarrer Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'app.py',
            '--server.port', '8501',
            '--server.address', 'localhost'
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application arrêtée par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")

def main():
    """Fonction principale"""
    print("🤖 ML Model Selector v2.0 - Démarrage")
    print("=" * 50)
    
    # Configuration
    setup_environment()
    
    # Test des modules
    test_modules()
    
    # Démarrer l'application
    start_application()

if __name__ == "__main__":
    main() 