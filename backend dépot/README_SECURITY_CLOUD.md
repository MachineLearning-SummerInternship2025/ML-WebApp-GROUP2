# 🔒 Sécurité & ☁️ Cloud - ML Model Selector

## 🚀 Nouveautés de la version 2.0

Cette version apporte des améliorations majeures en matière de **sécurité** et d'**intégration cloud** pour votre application ML Model Selector.

## 🔒 Améliorations de Sécurité

### 1. Gestionnaire de Sécurité Avancé (`security.py`)

#### Fonctionnalités principales :
- **Validation des mots de passe forts** : Vérification automatique de la complexité
- **Protection contre les attaques par force brute** : Limitation des tentatives de connexion
- **Rate limiting** : Protection contre le spam et les attaques DDoS
- **Chiffrement des données sensibles** : Utilisation de Fernet (cryptography)
- **Gestion sécurisée des sessions** : Tokens de session sécurisés
- **Sanitisation des entrées** : Protection contre les injections

#### Configuration de sécurité :
```python
# Variables d'environnement
MAX_LOGIN_ATTEMPTS=5          # Nombre max de tentatives de connexion
LOGIN_TIMEOUT=300            # Timeout en secondes
PASSWORD_MIN_LENGTH=8        # Longueur minimale du mot de passe
RATE_LIMIT_REQUESTS=100      # Limite de requêtes par heure
RATE_LIMIT_WINDOW=3600       # Fenêtre de temps en secondes
```

### 2. Validation des Mots de Passe

Le système vérifie automatiquement que les mots de passe contiennent :
- Au moins 8 caractères
- Au moins une majuscule
- Au moins une minuscule
- Au moins un chiffre
- Au moins un caractère spécial

### 3. Protection des Sessions

- **Timeout automatique** : Sessions expirées après inactivité
- **Tokens sécurisés** : Génération de tokens uniques et sécurisés
- **Gestion des permissions** : Système de rôles et permissions

## ☁️ Intégration Cloud

### 1. Gestionnaire de Stockage Cloud (`cloud_storage.py`)

#### Support AWS S3 :
- **Upload automatique** des datasets vers S3
- **Sauvegarde automatique** de la base de données
- **Gestion des métadonnées** pour chaque fichier
- **Récupération et restauration** depuis le cloud
- **Nettoyage automatique** des anciennes sauvegardes

#### Configuration AWS :
```bash
# Variables d'environnement requises
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_DEFAULT_REGION=us-east-1
AWS_S3_BUCKET=your-s3-bucket-name
```

### 2. Configuration Centralisée (`config.py`)

- **Gestion des variables d'environnement** via `.env`
- **Validation automatique** de la configuration
- **Détection de l'environnement** (dev/prod)
- **Configuration flexible** pour différents déploiements

### 3. Déploiement Automatisé (`deploy_cloud.py`)

#### Support de déploiement :
- **Docker local** : Déploiement avec docker-compose
- **AWS ECS** : Déploiement sur Amazon ECS
- **Terraform** : Infrastructure as Code
- **GitHub Actions** : CI/CD automatisé
- **Nginx** : Reverse proxy avec SSL

## 🛠️ Installation et Configuration

### 1. Installation des Dépendances

```bash
pip install -r requirements.txt
```

### 2. Configuration des Variables d'Environnement

Copiez le fichier `env_example.txt` vers `.env` et configurez :

```bash
cp env_example.txt .env
# Éditez .env avec vos valeurs
```

### 3. Configuration AWS (Optionnel)

Si vous souhaitez utiliser le stockage cloud :

```bash
# Créez un bucket S3
aws s3 mb s3://votre-bucket-name

# Configurez les permissions
aws s3api put-bucket-versioning --bucket votre-bucket-name --versioning-configuration Status=Enabled
```

## 🚀 Déploiement

### Déploiement Local avec Docker

```bash
python deploy_cloud.py --target local
```

### Déploiement sur AWS

```bash
python deploy_cloud.py --target aws
```

### Déploiement Complet

```bash
python deploy_cloud.py --target all
```

## 📋 Utilisation

### 1. Sécurité Renforcée

L'application utilise automatiquement les nouvelles fonctionnalités de sécurité :

```python
from security import require_authentication, require_permission

@require_authentication
def protected_function():
    # Cette fonction nécessite une authentification
    pass

if require_permission("admin"):
    # Code réservé aux administrateurs
    pass
```

### 2. Stockage Cloud

```python
from cloud_storage import get_cloud_storage

cloud = get_cloud_storage()

# Upload d'un dataset
success, s3_key = cloud.upload_dataset(
    file_data=file_bytes,
    filename="dataset.csv",
    username="user123"
)

# Sauvegarde de la base
success, backup_key = cloud.backup_database("users.db")
```

### 3. Configuration

```python
from config import get_config

config = get_config()
if config.is_production():
    print("Mode production activé")
if config.is_cloud_configured():
    print("Stockage cloud disponible")
```

## 🔧 Maintenance et Monitoring

### 1. Sauvegardes Automatiques

- **Sauvegarde quotidienne** de la base de données
- **Rétention configurable** des sauvegardes
- **Nettoyage automatique** des anciens fichiers

### 2. Logs et Monitoring

- **Logs structurés** pour le debugging
- **Métriques de performance** AWS CloudWatch
- **Alertes automatiques** en cas de problème

### 3. Mise à Jour

```bash
# Mise à jour de l'application
git pull origin main
pip install -r requirements.txt

# Redéploiement
python deploy_cloud.py --target local
```

## 🚨 Sécurité en Production

### 1. Variables d'Environnement

⚠️ **IMPORTANT** : Changez les clés par défaut en production !

```bash
# Générer des clés sécurisées
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Configuration du Serveur

```bash
# Désactiver le mode debug
DEBUG=False

# Utiliser HTTPS
HOST=0.0.0.0
PORT=443
```

### 3. Firewall et Réseau

- **Limiter l'accès** aux ports nécessaires
- **Utiliser un VPN** pour l'accès administrateur
- **Configurer des alertes** de sécurité

## 📚 Documentation Supplémentaire

### AWS S3
- [Documentation officielle AWS S3](https://docs.aws.amazon.com/s3/)
- [Meilleures pratiques de sécurité](https://docs.aws.amazon.com/s3/latest/dev/security-best-practices.html)

### Docker
- [Documentation Docker](https://docs.docker.com/)
- [Docker Security](https://docs.docker.com/engine/security/)

### Terraform
- [Documentation Terraform](https://www.terraform.io/docs)
- [AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)

## 🆘 Support et Dépannage

### Problèmes Courants

1. **Erreur de connexion AWS** : Vérifiez vos credentials
2. **Timeout de session** : Augmentez `SESSION_TIMEOUT`
3. **Rate limiting** : Ajustez `RATE_LIMIT_REQUESTS`

### Logs et Debug

```bash
# Vérifier la configuration
python config.py

# Tester la connexion AWS
python -c "import boto3; print(boto3.client('sts').get_caller_identity())"

# Vérifier les logs
tail -f app.log
```

## 🎯 Prochaines Étapes

- [ ] **Monitoring avancé** avec Prometheus/Grafana
- [ ] **Backup multi-région** AWS
- [ ] **Authentification 2FA** avec TOTP
- [ ] **Intégration OAuth** (Google, GitHub)
- [ ] **Support multi-cloud** (Azure, GCP)

---

**Version** : 2.0.0  
**Dernière mise à jour** : Décembre 2024  
**Mainteneur** : Équipe ML Model Selector 