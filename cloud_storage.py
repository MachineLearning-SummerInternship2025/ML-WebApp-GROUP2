import os
import boto3
import tempfile
from typing import Optional, Dict, List, Tuple
from botocore.exceptions import ClientError, NoCredentialsError
import streamlit as st
from datetime import datetime, timedelta
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CloudStorageManager:
    def __init__(self):
        """Initialise le gestionnaire de stockage cloud"""
        self.aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        self.s3_bucket = os.getenv('AWS_S3_BUCKET')
        
        # Configuration S3
        self.s3_client = None
        self.s3_resource = None
        self.is_configured = False
        
        # Initialiser la connexion S3 si les credentials sont disponibles
        self._initialize_s3()
    
    def _initialize_s3(self):
        """Initialise la connexion S3"""
        if self.aws_access_key and self.aws_secret_key and self.s3_bucket:
            try:
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                self.s3_resource = boto3.resource(
                    's3',
                    aws_access_key_id=self.aws_access_key,
                    aws_secret_access_key=self.aws_secret_key,
                    region_name=self.aws_region
                )
                
                # Vérifier que le bucket existe
                self.s3_client.head_bucket(Bucket=self.s3_bucket)
                self.is_configured = True
                logger.info(f"S3 configuré avec succès pour le bucket: {self.s3_bucket}")
                
            except (ClientError, NoCredentialsError) as e:
                logger.warning(f"Impossible de configurer S3: {e}")
                self.is_configured = False
        else:
            logger.info("Configuration S3 non trouvée, utilisation du stockage local uniquement")
            self.is_configured = False
    
    def upload_dataset(self, file_data: bytes, filename: str, username: str, 
                      metadata: Optional[Dict] = None) -> Tuple[bool, str]:
        """Upload un dataset vers S3"""
        if not self.is_configured:
            return False, "S3 non configuré"
        
        try:
            # Créer la clé S3
            s3_key = f"datasets/{username}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            
            # Métadonnées par défaut
            default_metadata = {
                'username': username,
                'upload_date': datetime.now().isoformat(),
                'file_size': str(len(file_data)),
                'original_filename': filename
            }
            
            if metadata:
                default_metadata.update(metadata)
            
            # Upload vers S3
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=file_data,
                Metadata=default_metadata,
                ContentType='application/octet-stream'
            )
            
            logger.info(f"Dataset uploadé avec succès: {s3_key}")
            return True, s3_key
            
        except Exception as e:
            logger.error(f"Erreur lors de l'upload S3: {e}")
            return False, str(e)
    
    def download_dataset(self, s3_key: str) -> Tuple[bool, bytes]:
        """Télécharge un dataset depuis S3"""
        if not self.is_configured:
            return False, b""
        
        try:
            response = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            file_data = response['Body'].read()
            logger.info(f"Dataset téléchargé avec succès: {s3_key}")
            return True, file_data
            
        except Exception as e:
            logger.error(f"Erreur lors du téléchargement S3: {e}")
            return False, b""
    
    def delete_dataset(self, s3_key: str) -> Tuple[bool, str]:
        """Supprime un dataset de S3"""
        if not self.is_configured:
            return False, "S3 non configuré"
        
        try:
            self.s3_client.delete_object(Bucket=self.s3_bucket, Key=s3_key)
            logger.info(f"Dataset supprimé avec succès: {s3_key}")
            return True, "Supprimé avec succès"
            
        except Exception as e:
            logger.error(f"Erreur lors de la suppression S3: {e}")
            return False, str(e)
    
    def list_user_datasets(self, username: str) -> List[Dict]:
        """Liste tous les datasets d'un utilisateur sur S3"""
        if not self.is_configured:
            return []
        
        try:
            prefix = f"datasets/{username}/"
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix
            )
            
            datasets = []
            for obj in response.get('Contents', []):
                # Récupérer les métadonnées
                try:
                    metadata_response = self.s3_client.head_object(
                        Bucket=self.s3_bucket,
                        Key=obj['Key']
                    )
                    metadata = metadata_response.get('Metadata', {})
                except:
                    metadata = {}
                
                datasets.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'],
                    'metadata': metadata
                })
            
            return datasets
            
        except Exception as e:
            logger.error(f"Erreur lors de la liste S3: {e}")
            return []
    
    def backup_database(self, db_path: str) -> Tuple[bool, str]:
        """Sauvegarde la base de données locale vers S3"""
        if not self.is_configured:
            return False, "S3 non configuré"
        
        try:
            # Lire le fichier de base de données
            with open(db_path, 'rb') as f:
                db_data = f.read()
            
            # Créer la clé de sauvegarde
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"backups/database_{timestamp}.db"
            
            # Upload de la sauvegarde
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=db_data,
                Metadata={
                    'backup_type': 'database',
                    'backup_date': timestamp,
                    'file_size': str(len(db_data))
                }
            )
            
            logger.info(f"Base de données sauvegardée: {s3_key}")
            return True, s3_key
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde: {e}")
            return False, str(e)
    
    def restore_database(self, s3_key: str, local_path: str) -> Tuple[bool, str]:
        """Restaure la base de données depuis S3"""
        if not self.is_configured:
            return False, "S3 non configuré"
        
        try:
            # Télécharger la sauvegarde
            success, db_data = self.download_dataset(s3_key)
            if not success:
                return False, "Impossible de télécharger la sauvegarde"
            
            # Écrire le fichier local
            with open(local_path, 'wb') as f:
                f.write(db_data)
            
            logger.info(f"Base de données restaurée depuis: {s3_key}")
            return True, "Restauration réussie"
            
        except Exception as e:
            logger.error(f"Erreur lors de la restauration: {e}")
            return False, str(e)
    
    def get_storage_stats(self) -> Dict:
        """Retourne les statistiques de stockage"""
        if not self.is_configured:
            return {"error": "S3 non configuré"}
        
        try:
            # Calculer la taille totale des datasets
            total_size = 0
            total_files = 0
            
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket)
            for obj in response.get('Contents', []):
                total_size += obj['Size']
                total_files += 1
            
            return {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "total_files": total_files,
                "bucket_name": self.s3_bucket,
                "region": self.aws_region
            }
            
        except Exception as e:
            logger.error(f"Erreur lors du calcul des stats: {e}")
            return {"error": str(e)}
    
    def cleanup_old_backups(self, days_to_keep: int = 30) -> Tuple[bool, str]:
        """Nettoie les anciennes sauvegardes"""
        if not self.is_configured:
            return False, "S3 non configuré"
        
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix="backups/"
            )
            
            for obj in response.get('Contents', []):
                if obj['LastModified'] < cutoff_date:
                    self.s3_client.delete_object(
                        Bucket=self.s3_bucket,
                        Key=obj['Key']
                    )
                    deleted_count += 1
            
            logger.info(f"Anciennes sauvegardes supprimées: {deleted_count}")
            return True, f"{deleted_count} sauvegardes supprimées"
            
        except Exception as e:
            logger.error(f"Erreur lors du nettoyage: {e}")
            return False, str(e)

# Instance globale du gestionnaire de stockage cloud
cloud_storage = CloudStorageManager()

def get_cloud_storage() -> CloudStorageManager:
    """Retourne l'instance du gestionnaire de stockage cloud"""
    return cloud_storage

def is_cloud_available() -> bool:
    """Vérifie si le stockage cloud est disponible"""
    return cloud_storage.is_configured

def get_storage_info() -> Dict:
    """Retourne les informations de stockage"""
    if is_cloud_available():
        return cloud_storage.get_storage_stats()
    else:
        return {
            "storage_type": "local",
            "message": "Stockage local uniquement - configurez S3 pour le cloud"
        } 