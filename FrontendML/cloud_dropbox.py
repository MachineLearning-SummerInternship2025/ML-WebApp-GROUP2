import dropbox
import os
from datetime import datetime
import tempfile

class DropboxCloudStorage:
    def __init__(self, access_token):
        # Configuration simple et rapide
        self.dbx = dropbox.Dropbox(access_token, timeout=15)
        self.folder = "/ml-selector"
        self.is_configured = True
    
    def upload_file(self, local_path, remote_path):
        """Upload un fichier vers Dropbox"""
        try:
            # Vérifier que le fichier existe
            if not os.path.exists(local_path):
                return False, f"Fichier local introuvable: {local_path}"
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(local_path)
            if file_size == 0:
                return False, "Fichier vide"
            
            # Pour les gros fichiers (> 50MB), utiliser l'upload par chunks
            if file_size > 50 * 1024 * 1024:  # 50MB
                return self._upload_large_file(local_path, remote_path)
            else:
                # Upload simple pour les petits fichiers
                with open(local_path, 'rb') as f:
                    self.dbx.files_upload(f.read(), f"{self.folder}{remote_path}")
                return True, f"Fichier uploadé vers {remote_path}"
            
        except Exception as e:
            return False, f"Erreur upload: {e}"
    
    def _upload_large_file(self, local_path, remote_path):
        """Upload de gros fichiers par chunks"""
        try:
            chunk_size = 4 * 1024 * 1024  # 4MB chunks
            file_size = os.path.getsize(local_path)
            
            # Commencer l'upload par chunks
            with open(local_path, 'rb') as f:
                # Premier chunk
                first_chunk = f.read(chunk_size)
                upload_session_start_result = self.dbx.files_upload_session_start(first_chunk)
                cursor = dropbox.files.UploadSessionCursor(
                    session_id=upload_session_start_result.session_id,
                    offset=f.tell()
                )
                
                # Chunks suivants
                while f.tell() < file_size:
                    chunk = f.read(chunk_size)
                    if f.tell() == file_size:
                        # Dernier chunk
                        commit = dropbox.files.CommitInfo(path=f"{self.folder}{remote_path}")
                        self.dbx.files_upload_session_finish(chunk, cursor, commit)
                    else:
                        # Chunk intermédiaire
                        self.dbx.files_upload_session_append_v2(chunk, cursor)
                        cursor.offset = f.tell()
            
            return True, f"Fichier volumineux uploadé vers {remote_path}"
            
        except Exception as e:
            return False, f"Erreur upload gros fichier: {e}"
    
    def download_file(self, remote_path, local_path):
        """Download un fichier depuis Dropbox"""
        try:
            metadata, response = self.dbx.files_download(f"{self.folder}{remote_path}")
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True, "Fichier téléchargé"
        except Exception as e:
            return False, str(e)
    
    def backup_database(self, db_path):
        """Sauvegarde la base de données"""
        try:
            # Vérifier que la base existe
            if not os.path.exists(db_path):
                return False, f"Base de données introuvable: {db_path}"
            
            # Vérifier la taille du fichier
            file_size = os.path.getsize(db_path)
            if file_size == 0:
                return False, "Base de données vide"
            
            # Créer le dossier backups s'il n'existe pas
            try:
                self.dbx.files_create_folder_v2(f"{self.folder}/backups")
            except Exception as e:
                if "already exists" not in str(e).lower():
                    pass  # Le dossier existe déjà
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            remote_path = f"/backups/users_{timestamp}.db"
            
            # Upload avec gestion d'erreur améliorée
            return self.upload_file(db_path, remote_path)
            
        except Exception as e:
            return False, f"Erreur lors de la sauvegarde: {e}"
    
    def list_files(self):
        """Liste tous les fichiers"""
        try:
            files = []
            result = self.dbx.files_list_folder(self.folder)
            for entry in result.entries:
                files.append({
                    'name': entry.name,
                    'size': entry.size,
                    'modified': entry.server_modified
                })
            return True, files
        except Exception as e:
            return False, str(e)
    
    def get_storage_info(self):
        """Informations sur l'espace de stockage"""
        try:
            account = self.dbx.users_get_current_account()
            files = self.list_files()
            total_size = sum(file['size'] for file in files[1]) if files[0] else 0
            return {
                'storage_type': 'dropbox',
                'total_files': len(files[1]) if files[0] else 0,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'account_name': account.name.display_name,
                'email': account.email
            }
        except Exception as e:
            return {
                'storage_type': 'local',
                'error': str(e)
            }
    
    def cleanup_old_backups(self, days_to_keep=30):
        """Nettoie les anciennes sauvegardes"""
        try:
            files = self.list_files()
            if not files[0]:
                return False, "Impossible de lister les fichiers"
            
            backup_files = [f for f in files[1] if f['name'].startswith('users_') and f['name'].endswith('.db')]
            current_time = datetime.now()
            
            deleted_count = 0
            for file in backup_files:
                file_age = (current_time - file['modified']).days
                if file_age > days_to_keep:
                    try:
                        self.dbx.files_delete_v2(f"{self.folder}/backups/{file['name']}")
                        deleted_count += 1
                    except:
                        pass
            
            return True, f"{deleted_count} sauvegardes supprimées"
        except Exception as e:
            return False, str(e) 