#!/usr/bin/env python3
"""
Création d'une nouvelle base de données propre
"""

import sqlite3
import os
import shutil
from datetime import datetime

def create_clean_database():
    """Crée une nouvelle base de données propre"""
    print("🚀 Création d'une nouvelle base de données propre")
    print("=" * 50)
    
    if not os.path.exists("users.db"):
        print("❌ Base de données introuvable")
        return False
    
    try:
        # Sauvegarde de l'ancienne base
        backup_name = f"users_old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        shutil.copy2("users.db", backup_name)
        print(f"💾 Ancienne base sauvegardée: {backup_name}")
        
        # Taille de l'ancienne base
        old_size = os.path.getsize("users.db")
        print(f"📁 Taille ancienne base: {old_size:,} bytes ({old_size / (1024*1024):.1f} MB)")
        
        # Récupérer les utilisateurs de l'ancienne base
        print("👥 Récupération des utilisateurs...")
        old_conn = sqlite3.connect("users.db")
        old_cursor = old_conn.cursor()
        
        # Récupérer tous les utilisateurs
        old_cursor.execute("SELECT username, password, name FROM users")
        users = old_cursor.fetchall()
        print(f"✅ {len(users)} utilisateurs récupérés")
        
        old_conn.close()
        
        # Supprimer l'ancienne base
        os.remove("users.db")
        print("🗑️ Ancienne base supprimée")
        
        # Créer une nouvelle base propre
        print("🔧 Création de la nouvelle base...")
        new_conn = sqlite3.connect("users.db")
        new_cursor = new_conn.cursor()
        
        # Créer la table users (structure originale)
        new_cursor.execute("""
            CREATE TABLE users (
                username TEXT PRIMARY KEY,
                password TEXT,
                name TEXT
            )
        """)
        
        # Créer la table datasets (vide, structure originale)
        new_cursor.execute("""
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                name TEXT,
                content BLOB,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insérer les utilisateurs
        for user in users:
            new_cursor.execute(
                "INSERT INTO users (username, password, name) VALUES (?, ?, ?)",
                user
            )
        
        # Commit et fermer
        new_conn.commit()
        new_conn.close()
        
        # Vérifier la nouvelle taille
        new_size = os.path.getsize("users.db")
        space_saved = old_size - new_size
        
        print(f"✅ Nouvelle base créée!")
        print(f"📁 Nouvelle taille: {new_size:,} bytes ({new_size / (1024*1024):.1f} MB)")
        print(f"💾 Espace récupéré: {space_saved:,} bytes ({space_saved / (1024*1024):.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    # Confirmation
    print("⚠️ ATTENTION: Ce script va:")
    print("   - Sauvegarder votre base actuelle")
    print("   - Créer une nouvelle base propre")
    print("   - Conserver vos utilisateurs")
    print("   - Supprimer toutes les données volumineuses")
    
    response = input("\nContinuer ? (oui/non): ")
    
    if response.lower() == "oui":
        if create_clean_database():
            print("\n🎉 Base de données nettoyée avec succès!")
        else:
            print("\n❌ Échec de la création")
    else:
        print("❌ Opération annulée")
