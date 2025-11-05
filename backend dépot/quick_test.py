#!/usr/bin/env python3
"""
Test rapide pour vérifier le fonctionnement de l'upload
"""

import streamlit as st
import pandas as pd
import tempfile
import os

def test_upload():
    """Test simple d'upload de fichier"""
    st.title("🧪 Test d'Upload - ML Model Selector v2.0")
    
    st.info("Ce test vérifie que l'upload de fichiers fonctionne correctement")
    
    # Test d'upload
    uploaded_file = st.file_uploader(
        "📁 Testez l'upload d'un fichier CSV",
        type=["csv"],
        help="Sélectionnez un fichier CSV pour tester"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Fichier uploadé avec succès: {uploaded_file.name}")
        st.info(f"📊 Taille: {len(uploaded_file.getvalue())} bytes")
        
        try:
            # Lire le fichier
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Fichier lu avec succès: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Afficher un aperçu
            st.subheader("📋 Aperçu des données")
            st.dataframe(df.head(), use_container_width=True)
            
            # Informations sur les colonnes
            st.subheader("🔍 Informations sur les colonnes")
            col_info = []
            for col in df.columns:
                dtype = df[col].dtype
                uniques = df[col].nunique()
                null_count = df[col].isnull().sum()
                
                col_info.append({
                    "Colonne": col,
                    "Type": str(dtype),
                    "Valeurs uniques": uniques,
                    "Valeurs manquantes": null_count
                })
            
            st.dataframe(pd.DataFrame(col_info), use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier: {e}")
    
    # Test des modules de sécurité
    st.markdown("---")
    st.subheader("🔒 Test des modules de sécurité")
    
    try:
        from security import get_security_manager
        security = get_security_manager()
        st.success("✅ Module de sécurité chargé")
        
        # Test de validation de mot de passe
        test_password = st.text_input("Testez un mot de passe:", value="Test123!", type="password")
        if test_password:
            is_valid, message = security.validate_password_strength(test_password)
            if is_valid:
                st.success(f"✅ Mot de passe valide: {message}")
            else:
                st.warning(f"⚠️ Mot de passe invalide: {message}")
                
    except Exception as e:
        st.error(f"❌ Erreur module sécurité: {e}")
    
    # Test des modules cloud
    st.markdown("---")
    st.subheader("☁️ Test des modules cloud")
    
    try:
        from cloud_storage import get_cloud_storage
        cloud = get_cloud_storage()
        st.success("✅ Module cloud chargé")
        
        if cloud.is_configured:
            st.success("☁️ Configuration AWS S3 détectée")
        else:
            st.info("💾 Mode stockage local (normal sans AWS)")
            
    except Exception as e:
        st.error(f"❌ Erreur module cloud: {e}")
    
    # Instructions
    st.markdown("---")
    st.subheader("📋 Instructions")
    st.markdown("""
    1. **Testez l'upload** : Sélectionnez un fichier CSV ci-dessus
    2. **Vérifiez la lecture** : Le fichier doit s'afficher correctement
    3. **Testez la sécurité** : Entrez un mot de passe pour tester la validation
    4. **Vérifiez le cloud** : L'état du stockage cloud s'affiche
    
    Si tout fonctionne, votre application est prête !
    """)

if __name__ == "__main__":
    test_upload() 