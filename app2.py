import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Smart Irrigation - Berkane", 
    page_icon="🌱", 
    layout="wide"
)

# --- FONCTION DU BILAN HYDRIQUE (Algorithme du Système Expert) ---
def calculer_bilan_hydrique(etc_predite, pluie_prevue, stock_hier, rfu_max=100.0):
    """
    Calcule le niveau d'eau du sol pour le lendemain et détermine le besoin d'arrosage.
    """
    # 1. Équation de conservation : Entrées (Pluie) - Sorties (ETc prédite par le LSTM)
    nouveau_stock = stock_hier + pluie_prevue - etc_predite
    
    # 2. Limites physiques du sol (Le stock reste entre 0 et la capacité max)
    nouveau_stock = max(0.0, min(nouveau_stock, rfu_max))
    
    # 3. Déclenchement automatique de l'irrigation sous le seuil de confort (50% de la RFU)
    seuil_critique = rfu_max * 0.5
    
    SEUIL_CONFORT = rfu_max * 0.8
    besoin_irrigation = 0.0
    if nouveau_stock < seuil_critique:
        besoin_irrigation = SEUIL_CONFORT - nouveau_stock
        
    return nouveau_stock, besoin_irrigation

# --- CHARGEMENT SÉCURISÉ DES ASSETS IA ---
@st.cache_resource
def load_assets():
    # Chargement du modèle de Deep Learning LSTM (.h5)
    model = tf.keras.models.load_model('mon_modele_irrigation.h5')
    # Chargement du scaler pour la transformation des données (.pkl)
    scaler = joblib.load('scaler.pkl')
    return model, scaler

try:
    model, scaler = load_assets()
    assets_charges = True
except Exception as e:
    st.error(f"⚠️ Erreur lors du chargement des fichiers IA (`mon_modele_irrigation.h5` ou `scaler.pkl`) : {e}")
    assets_charges = False

# --- INTERFACE UTILISATEUR (Dashboard) ---
st.title("🚜 Système d'Aide à la Décision : Irrigation de Précision")
st.markdown("Ce tableau de bord combine un réseau de neurones récurrents **LSTM** et un modèle de bilan hydrique pour optimiser l'arrosage dans la plaine de la Moulouya.")

st.divider()

if assets_charges:
    # --- BARRE LATÉRALE : SAISIE DES PARAMÈTRES MÉTÉO ---
    st.sidebar.header("📊 Conditions Météorologiques du Jour")
    st.sidebar.write("Ajustez les curseurs pour simuler le climat actuel à Berkane.")

    tmax = st.sidebar.slider("Température Maximale (°C)", 10.0, 50.0, 25.0)
    tmin = st.sidebar.slider("Température Minimale (°C)", 0.0, 30.0, 12.0)
    wind = st.sidebar.slider("Vitesse Maximale du Vent (km/h)", 0.0, 60.0, 15.0)
    rad = st.sidebar.slider("Rayonnement Solaire (MJ/m²)", 5.0, 35.0, 20.0)
    pluie = st.sidebar.slider("Précipitations du Jour (Pluie en mm)", 0.0, 50.0, 0.0)

    # --- CORPS PRINCIPAL : AFFICHAGE EN DEUX COLONNES ---
    col1, col2 = st.columns(2)

    # --- COLONNE 1 : LE PRÉDICTEUR DE L'IA ---
    with col1:
        st.subheader("🔮 Prédiction de la Consommation de la Plante")
        
        # Préparation du tenseur (1 sequence, 10 jours d'historique, 7 features météo)
        # Pour la démo en temps réel, on injecte les curseurs sur la dernière étape
        dummy_input = np.zeros((1, 10, 7)) 
        dummy_input[0, -1, 0] = tmax 
        dummy_input[0, -1, 1] = rad
        dummy_input[0, -1, 4] = wind
        dummy_input[0, -1, 5] = tmin
        dummy_input[0, -1, 6] = pluie

        # Exécution de la prédiction via le modèle LSTM
        prediction_norm = model.predict(dummy_input, verbose=0)
        
        # Dé-normalisation exacte en utilisant les métadonnées réelles du scaler.pkl
        idx_cible = 6
        min_etc = scaler.data_min_[idx_cible]
        max_etc = scaler.data_max_[idx_cible]
        etc_predite = abs((prediction_norm[0][0] * (max_etc - min_etc)) + min_etc)
        
        # Affichage du résultat de l'IA
        st.metric(
            label="Évapotranspiration prédite ($ET_c$) pour demain", 
            value=f"{etc_predite:.2f} mm/jour"
        )
        st.info("💡 Cette valeur calculée par l'IA correspond au volume d'eau exact que la culture va perdre par évaporation et transpiration demain.")

    # --- COLONNE 2 : SYSTÈME EXPERT AGRONOMIQUE ---
    with col2:
        st.subheader("💧 Suivi du Sol & Recommandation")
        
        RU_MAX = 100.0  # Capacité maximale théorique de stockage du sol (en mm)
        
        # Curseur permettant à l'agriculteur d'indiquer l'état d'humidité estimé de la veille
        stock_hier = st.slider("Niveau d'eau présent dans le sol hier (mm)", 0.0, RU_MAX, 50.0)
        
        # Appel de l'algorithme de calcul du bilan hydrique
        stock_demain, dose_irrigation = calculer_bilan_hydrique(etc_predite, pluie, stock_hier, rfu_max=RU_MAX)
        
        # Affichage du stock d'eau résiduel projeté pour demain
        st.metric(label="Stock d'eau estimé dans le sol demain", value=f"{stock_demain:.2f} mm")
        
        # Prise de décision automatique et affichage du verdict
        if dose_irrigation > 0:
            st.error(f"⚠️ ALERTE CRITIQUE : Le stock descend sous le seuil de confort ! Vous devez appliquer une dose d'irrigation de **{dose_irrigation:.2f} mm** pour saturer à nouveau le sol.")
        else:
            st.success("✅ SITUATION CONFORTABLE : Les réserves du sol sont suffisantes pour couvrir les besoins de demain. Aucune irrigation requise.")
            
        # Visualisation graphique de la jauge sous forme de barre de progression
        st.write("Jauge du réservoir d'eau du sol :")
        st.progress(stock_demain / RU_MAX)

    st.divider()
    st.caption("📍 **Zone d'application pilote :** Périmètre d'irrigation de la Basse Moulouya, Berkane, Maroc. Année universitaire : 2025-2026.")

else:
    st.warning("Veuillez placer vos fichiers `mon_modele_irrigation.h5` et `scaler.pkl` dans le même dossier que ce script pour lancer l'interface.")