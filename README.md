# 🌾 Système Intelligent d'Aide à la Décision pour l'Irrigation Connectée

Ce projet implémente une approche hybride combinant le **Deep Learning** et l'**agronomie computationnelle** pour optimiser la gestion des ressources hydriques dans la plaine de la Moulouya (Berkane). L'objectif est d'anticiper les besoins en eau des cultures pour éviter le stress hydrique tout en éliminant le gaspillage d'eau.

L'application est accessible en ligne : **[👉 Tester l'application en direct](https://smart-irrigation-system-hx4evkgdeyh3wpk6fcrj8y.streamlit.app/)**

---

## 🧠 Architecture Technique & Fonctionnalités

*   **Collecte de Données (API Open-Meteo) :** Analyse et extraction des données climatiques historiques (températures, rayonnement solaire, vitesse du vent, précipitations).
*   **Modèle Prédictif (LSTM) :** Réseau de neurones récurrents *Long Short-Term Memory* (développé avec TensorFlow) entraîné pour prédire l'Évapotranspiration de la culture ($ET_c$) à J+1 sur une fenêtre glissante de 10 jours.
*   **Moteur Agronomique (FAO-56) :** Algorithme de suivi dynamique du bilan hydrique du sol prenant en compte la Capacité au Champ, la Réserve Utile (RU) et la Réserve Facilement Utilisable (RFU).
*   **Système Expert d'Alerte :** Calcul automatisé de la dose exacte d'irrigation requise dès que les réserves du sol franchissent le seuil critique de confort (50%).

## 💻 Aperçu de l'Interface

![Dashboard Streamlit](dashboard_streamlit.png) *(Placez votre capture d'écran à la racine du dépôt pour l'afficher ici)*


