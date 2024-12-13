# Projet Finance : Prévisions des Actions Apple avec Prophet

Ce projet utilise des modèles de séries temporelles, notamment **Prophet**, pour effectuer des prévisions sur les prix des actions Apple. Le projet est élaboré en suivant une architecture modulaire, avec une distinction claire entre les composants backend et frontend.

---

## **Structure du Projet**

### **Racine**
- `.venv` : Environnement virtuel Python pour les dépendances.
- `README.md` : Documentation du projet.

### **backend**
- `arima_model.py` : Modèle ARIMA (alternative).
- `prophet_model.py` : Modèle Prophet pour les prévisions.
- `data_processing.py` : Script pour le traitement et nettoyage des données.
- `time_series_analysis.py` : Analyse exploratoire des séries temporelles.
- `requirements.txt` : Liste des dépendances Python pour le backend.

### **data**
- `cleaned/cleaned_apple_stock_data.csv` : Fichier de données nettoyées.
- `raw/` : Fichiers de données brutes.

### **frontend/components**
- `streamlit_app.py` : Application interactive pour visualiser les prévisions.
- `requirements.txt` : Dépendances pour l’application Streamlit.
- `model_prophet.py` : Modèle Prophet dédié à l’interface frontend.

---

## **Installation et Configuration**

### 1. **Cloner le Projet**
```bash
git clone <url_du_projet>
cd project_finance
```

### 2. **Configurer l’Environnement Virtuel**
```bash
python3 -m venv .venv
source .venv/bin/activate  # Sous macOS/Linux
.venv\Scripts\activate  # Sous Windows
```

### 3. **Installer les Dépendances**
#### Backend
```bash
pip install -r backend/requirements.txt
```
#### Frontend
```bash
pip install -r frontend/components/requirements.txt
```

### 4. **Vérifier les Données**
Assurez-vous que le fichier **`cleaned_apple_stock_data.csv`** est bien présent dans `data/cleaned/`.


## **Exécution du Projet**

### 1. **Lancer les Analyses Backend**
#### Prophet
```bash
cd backend
python prophet_model.py
```
#### ARIMA (en option)
```bash
python arima_model.py
```

### 2. **Lancer l’Interface Frontend**
#### Démarrer l’Application Streamlit
Depuis la racine du projet :
```bash
streamlit run frontend/components/streamlit_app.py
```

#### Description de l’Application
- **Slider** : Permet de choisir l’horizon de prévision (30 à 180 jours).
- **Bouton** : Entraîne le modèle et génère les prévisions.
- **Graphique** : Affiche les prévisions avec une bande d’incertitude.
- **Résultats** : Récapitulatif des métriques (MAE, RMSE).

## **Résultats et Métriques**

- **MAE** : Erreur absolue moyenne – mesure la différence moyenne entre les valeurs réelles et prédites.
- **RMSE** : Erreur quadratique moyenne – indique la dispersion des erreurs de prévision.

Les résultats obtenus montrent que le modèle Prophet suit efficacement les tendances historiques, avec une précision adaptée à des horizons de prévision raisonnables.


## **Fichiers Clés et Leur Fonctionnement**

### Backend
#### prophet_model.py
- Entraîne un modèle Prophet sur les données nettoyées.
- Génère des prévisions pour un horizon personnalisable.
- Sauvegarde et affiche les métriques MAE et RMSE.

#### time_series_analysis.py
- Réalise une analyse exploratoire des données temporelles.
- Décompose les tendances, la saisonnalité et les résidus.

### Frontend
#### streamlit_app.py
- Interface utilisateur permettant de manipuler les prévisions.
- Visualise les résultats sous forme de graphiques interactifs.
- Fournit un résumé des résultats avec des métriques d’évaluation.


## **Améliorations Futures**
- Intégration d’autres modèles de prévision pour comparaison (ex. : SARIMA).
- Ajout d’un module d’automatisation pour tester plusieurs horizons de prévision.
- Déploiement de l’application sur une plateforme comme Heroku ou AWS.
- Ajout de tests unitaires pour valider la robustesse des scripts.


## **Notes pour les Collègues**
- Si un problème survient lors de l’exécution, vérifiez d’abord que l’environnement virtuel est bien activé et que les dépendances sont installées.
- En cas de question sur le fonctionnement d’un script, référez-vous aux commentaires inclus dans le code ou contactez le référent du projet.

## **Contributeurs**
- [Mathieu Soussignan](https://www.mathieu-soussignan.com).
- Alan Jaffré.
- Ahmed Bahri.
