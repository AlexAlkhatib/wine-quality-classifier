## 📌 Objectif du Projet

L’objectif est de prédire la qualité d’un vin grâce à des modèles d’apprentissage supervisé :

* **Classification multi-classes** (notes de 3 à 9)
* **Classification simplifiée** (catégories : *faible*, *moyenne*, *haute qualité*)
* **Régression** (prédiction continue de la note)

Le pipeline inclut :

✔️ Prétraitement des données
✔️ Nettoyage et normalisation
✔️ Encodage des variables catégorielles
✔️ Construction de modèles de deep learning
✔️ Analyse des performances (MAE, accuracy…)
✔️ Visualisation des résultats


## 📂 Dataset

Dataset utilisé : **Wine Quality Dataset** (red + white wines)

Principales colonnes :

| Feature              | Description             |
| -------------------- | ----------------------- |
| fixed acidity        | Acidité fixe            |
| volatile acidity     | Acidité volatile        |
| citric acid          | Acide citrique          |
| residual sugar       | Sucre résiduel          |
| chlorides            | Chlorures               |
| free sulfur dioxide  | SO₂ libre               |
| total sulfur dioxide | SO₂ total               |
| density              | Densité                 |
| pH                   | Acidité                 |
| sulphates            | Sulfates                |
| alcohol              | Taux d’alcool           |
| quality              | Note de qualité (3 à 9) |

Répartition des types de vins :

* **4898 vins blancs**
* **1599 vins rouges**


## 🧹 Prétraitement des Données

### Étapes appliquées :

✔️ Vérification et imputation des valeurs manquantes (moyenne)
✔️ Normalisation des variables numériques (*MinMaxScaler*)
✔️ Encodage de la variable catégorielle *type* (red/white) via **OneHotEncoder**
✔️ Split des données :

* 70% train
* 15% validation
* 15% test


## 🤖 Modèles Construits

### 1️⃣ Modèle de Classification (10 classes)

Prédiction des notes de 1 à 10 via un réseau dense :

```
Dense(20, relu)
Dense(30, relu)
Dense(40, relu)
Dense(20, relu)
Dense(10, softmax)
```

### Résultats :

* **Accuracy test : ~59%**
* **MAE approx : 0.99**
* => Peu performant comparé à une approche régressive.


### 2️⃣ Modèle de Classification (3 classes)

Fusion des notes en catégories :

* **<6 : faible qualité**
* **=6 : qualité moyenne**
* **>6 : haute qualité**

Performances :

* **Accuracy test : ~61%**
* Meilleure stabilité que la classification multi-classes.


### 3️⃣ Modèle de Régression (Modèle le plus performant)

Prédiction continue de la qualité (normalisée entre 0 et 1)

Architecture dense :

```
Dense(5, relu)
Dense(10, relu)
Dense(10, relu)
Dense(1, sigmoid)
```

Performance :

* **MAE ≈ 0.16**
  ➡️ C’est le **meilleur modèle** pour prédire la qualité des vins.


## 📊 Visualisations

Le notebook inclut :

* Courbes **Loss / MAE** entraînement vs validation
* Répartition des notes
* Exploration statistique du dataset
* Comparaison des prédictions vs valeurs réelles


## 🚀 Technologies Utilisées

* **Python 3.x**
* pandas
* numpy
* scikit-learn
* TensorFlow / Keras
* seaborn / matplotlib
* Jupyter Notebook


## ▶️ Lancer le Projet

1. Cloner le projet :

```bash
git clone https://github.com/username/wine-quality-predictor.git
cd wine-quality-predictor
```

2. Installer les dépendances :

```bash
pip install -r requirements.txt
```

3. Lancer l’analyse :

```bash
jupyter notebook
```

Fichier principal :

```
wine_quality.ipynb
```


## ✨ Améliorations Futures

* Ajout d’un modèle **CNN 1D** pour exploiter les relations entre features
* Optimisation hyperparamétrique (GridSearch + KerasTuner)
* Déploiement API (FastAPI / Flask) pour une prédiction en temps réel
* Dashboard Streamlit interactif
* Ajout d’un modèle *ensemble learning* combinant régression & classification


## 👤 Auteur

**Alex Alkhatib**
Projet Deep Learning — Prédiction de la Qualité des Vins 🍷


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
