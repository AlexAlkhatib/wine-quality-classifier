# 🍷 Wine Quality Predictor — Deep Learning Classification & Regression

Prédire la qualité d'un vin (note de **3 à 9**) à partir de **11 mesures physico-chimiques**, avec des réseaux de neurones (TensorFlow/Keras) comparés à des modèles de référence scikit-learn.

Dataset : [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) — 6 497 vins rouges et blancs.

| | |
|---|---|
| **Tâches** | classification 7 classes · classification 3 classes · régression |
| **Modèles** | 3 MLP Keras (+ 2 ablations) · 6 baselines (Dummy, logistique/ridge, forêt aléatoire) |
| **Données** | 6 497 lignes × 13 colonnes → **5 329 vins uniques** → **25 features** |
| **Résultat clé** | le MLP **égale** la forêt aléatoire (MAE 0,550 vs 0,537) — il ne la dépasse pas |
| **Gain réel** | le **feature engineering** : `sulphates / chlorides` passe de r = +0,04 à **r = +0,31** |

---

## 📂 Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `wine-quality-classifier.ipynb` | notebook principal, **entièrement exécuté** (figures et résultats inclus) |
| `winequality.csv` | dataset source (6 497 lignes) |
| `requirements.txt` | dépendances (versions testées) |
| `wine-quality-classifier.pdf` | export PDF de la **version précédente** du notebook (non régénéré) |

---

## 🎯 Objectifs

1. **Auditer** les données : valeurs manquantes, doublons, asymétrie, déséquilibre des classes.
2. **Feature engineering** : construire des variables qui traduisent un raisonnement œnologique (équilibre acide, soufre actif, rapport sucre/alcool…).
3. **Visualiser** chaque signal et l'**interpréter** en langage métier.
4. **Comparer trois formulations** du même problème (7 classes, 3 classes, régression) et les situer face à des baselines solides.
5. **Expliquer** les résultats (courbes d'apprentissage, matrices de confusion, biais par note, importance par permutation) et en tirer des conclusions honnêtes.

---

## 🧭 Structure du notebook

| § | Contenu |
|---|---|
| 1 | Environnement & reproductibilité (graines fixées) |
| 2 | Chargement et audit du jeu de données |
| 3 | Nettoyage (doublons, valeurs manquantes) |
| 4 | **Feature engineering** — 13 variables dérivées |
| 5 | Analyse exploratoire : 7 visualisations interprétées |
| 6 | Importance des variables (information mutuelle, forêt aléatoire) |
| 7 | Découpage train / validation / test **sans fuite** |
| 8 | Baselines scikit-learn |
| 9 | MLP — classification 7 classes (+ ablation de pondération) |
| 10 | MLP — classification 3 classes (+ ablation de pondération) |
| 11 | MLP — régression |
| 12 | Interprétabilité : importance par permutation |
| 13 | Recherche d'hyperparamètres (3 architectures) |
| 14 | Synthèse comparative |
| 15 | Conclusion, limites et perspectives |

---

## 📊 Dataset

| Feature | Description | Remarque |
|---|---|---|
| `fixed acidity` | acidité fixe | |
| `volatile acidity` | acidité volatile | défaut (piqûre acétique) au-delà de ~0,05 g/L |
| `citric acid` | acide citrique | fraîcheur |
| `residual sugar` | sucre résiduel | de 0,6 à 65,8 g/L |
| `chlorides` | chlorures | distribution très asymétrique (skew 5,3) |
| `free sulfur dioxide` | SO₂ libre | soufre encore actif |
| `total sulfur dioxide` | SO₂ total | de 6 à 440 mg/L |
| `density` | densité | écart-type 0,003 : quasi constante |
| `pH` | acidité | |
| `sulphates` | sulfates | stabilité microbiologique |
| `alcohol` | taux d'alcool | **variable la plus prédictive** |
| `quality` | **cible** (3 à 9) | jugement humain agrégé |
| `type` | rouge / blanc | 4 898 blancs, 1 599 rouges |

### Qualité des données

| Contrôle | Constat | Action |
|---|---|---|
| Valeurs manquantes | **38 cellules (0,05 %)**, sur 34 lignes | conservation + **imputation médiane apprise sur le train** |
| Doublons | **1 168 lignes (18 %)** | suppression → **5 329 vins uniques** (évite la fuite train/test) |
| Asymétrie | `chlorides` 5,3 · `sulphates` 1,8 · `acidity` 1,7 | `log1p` sur 4 variables + standardisation |
| Déséquilibre | notes 5–6 = **77 %** ; note 9 = **5 vins** | métriques adaptées + ablation de pondération |

---

## 🛠️ Feature engineering — 13 variables construites

| Variable | Formule | Justification |
|---|---|---|
| `total_acidity` | fixed + volatile + citric | acidité globale perçue |
| `volatile_ratio` | volatile / total_acidity | c'est la **part** d'acide volatile qui est un défaut |
| `citric_ratio` | citric / total_acidity | fraîcheur relative |
| `bound_so2` | total − free SO2 | soufre déjà consommé |
| `free_so2_ratio` | free / total SO2 | soufre encore **actif** |
| `so2_per_alcohol` | total SO2 / alcohol | dose de conservateur rapportée au degré |
| `sugar_alcohol` | residual sugar / alcohol | équilibre moelleux / puissance |
| `alcohol_density` | alcohol / density | concentration alcoolique corrigée |
| `sulphates_chlorides` | sulphates / chlorides | rapport stabilité / défaut — **meilleur gain** |
| `log_*` ×4 | log1p(fixed acidity, residual sugar, chlorides, sulphates) | correction d'asymétrie |

**Gains mesurés (corrélation avec `quality`)**

| Variable brute | r | Variable construite | r |
|---|---|---|---|
| `alcohol` | +0,470 | `alcohol_density` | +0,470 (aucun gain) |
| `sulphates` | +0,041 | **`sulphates_chlorides`** | **+0,306** |
| `free sulfur dioxide` | +0,054 | `free_so2_ratio` | +0,125 |
| `volatile acidity` | −0,264 | `volatile_ratio` | −0,232 |

> Le feature engineering est **validé par trois méthodes indépendantes** : corrélation, information mutuelle (4 ratios dans le top 10) et importance par permutation (`free_so2_ratio` 4ᵉ).

> Une 14ᵉ variable (`density_excess`) a été **supprimée** après l'analyse de colinéarité : transformation affine de `density` (r = 1,000), donc sans aucune information.

---

## 📈 Visualisations & interprétations

| Visualisation | Ce qu'elle montre |
|---|---|
| Distribution de la note (global + par type) | 77 % des vins en 5–6 ; notes 8–9 quasi exclusivement blanches |
| Répartition en 3 classes | 37 % / 44 % / 19 % — ratio ramené de 462:1 à 2,3:1 |
| Boîtes à moustaches par note | alcool 9,6 → 11,5 % ; part d'acide volatile 0,043 → 0,035 ; ratio sulfates/chlorides 8,9 → 12,5 |
| Violons alcool × classe × type | effet de l'alcool **identique** pour rouges et blancs (+1,9 point) |
| Nuage alcool × acidité volatile | la zone des bons vins est un **quadrant**, pas un seuil ; acidité volatile 0,52 (rouge) vs 0,26 (blanc) |
| Matrice de corrélation (25 features) | blocs de colinéarité alcool/densité et soufre ; redondance `alcohol_density` ↔ `alcohol` (r = 1,000) |
| Pairplot des 4 meilleures variables | classes **non séparables linéairement** → modèle multivarié nécessaire |
| Information mutuelle vs importance forêt | alcool/densité en tête, suivi des ratios construits |
| Courbes d'apprentissage (×3) | sur-apprentissage contenu par early stopping + ReduceLROnPlateau |
| Matrices de confusion (7 et 3 classes) | 7 classes : extrêmes jamais prédits ; 3 classes : « haute » la moins bien rappelée |
| Prédit vs réel, résidus, biais par note | régression vers la moyenne : notes 3–4 surestimées, 7–8 sous-estimées |
| Importance par permutation | densité, acidité volatile, `alcohol_density`, `free_so2_ratio` |

---

## 🤖 Modèles & résultats

Protocole : **70 / 15 / 15** stratifié (3 730 / 799 / 800 vins), **imputation + standardisation ajustées sur le train uniquement**. Toutes les valeurs ci-dessous sont mesurées sur le **jeu de test**.

### Classification

| Modèle | 3 classes (accuracy) | 3 classes (F1 macro) | 7 classes (accuracy) |
|---|---|---|---|
| Dummy (classe majoritaire) | 0,436 | 0,202 | 0,436 |
| Régression logistique | 0,588 | 0,559 | 0,544 |
| Forêt aléatoire (400 arbres) | **0,621** | **0,608** | **0,568** |
| **MLP (deep learning)** | 0,609 | 0,602 | 0,566 |

* MLP 7 classes : 14 791 paramètres, `Dense(128→64→32)` + BatchNorm + Dropout 0,3 → **95,5 % des vins prédits à ±1 point**.
* MLP 3 classes : `Dense(64→32)` → rappel 0,66 / 0,60 / 0,53 (faible / moyenne / haute).

### Régression

| Modèle | MAE | RMSE | R² |
|---|---|---|---|
| Dummy (moyenne) | 0,694 | 0,878 | −0,000 |
| Ridge (linéaire) | 0,566 | 0,736 | 0,296 |
| Forêt aléatoire (400 arbres) | **0,537** | **0,696** | **0,371** |
| **MLP (deep learning)** | 0,550 | 0,698 | 0,368 |
| MLP (config. optimisée) | 0,551 | — | — |

* Le MLP prédit la note **en points de qualité** (cible standardisée puis dé-standardisée), perte **Huber**, sortie linéaire.
* **95,8 %** des vins prédits à ±1 point ; **452 / 800** notes exactes.

### Ablations — faut-il pondérer les classes ?

| Tâche | Sans pondération | Avec poids | Verdict |
|---|---|---|---|
| 7 classes | acc **0,566** / bal 0,279 | acc 0,392 / bal **0,403** | pondérer **détruit** l'exactitude (sous le Dummy) → non retenu |
| 3 classes | acc **0,609** / bal 0,597 | acc 0,582 / bal **0,620** | arbitrage serré → exactitude privilégiée |

### Recherche d'hyperparamètres (régression, sélection sur validation)

| Configuration | Paramètres | MAE validation | MAE test |
|---|---|---|---|
| A — (64, 32), dropout 0,10 | 3 777 | **0,555** | 0,551 |
| B — (128, 64), dropout 0,20 | 11 649 | 0,562 | **0,547** |
| C — (256, 128, 64), dropout 0,30 | 47 873 | 0,557 | 0,548 |

> Écart total : **0,008 point de MAE**. La taille du réseau n'est pas le levier : le signal plafonne.

---

## 💡 Conclusions

1. **Le deep learning n'écrase pas les méthodes classiques** sur ce dataset tabulaire de 5 329 lignes : le MLP est **au niveau** de la forêt aléatoire (MAE 0,550 vs 0,537), jamais devant. Résultat attendu sur des données tabulaires de taille moyenne.
2. **L'apport du réseau est la formulation** : un seul modèle continu fournit une note, une amplitude d'erreur et une tolérance ±1 point, et se décline en classification.
3. **Le feature engineering est le vrai gain** : `sulphates_chlorides` (r = +0,31 contre +0,04) et `free_so2_ratio` sont réellement utilisés par le modèle.
4. **Les baselines et ablations sont indispensables** : sans elles, un modèle pondéré à 0,392 (sous le Dummy) aurait pu être présenté comme « équilibré ».
5. **La limite est dans les données** : notes 3 (30 vins) et 9 (5 vins) inapprenables, cible bruitée, 5 000 lignes. Aucun réglage ne fera passer le R² de 0,37 à 0,80.
6. **La régression vers la moyenne n'est pas un bug** : c'est la réponse optimale face à ce niveau de bruit (biais +2,1 sur la note 3, −1,6 sur la note 8).

---

## ▶️ Lancer le projet

```bash
git clone https://github.com/username/wine-quality-predictor.git
cd wine-quality-predictor
pip install -r requirements.txt
jupyter notebook wine-quality-classifier.ipynb
```

**Versions testées** (Python 3.12) : pandas 3.1 · numpy 2.2 · scikit-learn 1.9 · TensorFlow 2.21 (CPU) · matplotlib 3.11 · seaborn 0.13

**Reproductibilité** : graines fixées (Python, NumPy, TensorFlow) → le notebook reproduit les mêmes scores à l'identique. Exécution complète : **≈ 3 à 6 minutes sur CPU**, aucun GPU requis.

---

## 🚀 Perspectives

1. **Régression ordinale** (perte ordinale, seuils appris) — la formulation la plus adaptée à une cible discrète ordonnée.
2. **Validation croisée 5 folds** + intervalles de confiance, avant de conclure qu'un modèle « gagne » sur 0,01 de MAE.
3. **Ensemble** MLP + forêt + `HistGradientBoostingRegressor`.
4. **Réduction des mesures** à 4–5 variables (alcool/densité, acidité volatile, SO₂ libre, sulfates/chlorides) → capteur embarqué.
5. **Modèles séparés rouges / blancs** : les corrélations changent de signe entre les types (acidité fixe : +0,12 chez les rouges, −0,11 chez les blancs).
6. **Déploiement** : API FastAPI + dashboard Streamlit.

---

## 👤 Auteur

**Alex Alkhatib**
Projet Deep Learning — Prédiction de la Qualité des Vins 🍷

## 📄 Licence

MIT License
Copyright (c) 2025 Alex Alkhatib
