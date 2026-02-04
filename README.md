# Reconnaissance d'Animaux avec Réseaux de Neurones

Projet simple de classification d'images pour reconnaître trois types d'animaux : éléphants, tigres et girafes.

## Description

Ce projet utilise un réseau de neurones convolutif (CNN) pour classer des images d'animaux en trois catégories :
- Éléphant
- Tigre
- Girafe

## Installation

Installer les dépendances nécessaires :

```bash
pip install -r requirements.txt
```

## Structure des données

Les images d'entraînement doivent être organisées dans les dossiers suivants :
- `data/elephant/` - Images d'éléphants
- `data/tigre/` - Images de tigres
- `data/giraffe/` - Images de girafes

Les images de test doivent être placées dans :
- `data/test/` - Images de test (le nom du fichier doit commencer par `E`, `T` ou `G` pour indiquer la classe attendue)

## Utilisation

### 1. Entraîner le modèle

Lancer l'entraînement du modèle :

```bash
python train_animals.py
```

Le script va :
- Charger les images depuis les dossiers `data/elephant`, `data/tigre` et `data/giraffe`
- Entraîner un réseau de neurones simple
- Sauvegarder le modèle dans `models/animal_classifier.h5`

### 2. Tester le modèle

Tester le modèle avec des images de test :

```bash
python test_animals.py
```

Le script va :
- Charger le modèle sauvegardé
- Tester les images dans `data/test/`
- Afficher les résultats avec la précision globale et par classe

**Note :** Les noms de fichiers de test doivent commencer par :
- `E` pour éléphant
- `T` pour tigre
- `G` pour girafe

## Prétraitements appliqués

Les images subissent les prétraitements suivants avant l'entraînement :

1. **Redimensionnement** : Toutes les images sont redimensionnées à 128x128 pixels
2. **Conversion de couleur** : Conversion de BGR (OpenCV) vers RGB
3. **Normalisation** : Les valeurs de pixels sont normalisées de [0, 255] à [0, 1]

### Augmentation de données (entraînement uniquement)

Pour améliorer la robustesse du modèle, les images d'entraînement subissent une augmentation de données :
- **Rotation** : jusqu'à 20 degrés
- **Zoom** : entre 80% et 120%
- **Retournement horizontal** : retournement aléatoire de l'image

Ces transformations sont appliquées aléatoirement à chaque époque d'entraînement pour augmenter artificiellement la taille du jeu de données.

## Architecture du modèle CNN

Le modèle utilise **TensorFlow/Keras** pour construire un réseau de neurones convolutif (CNN) avec l'architecture suivante :

### 1. Couches convolutionnelles (Extraction des caractéristiques visuelles)

Le modèle contient **3 couches convolutionnelles** qui extraient les caractéristiques visuelles des images :

- **Couche 1** : `Conv2D(32, (3, 3))` - 32 filtres de taille 3x3
- **Couche 2** : `Conv2D(64, (3, 3))` - 64 filtres de taille 3x3
- **Couche 3** : `Conv2D(64, (3, 3))` - 64 filtres de taille 3x3

Toutes les couches convolutionnelles utilisent :
- **Fonction d'activation ReLU** pour introduire la non-linéarité
- **Filtres de taille 3x3** (recommandé pour débuter)

### 2. Couches de pooling (Réduction de la dimensionnalité)

Après chaque couche convolutionnelle, une couche de **MaxPooling2D(2, 2)** est appliquée pour :
- Réduire la dimensionnalité des données
- Diminuer le nombre de paramètres
- Améliorer l'efficacité du calcul

### 3. Couches denses (Classification finale)

Après l'aplatissement des données (`Flatten`), le modèle contient :

- **Couche dense intermédiaire** : `Dense(128)` avec activation ReLU
- **Dropout(0.5)** : Régularisation pour éviter le surapprentissage (50% des neurones désactivés aléatoirement)
- **Couche de sortie** : `Dense(3)` pour les 3 classes d'animaux

### 4. Sortie softmax (Probabilités pour chaque classe)

La dernière couche utilise la fonction d'activation **softmax** qui :
- Convertit les sorties en probabilités
- Assure que la somme des probabilités pour toutes les classes = 1
- Permet d'obtenir la classe prédite avec le plus haut score de confiance

### Configuration du modèle

- **Optimiseur** : Adam (adaptatif et efficace)
- **Fonction de perte** : `categorical_crossentropy` (adaptée à la classification multi-classes)
- **Métrique** : Accuracy (précision)
- **Taille d'entrée** : 128x128 pixels, 3 canaux (RGB)

Cette architecture simple et efficace suit les bonnes pratiques pour la classification d'images et est idéale pour débuter avec les réseaux de neurones convolutifs.

## Entraînement et validation du modèle

### 1. Séparation des données

Les images sont divisées en deux ensembles pour évaluer objectivement les performances :

- **Ensemble d'entraînement** : 80% des images
- **Ensemble de test** : 20% des images

La séparation utilise :
- **Stratification** (`stratify=y`) : Maintient les proportions de chaque classe dans les deux ensembles
- **Random state** : `random_state=42` pour garantir la reproductibilité des résultats

### 2. Optimisation

Le modèle utilise les paramètres d'optimisation suivants :

- **Optimiseur** : **Adam** - Optimiseur adaptatif qui ajuste automatiquement le taux d'apprentissage
- **Fonction de perte** : **`categorical_crossentropy`** - Entropie croisée adaptée à la classification multi-classes
- **Métrique** : **Accuracy** - Précision pour suivre les performances pendant l'entraînement

### 3. Éviter le surapprentissage

Plusieurs techniques sont mises en place pour garantir la généralisation du modèle :

#### Augmentation de données
Les images d'entraînement subissent des transformations aléatoires à chaque époque :
- Rotation jusqu'à 20 degrés
- Zoom entre 80% et 120%
- Retournement horizontal

Cela augmente artificiellement la taille du jeu de données et améliore la robustesse du modèle.

#### Dropout
Une couche de **Dropout(0.5)** est appliquée dans la couche dense :
- Désactive aléatoirement 50% des neurones pendant l'entraînement
- Force le modèle à ne pas dépendre de neurones spécifiques
- Réduit le risque de surapprentissage

#### Validation pendant l'entraînement
Le modèle est évalué sur l'ensemble de validation à chaque époque :
- Permet de surveiller les performances en temps réel
- Détecte le surapprentissage si l'accuracy de validation stagne ou diminue
- Aide à déterminer le nombre optimal d'époques

### Processus d'entraînement

1. Chargement et prétraitement des images
2. Séparation train/test (80/20)
3. Configuration de l'augmentation de données
4. Création et compilation du modèle
5. Entraînement avec validation à chaque époque
6. Évaluation finale sur l'ensemble de test
7. Sauvegarde du modèle dans `models/animal_classifier.h5`

## Fichiers

- `train_animals.py` - Script d'entraînement du modèle
- `test_animals.py` - Script de test du modèle
- `requirements.txt` - Dépendances Python
- `models/animal_classifier.h5` - Modèle sauvegardé (généré après l'entraînement)
