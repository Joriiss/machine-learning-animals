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
- Sauvegarder le modèle dans `animal_classifier.h5`

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

## Fichiers

- `train_animals.py` - Script d'entraînement du modèle
- `test_animals.py` - Script de test du modèle
- `requirements.txt` - Dépendances Python
- `animal_classifier.h5` - Modèle sauvegardé (généré après l'entraînement)
