# Documentation complète du projet — IA d'évaluation de la qualité visuelle des affiches publicitaires (JCDecaux)

**But de ce document** : synthèse technique exhaustive de tout le travail réalisé (code, données, méthodologie, résultats, fichiers produits), destinée à être utilisée comme base factuelle pour la rédaction du mémoire. Tous les chiffres cités sont réels, mesurés sur ce projet — aucun n'est un exemple illustratif, sauf mention explicite contraire.

**Sujet du mémoire** : Vers une IA d'évaluation qualitative des visuels publicitaires : analyse d'image, style, contraste et visibilité.

---

## Table des matières

1. Vue d'ensemble et architecture du pipeline
2. Les données
3. Le modèle CNN multi-tâches
4. Robustesse : déséquilibre de classes et pondération
5. Résultats — métriques complètes des 20 attributs
6. OCR (extraction de texte)
7. Vérification de cohérence logo/marque
8. Score composite et recommandations (formule complète)
9. Application utilisateur (Streamlit)
10. Test sur nouvelles images (2024)
11. Export au format de l'équipe métier
12. Bugs rencontrés et corrigés (narratif méthodologique)
13. Tous les fichiers produits par le projet
14. Limites connues
15. Décisions actées avec l'utilisateur (traçabilité)
16. Perspectives

---

## 1. Vue d'ensemble et architecture du pipeline

Le projet est développé dans `jcdecaux_project/`. Pipeline complet, dans l'ordre d'exécution :

```
1. src/excel_matching.py   → apparie les lignes Excel (BDDCréaJCDecaux.xlsx) aux fichiers image
2. src/encoding.py         → encode les 20 attributs cibles, fusionne les classes rares, calcule les poids de classe
3. src/dataset.py          → Dataset PyTorch multi-tâches (chargement image + augmentations)
4. src/model.py            → architecture du CNN multi-tâches (EfficientNet-B0 + 20 têtes)
5. src/train.py            → entraînement (perte pondérée par classe, arrêt anticipé)
6. src/evaluate.py         → métriques (accuracy, F1, précision/rappel par classe, matrices de confusion)
7. src/ocr.py              → extraction de texte (EasyOCR) + détection de QR code (OpenCV)
8. src/logo_match.py       → référence de logos par marque + vérification de cohérence
9. src/recommend.py        → score composite /100 + recommandations (moteur de règles)
10. src/inference.py       → prédiction sur une image ou un lot (predict_batch), devinette de marque
11. src/team_export.py     → export au format de la base d'origine, incrémental
12. src/compare_predictions.py → comparaison prédit vs réel sur images déjà labellisées
13. src/img_utils.py       → ouverture d'image robuste (formats variés, haute résolution)
14. app.py                 → application Streamlit (interface utilisateur finale)
15. api/main.py            → API FastAPI /predict (existante, non utilisée par l'app finale)
```

**Stack technique** : Python 3.12, PyTorch 2.3 + torchvision 0.18 (CNN, CPU uniquement — pas de GPU disponible), scikit-learn (encodage, métriques), OpenCV 4.9 (traitement d'image, QR code), EasyOCR 1.7 (OCR), pandas/openpyxl (données), Streamlit 1.39 (application), FastAPI (API, non utilisée dans la version finale).

---

## 2. Les données

- Base source : `BDDCréaJCDecaux.xlsx`, feuille "Feuil2", **1165 lignes** (campagnes publicitaires 2019-2023), 29 colonnes.
- Colonnes de métadonnées (jamais des cibles de prédiction) : `ID JCDECAUX`, `Marques`, `Année`, `Reftest`, `Nom_fichier_visuel`, `FAMILLE`, `CLASSE`, `Nombre de visuels`.
- **1147 lignes** avec un code retrouvé côté images, 18 sans image.
- **935 images** téléchargées (depuis SharePoint, via un bot Python), toutes associées à un code.
- Après appariement final (`excel_matching.py`, normalisation des noms + correspondance tolérante) : **1007 lignes exploitables** avec image (`data/output/df_labeled.csv`), 231 images non appariées.
- Poids moyen d'image : 0,42 Mo (min 45 Ko, max 9,6 Mo). Formats : `.jpg` (940), `.png` (13).
- Répartition très déséquilibrée par famille produit : BOISSONS (145 images), DISTRIBUTION (72), MODE_ET_ACCESSOIRES (71), HYGIENE_BEAUTE (65)... jusqu'à APPAREILS_MENAGERS (2) et INDUSTRIE (5).
- **188 marques uniques**, très inégalement représentées : BUT (88 images), FREE TELECOM (58), EURO DISNEY (25), MCDONALD'S (20+12), SOCIETE GENERALE (19), LACOSTE (19)... la plupart des marques ont moins de 5 images.

### 20 attributs cibles retenus (après exclusion de la Notoriété, voir §15)

`Type de campagne`, `Couleur dominante`, `Couleur propre à la marque ?`, `1er point d'accroche`, `Personnages`, `Si Egérie : est-elle propre à la marque ou également sur d'autres marques ?`, `Utilisation de l'égérie dans le temps`, `Genre`, `Tranche d'âge`, `Quantité/Place du texte dans le visuel`, `Discours utilisé`, `Style de texte (Lettres)`, `Taille du logo`, `Éléments de branding`, `Présence d'un QR Code`, `Langue du claim`, `Contraste`, `Place du logo`, `Mise en avant prix`, `Charge visuelle`.

---

## 3. Le modèle CNN multi-tâches

**Architecture** (`src/model.py`, classe `MultiTaskModel`) :
- **Backbone partagé** : `EfficientNet-B0` (pré-entraîné sur ImageNet, `torchvision.models.efficientnet_b0`), qui extrait un vecteur de caractéristiques visuelles (1280 dimensions) à partir de l'image. Option `ResNet50` disponible dans le code mais non utilisée.
- **20 têtes de classification**, une par attribut cible : chaque tête est une couche linéaire (`nn.Linear`) qui prend le vecteur de features du backbone et produit une distribution de probabilité sur les classes de cet attribut.
- Toutes les têtes sont entraînées **simultanément** (multi-tâches) : le backbone apprend une représentation visuelle générale utile à toutes les tâches, plus efficace que 20 modèles indépendants (moins de paramètres, partage d'information entre tâches corrélées).

**Prétraitement des images** (`src/dataset.py`) :
- Redimensionnement à **160×160 pixels** (réduit de 224×224 initialement prévu, pour tenir dans la mémoire disponible — voir §12).
- Normalisation avec les statistiques ImageNet (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
- Augmentation de données à l'entraînement uniquement : flip horizontal aléatoire, rotation aléatoire (±10°), jitter de couleur (luminosité/contraste/saturation).
- Split train/validation/test : **80% / 10% / 10%**.

**Entraînement** (`src/train.py`) :
- Optimiseur : `AdamW`, learning rate `1e-4`, weight decay `1e-5`.
- Scheduler : `ReduceLROnPlateau` (réduit le learning rate si la perte de validation stagne).
- Fonction de perte : `CrossEntropyLoss` par tâche, **pondérée par classe** (voir §4), moyennée sur les 20 tâches à chaque batch.
- Batch size : **4** (réduit de 16 initialement, contrainte mémoire — voir §12).
- Arrêt anticipé (early stopping) : patience de 6 epochs sans amélioration de la perte de validation.
- **Résultat d'entraînement réel** : arrêt à l'epoch **20** (sur un budget maximum de 30), meilleur modèle obtenu à l'epoch **14** (perte de validation = 0,8413).

---

## 4. Robustesse : déséquilibre de classes et pondération

**Problème identifié** : plusieurs attributs ont des classes très peu représentées. Exemples mesurés (nombre de classes avec moins de 15 occurrences sur 1007 lignes) :
- `Personnages` : 10 classes rares sur 18
- `1er point d'accroche` : 7 sur 13
- `Tranche d'âge` : 7 sur 13
- `Couleur dominante` : 5 sur 17

**Deux corrections appliquées** (`src/encoding.py`) :

1. **Fusion des classes rares** : toute classe avec moins de `RARE_CLASS_MIN = 15` occurrences est regroupée dans une catégorie `"Autre"`. Exemple mesuré : `1er point d'accroche` passe de 12 à 7 classes après fusion ; `Tranche d'âge` de 12 à 7.
2. **Poids de classe (inverse-fréquence)** : pour chaque attribut, chaque classe reçoit un poids inversement proportionnel à sa fréquence (`class_weights.pkl`), utilisé dans `CrossEntropyLoss(weight=...)`. Une erreur sur une classe rare "coûte" donc plus cher au modèle qu'une erreur sur une classe fréquente, ce qui force l'apprentissage des cas rares.

Ces deux corrections sont la cause principale du gain de performance mesuré (voir §5) — bien plus que le choix d'architecture.

---

## 5. Résultats — métriques complètes des 20 attributs

Mesurées sur le jeu de test (10% des 1007 images, non vues à l'entraînement), avec `src/evaluate.py`. Métriques : **Accuracy** (taux de bonnes réponses) et **F1-score pondéré** (combine précision et rappel, plus fiable que l'accuracy seule quand les classes sont déséquilibrées).

| Attribut | Accuracy | F1 pondéré | Nb. classes |
|---|---|---|---|
| Type de campagne | 0,725 | 0,725 | 6 |
| Couleur dominante | 0,598 | 0,596 | 13 |
| Couleur propre à la marque ? | 0,608 | 0,604 | 5 |
| 1er point d'accroche | 0,794 | 0,796 | 7 |
| Personnages | 0,647 | 0,645 | 10 |
| Si Egérie : propre à la marque ou non | 0,824 | 0,823 | 4 |
| Utilisation de l'égérie dans le temps | 0,765 | 0,768 | 4 |
| Genre | 0,775 | 0,777 | 6 |
| Tranche d'âge | 0,725 | 0,737 | 7 |
| Quantité/Place du texte | 0,598 | 0,595 | 5 |
| Discours utilisé | 0,657 | 0,679 | 6 |
| Style de texte (Lettres) | 0,667 | 0,682 | 7 |
| Taille du logo | 0,716 | 0,716 | 5 |
| Éléments de branding | 0,618 | 0,607 | 5 |
| **Présence d'un QR Code** | 0,902 | 0,900 | 3 |
| Langue du claim | 0,794 | 0,802 | 5 |
| Contraste | 0,618 | 0,618 | 4 |
| Place du logo | 0,598 | 0,591 | 8 |
| **Mise en avant prix** | 0,922 | 0,923 | 4 |
| Charge visuelle | 0,716 | 0,711 | 4 |

*(Fichier source : `data/output/evaluation/metrics.xlsx`, feuille `resume_par_attribut` ; précision/rappel détaillés par classe individuelle dans la feuille `precision_rappel_par_classe` ; matrices de confusion en image dans `data/output/evaluation/cm_*.png`, une par attribut.)*

### Comparaison avant / après les corrections de robustesse (§4)

Mesurée sur un premier modèle entraîné sans fusion des classes rares ni pondération, puis sur le modèle final :

| Attribut | F1 avant correction | F1 après correction |
|---|---|---|
| 1er point d'accroche | 0,19 | **0,80** |
| Couleur dominante | 0,23 | **0,60** |
| Présence d'un QR Code | 0,49 | **0,90** |
| Langue du claim | 0,45 | **0,80** |
| Mise en avant prix | 0,60 | **0,92** |
| Charge visuelle | 0,53 | **0,71** |
| Contraste | 0,50 | **0,62** |

**Lecture critique importante** : l'accuracy de "Présence d'un QR Code" a légèrement *baissé* (0,97 → 0,90) alors que son F1 a presque doublé (0,49 → 0,90). Explication : la classe est très déséquilibrée (la quasi-totalité des affiches n'ont pas de QR code) — un modèle qui prédit toujours "Non" atteint déjà ~0,97 d'accuracy sans rien discriminer. Le F1 pondéré est donc la métrique de référence à citer dans le mémoire, pas l'accuracy brute.

---

## 6. OCR (extraction de texte)

**Outil** : EasyOCR (bibliothèque Python pure, réseaux de neurones pré-entraînés pour détection + reconnaissance de texte). Remplace Tesseract (prévu initialement dans la bibliographie) car le binaire Tesseract n'est pas installé sur la machine de développement — décision technique documentée, pas un choix scientifique.

**Fonctions** (`src/ocr.py`) :
- `extract_text(image_path)` → texte détecté, nombre de caractères, nombre de zones de texte détectées, **ratio de surface occupée par le texte** (aire des boîtes de texte / aire totale de l'image — proxy quantitatif de la densité de texte), **proportion de lettres en majuscule** (proxy du style typographique).
- `detect_qr_code(image_path)` → détection de QR code via `cv2.QRCodeDetector` (OpenCV), déterministe, sans réseau de neurones. Sert de vérification indépendante de l'attribut "Présence d'un QR Code" prédit par le CNN.

**Paramètre technique important** : `canvas_size=480` (résolution interne de traitement d'EasyOCR, réduite de la valeur par défaut 2560px) — nécessaire pour éviter un crash mémoire (voir §12, bug n°1).

---

## 7. Vérification de cohérence logo/marque

**Problème** : aucune base de données publique de logos ne couvre les ~188 marques annonceurs spécifiques à ce projet.

**Méthode retenue** (`src/logo_match.py`), sans entraînement d'un détecteur dédié :
1. Recadrage automatique de la zone probable du logo, via l'attribut déjà labellisé `Place du logo` (7 zones : Haut/Bas × Gauche/Milieu/Droit + "Dans le corps du visuel"), mappées vers des coordonnées relatives dans l'image (`PLACE_LOGO_BBOX_MAP`, `config/config.py`).
2. Extraction d'un vecteur de caractéristiques (embedding) de ce recadrage, en réutilisant le **backbone EfficientNet-B0 déjà entraîné** (pas de réseau supplémentaire).
3. Construction d'une bibliothèque de référence : pour chaque marque avec au moins 2 images (`LOGO_MIN_IMAGES_PER_BRAND = 2`), l'embedding moyen des recadrages est stocké → **139 marques couvertes sur 188** (61 lignes ignorées : image manquante ou marque avec une seule image).
4. À l'inférence : comparaison par **similarité cosinus** entre l'embedding de la nouvelle image et celui de la marque déclarée (+ recherche de la marque la plus proche parmi toutes les références).
5. Seuil de cohérence : `LOGO_MATCH_THRESHOLD = 0.5`.

**Limite mesurée et à assumer** : même pour une marque bien présente dans la référence (ex. IKEA), la similarité obtenue peut être faible (0,32 mesuré sur un test, sous le seuil, donc jugée "incohérente" alors que l'image est une vraie affiche IKEA). C'est un recadrage approximatif comparé par similarité visuelle générale (pas un détecteur de logo dédié type YOLO) — fonctionne mécaniquement, précision perfectible.

---

## 8. Score composite et recommandations (formule complète)

**Principe** : moteur de règles déterministe (`src/recommend.py`), **pas un LLM**. Chaque note est calculée par une formule explicite et reproductible — choix méthodologique délibéré pour rester dans l'esprit d'une évaluation *objective* (cohérent avec la problématique du mémoire) et pour que le score soit explicable, vérifiable, reproductible (contrairement à un LLM qui donnerait un résultat différent à chaque appel).

### Étape 1 — valeurs numériques fixes par attribut prédit

```
Contraste :        "Très contrasté" → 1,0 | "Un peu contrasté" → 0,6 | "Pas contrasté" → 0,2
Style de texte :   "Majuscule gras" → 1,0 | "Masjuscule sans gras" → 0,8 |
                   "Minuscule gras" → 0,6 | "Minuscule sans gras" → 0,4
Charge visuelle :  "Faible (3 éléments)" → 0,7 | "Moyen (4-5)" → 1,0 | "Élevée (plus de 6)" → 0,5
Couleur propre à la marque : "Dominante" → 1,0 | "Présente" → 0,7 | "Non" → 0,4
Valeur par défaut (classe "Non renseigné"/"Autre"/inconnue) : 0,5
```

### Étape 2 — 4 sous-scores (0 à 1, puis ×100 pour l'affichage)

```
Lisibilité   = 0,5 × Contraste + 0,3 × Style_texte + 0,2 × OCR_ratio_texte
Contraste    = valeur directe de l'attribut prédit
Esthétique   = 0,7 × Charge_visuelle + 0,3 × Couleur_propre_marque
Cohérence    = 0,7 × Logo_coherent + 0,3 × Accord_QR_code
```

Où :
- `OCR_ratio_texte` : 1,0 si le texte occupe entre 1% et 30% de la surface de l'image (plage jugée idéale) ; décroît linéairement vers 0 au-delà de 30% (trop de texte = affiche surchargée) ; 0,5 (neutre) si moins de 1% (quasi aucun texte détecté).
- `Logo_coherent` : 1,0 si `logo_is_consistent` est vrai, 0,2 si faux, 0,5 si non évaluable (pas de marque déclarée ou pas de référence).
- `Accord_QR_code` : 1,0 si la présence de QR code prédite par le CNN concorde avec la détection OpenCV, 0,3 sinon, 0,5 si non évaluable.

### Étape 3 — score global sur 100

```
Score_global = 100 × (0,40 × Lisibilité + 0,30 × Contraste + 0,20 × Esthétique + 0,10 × Cohérence)
```

**⚠️ Pondération 40/30/20/10 : reprise du plan de mémoire initial (exemple illustratif), pas validée scientifiquement ni avec le directeur — point explicitement à trancher/discuter.**

### Étape 4 — recommandations textuelles (règles "si...alors...")

Exemples de règles implémentées :
- Si le score de contraste < 0,5 → *"Contraste jugé insuffisant entre le texte/le sujet et le fond : envisager d'augmenter l'écart de luminosité ou de couleur."*
- Si le style de texte prédit est peu lisible (minuscule/sans gras) → *"Le style typographique détecté réduit la lisibilité à distance ; privilégier une casse majuscule et un graissage plus marqué (norme AFNOR)."*
- Si le texte OCR occupe plus de 30% de l'image → *"Réduire la quantité de texte pour clarifier le message principal."*
- Si `Charge visuelle = "Élevée"` → *"Simplifier la composition peut améliorer l'impact visuel (cf. Aesthetic Visual Analysis, MIT)."*
- Si le logo est jugé incohérent avec la marque déclarée → message d'alerte correspondant.
- Si aucun point n'est détecté → *"Aucun point d'attention majeur détecté sur les critères évalués."*

---

## 9. Application utilisateur (Streamlit)

**Fichier** : `app.py` (racine de `jcdecaux_project/`). Lancement : `python -m streamlit run app.py`, accessible sur `http://localhost:8501`.

**Fonctionnement** :
1. L'utilisateur dépose une image.
2. La marque est **devinée automatiquement depuis le nom du fichier** (`guess_brand_from_filename`, `src/inference.py`) et pré-remplie dans un champ modifiable — utile car pour la majorité des nouvelles affiches, le nom de fichier contient directement le nom de la marque (ex. "BOMPARD (1).jpg" → "BOMPARD"). Heuristique : suppression des suffixes numériques de fin de nom précédés d'un séparateur (" (1)", "-2"...), en prenant soin de ne pas tronquer les marques se terminant légitimement par un chiffre (ex. "TF1", "M6" restent intacts).
3. Le pipeline complet est exécuté (CNN → OCR → logo → score).
4. Affichage : score global, 4 sous-scores, recommandations, résultat de cohérence logo, texte OCR détecté, détail des 20 attributs.
5. **Bouton de téléchargement** : exporte/complète automatiquement un fichier Excel au format de la base d'origine de l'équipe (voir §11), mis à jour à chaque image analysée.

**Choix technique** : Streamlit plutôt que FastAPI + frontend séparé (alternative écartée) — suffisant pour un usage local mono-utilisateur, déjà installé dans l'environnement, une seule fichier Python sans code frontend séparé.

**Tests réalisés** :
- Image connue (marque M6, présente dans les données d'entraînement) → logo jugé cohérent, similarité 0,78, score 95,8/100.
- Nouvelle image (marque absente de la référence) → le système répond honnêtement qu'aucune comparaison n'est possible, plutôt que d'inventer une cohérence.
- Fichier invalide (non-image) → message d'erreur propre, pas de plantage.

---

## 10. Test sur nouvelles images (2024)

Une nouvelle base d'images (`nv_images/2024/`, ~2565 images réparties dans 46 sous-dossiers hebdomadaires) a été fournie après l'entraînement, pour valider la généralisation du modèle à des données jamais vues.

**Traité à ce jour** : sous-dossiers `s01` à `s05`, **232 images**, 0 erreur (après corrections, voir §12).
- Score global : moyenne 73,2/100, minimum 39,4, maximum 94,7 — bonne dispersion, le modèle différencie bien les affiches.
- QR codes détectés : 11 images sur 232.
- Marques 2024 majoritairement absentes de la référence logo construite sur les données 2019-2023 (BOMPARD, CIVA, EMMA MATELAS... sont de nouveaux annonceurs) — la vérification logo n'est donc évaluable de façon fiable que pour les marques récurrentes (ex. BMW, IKEA, présentes dans les deux périodes).
- Reste à traiter : ~2330 images (`s06` à `S47`).

---

## 11. Export au format de l'équipe métier

**Fichier** : `src/team_export.py`. Génère/complète `data/output/BDD_nouvelles_affiches_a_valider.xlsx`, avec **exactement les mêmes colonnes que `BDDCréaJCDecaux.xlsx`** (même noms, même ordre), pour que l'équipe puisse réutiliser les prédictions dans son propre processus d'étiquetage sans changer d'outil.

- Colonnes non déductibles d'une image seule (`ID JCDECAUX`, `Reftest`, `FAMILLE`, `CLASSE`, `Nombre de visuels`) : laissées vides.
- `Marques` : remplie par la marque devinée/déclarée. `Année` : déduite du dossier parent si un motif à 4 chiffres est présent dans le chemin (ex. "2024"). `Notoriété (base YouGov)` : laissée vide (attribut exclu du modèle, voir §15).
- **Comportement incrémental** : chaque nouvel appel (traitement par lots ou via l'application) ajoute les nouvelles lignes au fichier existant ; une image déjà présente (même `Nom_fichier_visuel`) est mise à jour, pas dupliquée.

---

## 12. Bugs rencontrés et corrigés (narratif méthodologique)

Ce narratif peut être valorisé dans le mémoire comme partie du travail d'ingénierie réel (pas seulement le résultat final) :

1. **Fichier `encoding.py` corrompu** : contenait par erreur le code d'une boucle d'entraînement au lieu de la logique d'encodage — signe d'un développement itératif. Corrigé : séparation stricte des responsabilités, chaque étape du pipeline relit ses données sur disque plutôt que de dépendre d'un état partagé entre imports Python (fragile).
2. **`MemoryError` à l'entraînement** : la machine de développement (~8 Go RAM, souvent <1,5 Go libre) ne supportait pas la résolution/batch size initiaux (224×224, batch 16). Corrigé : résolution réduite à 160×160, batch size à 4.
3. **Segmentation Fault (crash mémoire natif) sur l'OCR** : EasyOCR traite les images en interne jusqu'à 2560px par défaut, ce qui sature la mémoire sur les grandes images (affiches portrait haute résolution) et provoque un crash bas niveau (non récupérable par un simple `try/except` Python). Corrigé : `canvas_size=480`.
4. **Erreur d'encodage console Windows** : la console par défaut (cp1252) ne supporte pas les caractères Unicode (ex. "█") utilisés par la barre de progression du téléchargement des poids EasyOCR, provoquant un crash au tout premier lancement. Corrigé : forçage de l'encodage UTF-8 sur stdout/stderr.
5. **Images haute résolution rejetées ("decompression bomb")** : certaines images scannées dépassaient 289 mégapixels, au-delà de la limite de sécurité par défaut de Pillow. Corrigé (`src/img_utils.py`) : limite levée + décodage progressif (`Image.draft()`) pour éviter de charger l'image complète en mémoire avant redimensionnement.
6. **Incohérence de format de checkpoint** : les différentes versions du script d'entraînement sauvegardaient le modèle sous deux formats différents (dictionnaire `{"model_state":...}` vs état brut), cassant le chargement du modèle par l'API/l'inférence. Corrigé partout de manière cohérente.

**Point d'attention opérationnel** : cette machine ne supporte pas de faire tourner l'application Streamlit et un traitement par lots en même temps (mémoire insuffisante) — à mentionner si le mémoire évoque le déploiement/la scalabilité.

---

## 13. Tous les fichiers produits par le projet

### Code (`jcdecaux_project/src/`, `config/`, `api/`, racine)
`config/config.py`, `src/excel_matching.py`, `src/encoding.py`, `src/dataset.py`, `src/model.py`, `src/train.py`, `src/evaluate.py`, `src/ocr.py`, `src/logo_match.py`, `src/recommend.py`, `src/inference.py`, `src/team_export.py`, `src/compare_predictions.py`, `src/img_utils.py`, `app.py`, `api/main.py`.

### Artefacts du modèle (`artifacts/`)
- `best_model.pth` — poids du modèle entraîné (meilleure epoch).
- `encoders.pkl` — encodeurs `LabelEncoder` par attribut + `target_cols` + `col_types`.
- `encoders_mapping.xlsx` — mapping lisible classe ↔ identifiant numérique.
- `class_weights.pkl` — poids de classe (inverse-fréquence) par attribut.
- `logo_reference.pkl` — bibliothèque d'embeddings de logo par marque (139 marques).

### Données et rapports (`data/output/`)
- `df_labeled.csv` — données appariées Excel ↔ images (1007 lignes).
- `df_labeled_with_enc.csv` — idem, avec colonnes encodées pour l'entraînement.
- `images_trouvees.csv`, `images_non_trouvees.csv`, `images_in_folder.csv`, `excel_rows_without_image_name.csv` — diagnostics de l'appariement.
- `evaluation/metrics.xlsx` — métriques par attribut (accuracy, F1) + précision/rappel par classe.
- `evaluation/cm_*.png` — matrices de confusion (une image par attribut, 20 fichiers).
- `logs/training_history.csv` — historique d'entraînement (perte train/val par epoch).
- `comparaison_predit_vs_reel.xlsx` — comparaison prédictions vs vraies valeurs sur échantillon.
- `new_images_report_s01_s05_*.xlsx` — rapport détaillé des 232 nouvelles images testées (prédictions + OCR + logo + score + recommandations).
- `BDD_nouvelles_affiches_a_valider.xlsx` — export incrémental au format de l'équipe (voir §11).
- `logs/*.log` — journaux d'exécution des différents traitements (entraînement, tests, application).

---

## 14. Limites connues

1. **Pondération du score (40/30/20/10) non validée scientifiquement** ni avec le directeur de mémoire — reprise d'un exemple illustratif du plan initial.
2. **Précision de la vérification logo limitée**, y compris sur des marques connues (similarité 0,32 mesurée sur IKEA, sous le seuil de 0,5) — approche pragmatique par recadrage + similarité visuelle générale, pas un détecteur de logo dédié.
3. **Couverture de la référence logo** : 139 marques sur 188 (celles avec ≥ 2 images) ; les marques 2024 sont majoritairement absentes de cette référence construite sur 2019-2023.
4. **Contrainte matérielle** : entraînement en résolution réduite (160×160 au lieu de 224×224) et petit batch size (4), faute de GPU et de RAM suffisante — un environnement mieux équipé permettrait probablement d'affiner les résultats.
5. **Dataset limité et déséquilibré** : 1007 images pour 20 tâches, certaines classes très peu représentées même après fusion — risque de surapprentissage résiduel malgré les corrections apportées.
6. **OCR imparfait sur police stylisée** : quelques caractères mal reconnus sur des typographies publicitaires très stylisées (attendu, propre à tout système OCR généraliste).

---

## 15. Décisions actées avec l'utilisateur (traçabilité pour la rédaction)

- **Exclusion de "Notoriété (base YouGov)"** des cibles du modèle : décision d'équipe, car (a) c'était un score continu mal modélisé en classification (accuracy de 0,15 mesurée avant exclusion), et (b) ce n'est pas un critère de qualité visuelle intrinsèque mais une mesure externe de notoriété de marque — hors périmètre strict de la problématique.
- **Pas de génération d'affiches par IA générative** : évoquée comme piste mais explicitement écartée — hors périmètre de la problématique ("évaluer" ≠ "générer"), et aucune infrastructure de génération d'image disponible.
- **Pas de LLM pour calculer le score** : le score doit rester reproductible et explicable (objectif académique), un LLM pourrait éventuellement reformuler les recommandations en langage naturel plus tard, mais ne doit jamais calculer le score lui-même.
- **`projet_bien_structurer/`** (squelette FastAPI initial) n'a pas été utilisé pour l'application finale : jamais connecté à de vraies données/modèle, l'application a été construite directement dans `jcdecaux_project/` (`app.py`) pour réutiliser le pipeline déjà testé sans duplication de code.

---

## 16. Perspectives (pistes non implémentées, à mentionner en conclusion)

- Faire valider la pondération du score par le directeur/l'équipe métier.
- Traiter le reste de la nouvelle base 2024 (~2330 images).
- Détection d'objets/visages (YOLOv8) pour enrichir les features au-delà du seul embedding CNN global.
- CLIP zero-shot comme baseline de comparaison pour l'esthétique, sans entraînement supervisé supplémentaire.
- Reformulation des recommandations en langage naturel via LLM (habillage textuel uniquement, pas le calcul du score).
