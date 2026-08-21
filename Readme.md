# Analyse du projet

**Sujet** : Vers une IA d'évaluation qualitative des visuels publicitaires : analyse d'image, style, contraste et visibilité
**Auteur** : AIT TAYEB Lyes
**Objet de ce document** : synthèse factuelle du travail technique déjà réalisé (code, données, résultats), à utiliser comme matière première pour structurer et rédiger le mémoire.

---

## 1. Contexte et problématique (rappel du sujet officiel)

> Comment l'intelligence artificielle peut-elle être utilisée pour évaluer automatiquement la qualité visuelle d'une affiche publicitaire selon des critères normés comme le style, le contraste ou la lisibilité ?

Le projet vise à concevoir un scoring automatique de la qualité visuelle des affiches en combinant computer vision et deep learning, à partir de critères mesurables : contraste, lisibilité du texte, saturation des couleurs, simplicité du style graphique, etc.

### Bibliographie de référence fournie

| Source | Usage prévu |
|---|---|
| Papers with Code – Image Classification | Repérage des meilleurs modèles |
| CLIP (OpenAI) | Compréhension image-texte |
| Deep Learning for Computer Vision – A. Rosebrock | Référence traitement d'image en Python |
| OpenCV | Mesure contrastes, formes, couleurs |
| Image Quality Assessment – Survey | Méthodes de scoring qualité image |
| Google AutoML Vision | Modèle custom |
| Kaggle Dataset - Advertisement Images | Données d'entraînement/test |
| YOLOv8 | Détection logo, texte, visages |
| Scikit-image | Traitement d'image scientifique |
| Aesthetic Visual Analysis – MIT | Étude du style visuel |
| Vision Transformers (ViT) | Modèle de vision récent |
| HuggingFace – Image classification models | Modèles pré-entraînés |
| Color theory in advertising – Nielsen | Impact des couleurs |
| Contraste et lisibilité – AFNOR | Normes de lisibilité |
| Tesseract OCR | Extraction de texte / lisibilité |
| PyTorch Image Models (TIMM) | Modèles de classification |

---

## 2. Ce qui a été réellement construit

Le dépôt contient **deux implémentations** du même pipeline :

- **`jcdecaux_project/`** — version opérationnelle : données réelles, modèle entraîné (`artifacts/best_model.pth`), encodeurs sauvegardés, pipeline robuste + OCR + détection de logo (v2, voir §2.1).

### 2.1 Pipeline réel (v2 — après robustesse + OCR + logo)

1. **Ingestion** — images téléchargées depuis SharePoint (bot Python), stockées dans `data/input/Visuale/`.
2. **Matching Excel ↔ images** (`src/excel_matching.py`) — normalisation des noms de fichiers (suppression accents, minuscules, nettoyage espaces/quotes) puis appariement tolérant : correspondance exacte → inclusion partielle → essai d'extensions.
3. **Encodage robuste** (`src/encoding.py`) — 20 attributs cibles (Notoriété exclue), fusion des classes rares (< 15 occurrences) dans une catégorie "Autre" par colonne, `LabelEncoder`, poids de classe inverse-fréquence par tâche sauvegardés (`artifacts/class_weights.pkl`). Le DataFrame encodé est persisté sur disque (`data/output/df_labeled_with_enc.csv`) plutôt que transmis par état mutable partagé entre imports (source d'un bug de conception dans la v1).
4. **Dataset PyTorch** (`src/dataset.py`) — `MultiTaskDataset` renvoyant `(image, dict_targets, chemin)`, avec augmentations (flip horizontal, rotation ±10°, jitter couleur) et split 80/10/10.
5. **Modèle** (`src/model.py`) — CNN **multi-tâches** : backbone `EfficientNet-B0` partagé, puis une tête linéaire par attribut cible (`nn.ModuleDict`). Résolution réduite à 160×160 et batch_size=4 (contrainte mémoire de la machine d'entraînement, ~8 Go RAM).
6. **Entraînement** (`src/train.py`) — AdamW, `ReduceLROnPlateau`, early stopping (patience 6), **perte pondérée par classe par tâche** (`CrossEntropyLoss(weight=...)`) pour compenser le déséquilibre.
7. **Évaluation** (`src/evaluate.py`) — accuracy, F1 pondéré **et désormais précision/rappel par classe** (`classification_report`), matrices de confusion exportées.
8. **OCR** (`src/ocr.py`) — extraction de texte (EasyOCR), ratio de surface occupée par le texte, casse dominante ; détection de QR code déterministe (`cv2.QRCodeDetector`, sans entraînement).
9. **Cohérence logo/marque** (`src/logo_match.py`) — recadrage de la zone de logo via l'attribut labellisé `Place du logo`, embedding par le backbone entraîné, bibliothèque de référence par marque, comparaison par similarité cosinus.
10. **Score + recommandations** (`src/recommend.py`) — moteur de règles déterministe (sans LLM), sous-scores Lisibilité/Contraste/Esthétique/Cohérence pondérés 40/30/20/10, score global `/100`, constats textuels en français.
11. **Inférence batch** (`src/inference.py::predict_batch`) — traite un dossier de nouvelles images de bout en bout (CNN + OCR + logo + score) et exporte un rapport Excel consolidé — c'est la fonction prévue pour traiter la nouvelle base d'images à venir.
12. **API** (`api/main.py`) — inchangée pour l'instant (phase application, plus tard), corrigée uniquement pour rester compatible avec le nouveau format de checkpoint.

Cette architecture correspond à l'approche "computer vision + deep learning multi-critères" du sujet, avec un angle plus riche que le scoring 0-100 générique : le modèle prédit **20 attributs descriptifs individuels de l'affiche**, enrichis par de l'OCR et une vérification de cohérence logo/marque, combinés en un score composite explicable.

### 2.2 Stack technique

`torch` 2.3 / `torchvision` 0.18 (EfficientNet-B0, CPU), `pandas`, `scikit-learn` (LabelEncoder, métriques), `opencv-python` (QR code, traitement d'image), `easyocr` (OCR — remplace Tesseract, dont le binaire n'est pas installé sur la machine de développement), `Pillow`, `openpyxl`, `FastAPI` + `uvicorn`, `matplotlib`, `tqdm`.

---

## 3. Les données (chiffres réels à citer)

- Base Excel : **1165 lignes** (campagnes/visuels) — **1147 avec code retrouvé côté images**, 18 sans image.
- **935 images téléchargées**, toutes associées à un code (0 orpheline).
- Après matching final : **1007 lignes labellisées avec image** exploitées pour l'entraînement (`df_labeled.csv`), 231 images non appariées.
- Poids moyen d'image : 0,42 Mo (min 45 Ko, max 9,6 Mo) ; formats `.jpg` (940) et `.png` (13).
- Répartition très déséquilibrée par famille produit : BOISSONS (145), DISTRIBUTION (72), MODE_ET_ACCESSOIRES (71), HYGIENE_BEAUTE (65)... jusqu'à **APPAREILS_MENAGERS (2 images) et INDUSTRIE (5 images)**.
- **20 attributs cibles** retenus (voir §5 sur l'exclusion de la Notoriété) : Type de campagne, Couleur dominante, Couleur propre à la marque, 1er point d'accroche, Personnages, Egérie (propre/autre marque), Utilisation de l'égérie dans le temps, Genre, Tranche d'âge, Quantité/Place du texte, Discours utilisé, Style de texte, Taille du logo, Éléments de branding, Présence QR Code, Langue du claim, Contraste, Place du logo, Mise en avant prix, Charge visuelle.

---

## 4. Résultats obtenus — avant / après robustesse

Le pipeline v2 (classes rares fusionnées + perte pondérée par classe) a été entraîné et évalué. Comparaison directe sur le même jeu de test :

| Attribut | Accuracy avant | F1 avant | Accuracy après | F1 après |
|---|---|---|---|---|
| Présence d'un QR Code | 0.97 | 0.49 | 0.90 | **0.90** |
| Langue du claim | 0.80 | 0.45 | 0.79 | **0.80** |
| Mise en avant prix | 0.74 | 0.60 | 0.92 | **0.92** |
| Charge visuelle | 0.56 | 0.53 | 0.72 | **0.71** |
| Contraste | 0.58 | 0.50 | 0.62 | **0.62** |
| Couleur dominante | 0.35 | 0.23 | 0.60 | **0.60** |
| 1er point d'accroche | 0.58 | 0.19 | 0.79 | **0.80** |

Autres attributs (v2 uniquement, non mesurés dans le premier rapport) : Type de campagne (F1 0.73), Personnages (0.64), Genre (0.78), Tranche d'âge (0.74), Taille du logo (0.72), Style de texte (0.68) — le détail complet des 20 attributs est dans `jcdecaux_project/data/output/evaluation/metrics.xlsx` (deux feuilles : résumé par attribut, précision/rappel par classe).

### Lecture critique (section "Résultats et limites" du mémoire)

- Le F1 est resté la métrique de référence (pas l'accuracy brute) : sur "Présence d'un QR Code", l'accuracy a même légèrement baissé (0.97 → 0.90) alors que le F1 a presque doublé (0.49 → 0.90) — la v1 se contentait de prédire la classe majoritaire ("Non"), la v2 discrimine réellement les deux classes.
- Le gain le plus spectaculaire est sur les attributs à classes très déséquilibrées avant fusion : `1er point d'accroche` (F1 0.19 → 0.80) et `Couleur dominante` (F1 0.23 → 0.60) — confirmant que le déséquilibre de classes, pas l'architecture, était le facteur limitant principal.
- Modèle entraîné à résolution 160×160 (au lieu de 224×224) et batch_size=4, contrainte par la mémoire disponible sur la machine de développement (~8 Go RAM) — à mentionner comme limite technique ; une machine avec plus de RAM/GPU permettrait probablement d'affiner encore ces résultats à résolution native.
- Arrêt anticipé déclenché à l'epoch 20 (meilleur modèle : epoch 14, val_loss=0.84), sur un budget de 30 epochs maximum — le modèle n'a pas eu besoin du budget complet.

---

## 5. Décision méthodologique : exclusion de "Notoriété (base YouGov)"

**Cette variable est retirée du périmètre d'évaluation du mémoire.**

Justification à formuler dans la section "Limites" :

1. C'était en réalité un score continu (0 à 1, ex. 0.74, 0.89) mal modélisé en classification catégorielle — d'où l'accuracy très faible observée (0.15) lors des premiers tests.
2. Plus fondamentalement, contrairement aux 20 autres attributs, cette variable ne décrit pas la qualité visuelle intrinsèque de l'affiche mais une mesure externe de notoriété de marque (donnée YouGov) — hors du périmètre strict de la problématique ("évaluer la qualité visuelle selon des critères normés comme le style, le contraste, la lisibilité").

Le tableau de résultats du mémoire passe donc de 21 à **20 attributs cibles**, tous directement liés à la qualité visuelle, ce qui renforce la cohérence du mémoire avec sa problématique.

---

## 6. OCR et cohérence logo/marque — résultats

- **OCR (EasyOCR)** : testé sur un échantillon, extrait correctement le texte principal des affiches (ex. "TIFFANY&CO. JCDecaux", "JCDecaux ENQUÊTER..."), avec les limites attendues d'un OCR sur police stylisée/publicitaire (quelques caractères mal reconnus). Le ratio de surface occupée par le texte (`ocr_text_area_ratio`) varie de 0.037 à 0.145 sur l'échantillon testé — cohérent avec des affiches où le texte est un élément parmi d'autres, pas le sujet principal.
- **Détection de QR code** (`cv2.QRCodeDetector`) : déterministe, aucun faux positif observé sur l'échantillon (aucune des 5 images testées n'avait de QR code, toutes correctement détectées comme telles).
- **Référence logo** : construite sur 139 marques (sur 188, celles avec ≥ 2 images), 61 lignes ignorées (image manquante ou marque non renseignée). Sur l'échantillon testé, le logo détecté correspond en similarité cosinus de 0.56 à 0.78 selon la marque — la marque avec le plus d'images de référence (BUT, Free) donnera mécaniquement une référence plus fiable que celles avec seulement 2 images, point à documenter comme limite (few-shot logo verification).
- **Score composite + recommandations** (`src/recommend.py`) : scores obtenus sur l'échantillon testé entre 77 et 92,5 / 100, avec déclenchement correct de la règle "Charge visuelle élevée" sur l'image qui en avait besoin — le moteur de règles fonctionne comme prévu, sans LLM.

Point d'attention technique : le module OCR a nécessité un correctif d'encodage (la console Windows par défaut, cp1252, ne supporte pas les caractères Unicode utilisés par la barre de progression d'EasyOCR) — corrigé en forçant stdout/stderr en UTF-8 dans `src/ocr.py`.

---

## 7. Structure de mémoire proposée

1. **Introduction** — contexte de l'affichage publicitaire, enjeu de la qualité visuelle, objectif du scoring automatique.
2. **État de l'art** — bibliographie fournie (CLIP, ViT, YOLOv8, Tesseract, AFNOR, Nielsen, TIMM...), présentée comme fondement théorique choisi avant implémentation.
3. **Données et méthodologie** — le vrai pipeline (ingestion SharePoint → matching Excel/images → encodage robuste → dataset multi-tâches → modèle EfficientNet-B0 multi-têtes → entraînement pondéré → OCR → cohérence logo → score composite), avec les vrais chiffres (1165 lignes, 935 images, 20 attributs).
4. **Résultats** — tableau de métriques avant/après robustesse par attribut (§4), résultats OCR/logo (§6).
5. **Limites et pistes d'amélioration** — section honnête, transformée en feuille de route (voir §8).
6. **Conclusion et perspectives** — étape suivante : application utilisateur (upload affiche → score + explications), traitement de la nouvelle base d'images à venir.

---

## 8. Application utilisateur

**Réalisée** : `jcdecaux_project/app.py`, interface Streamlit (déjà installée, aucune dépendance ajoutée à configurer). Lancement : `python -m streamlit run app.py` depuis `jcdecaux_project/`, puis ouvrir `http://localhost:8501`.

- L'utilisateur dépose une affiche (+ optionnellement le nom de la marque annoncée) et obtient : le score global `/100`, les 4 sous-scores (Lisibilité/Contraste/Esthétique/Cohérence), la liste des recommandations, le résultat de la vérification logo/marque, le texte OCR détecté, et le détail des 20 attributs prédits.
- La logique métier (`analyze_poster()`) est séparée de l'affichage Streamlit, ce qui permet de la tester directement en script sans interface — validé sur une image connue (marque M6 : logo cohérent, similarité 0.78, score 95.8/100) et sur une nouvelle image (marque BOMPARD, absente de la référence : le système répond honnêtement "pas de comparaison possible" plutôt que d'inventer une cohérence).
- Gestion d'erreur testée : un fichier non-image est rejeté proprement (message d'erreur, pas de crash).
- **Limite connue** : la vérification logo ne peut juger que les marques déjà présentes dans la référence construite à partir des données 2019-2023 (voir §6) ; pour les nouvelles marques 2024, seul le score global (indépendant du logo) reste pleinement fiable.

Décidé et non implémenté (voir aussi §7 pour l'historique) : reformulation des recommandations par LLM et génération d'affiches par prompt — écartés du périmètre "évaluation de la qualité visuelle" du mémoire, à ne mentionner qu'en perspective de conclusion si souhaité.

---

## 9. Fichiers clés du dépôt (pour référence rapide)

```
jcdecaux_project/
├── app.py                    # application Streamlit (dépôt d'affiche → score + recommandations)
├── config/config.py          # chemins, hyperparamètres, seuils, mapping zones logo
├── src/excel_matching.py     # appariement Excel ↔ images
├── src/encoding.py           # cibles + fusion classes rares + poids de classe
├── src/dataset.py            # Dataset PyTorch multi-tâches + augmentations
├── src/model.py              # CNN multi-tâches (EfficientNet-B0 / ResNet50)
├── src/train.py              # entraînement, perte pondérée par classe
├── src/evaluate.py           # métriques par tâche + précision/rappel par classe
├── src/ocr.py                # extraction de texte (EasyOCR) + détection QR code
├── src/logo_match.py         # référence logo par marque + matching par similarité
├── src/recommend.py          # score composite /100 + recommandations (règles)
├── src/inference.py          # prédiction unitaire + predict_batch (nouvelles images)
├── src/img_utils.py          # ouverture d'image robuste (haute résolution, formats variés)
├── src/compare_predictions.py # comparaison prédit vs réel sur images déjà labellisées
├── api/main.py                # API FastAPI /predict (non utilisée par l'app Streamlit)
└── artifacts/                # best_model.pth, encoders.pkl, class_weights.pkl, logo_reference.pkl


```
