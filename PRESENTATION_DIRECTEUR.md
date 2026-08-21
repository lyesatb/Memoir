# Présentation au directeur — IA d'évaluation de la qualité visuelle des affiches JCDecaux

*Document de préparation, à transformer en slides (Gamma). Contient toutes les explications techniques pour pouvoir répondre aux questions.*

---

## 1. Rappel de la problématique

> Comment l'intelligence artificielle peut-elle évaluer automatiquement la qualité visuelle d'une affiche publicitaire selon des critères normés (style, contraste, lisibilité) ?

Objectif : un système qui regarde une affiche et donne un score de qualité visuelle + explique ce qui ne va pas, à partir de critères mesurables (pas d'avis subjectif).

---

## 2. Vue d'ensemble de la solution — le pipeline

```
Affiche (image)
   │
   ├──► Modèle CNN entraîné  ──► 20 attributs visuels (contraste, style du texte, logo, couleurs...)
   ├──► OCR (lecture du texte) ──► texte détecté, quantité de texte, QR code
   ├──► Vérification du logo  ──► le logo correspond-il à la marque annoncée ?
   │
   └──► Moteur de règles ──► Score /100 + recommandations écrites
```

Trois briques d'IA/traitement d'image (CNN, OCR, comparaison visuelle de logo) + un moteur de règles qui combine leurs résultats en un score explicable.

---

## 3. Les données

- Base fournie par JCDecaux : **1165 campagnes publicitaires** (2019-2023), chacune labellisée manuellement par l'équipe sur ~20 critères visuels (contraste, style du texte, taille du logo, couleur dominante, etc.).
- **935 images** récupérées et associées à ces labels → **1007 lignes exploitables** après nettoyage (une campagne peut avoir plusieurs visuels).
- Ces labels manuels sont ce qui permet d'entraîner un modèle qui apprend à reconnaître automatiquement ces mêmes critères sur une nouvelle image.

---

## 4. Le modèle IA — un CNN multi-tâches

**Qu'est-ce qu'un CNN ?** Un réseau de neurones convolutif : un type de modèle de deep learning spécialisé dans l'analyse d'image, qui apprend à reconnaître des motifs visuels (contours, textures, couleurs, formes) directement à partir des pixels.

**Pourquoi "multi-tâches" ?** Au lieu d'entraîner 20 modèles séparés (un par attribut), un seul modèle partage une base commune d'analyse visuelle (le "backbone", ici **EfficientNet-B0**, une architecture standard et éprouvée) puis se ramifie en **20 petites têtes de prédiction**, une par attribut. Avantage : le modèle apprend une représentation visuelle générale de l'affiche, réutilisée pour tous les critères — plus efficace que 20 modèles indépendants.

**Comment il apprend** : on lui montre les 1007 images avec leurs vraies étiquettes (contraste réel, style réel, etc.), il propose une prédiction, on compare à la vraie valeur, on ajuste ses paramètres internes pour réduire l'erreur — répété sur plusieurs passages (epochs) jusqu'à ce que les prédictions se stabilisent (arrêt automatique quand ça ne s'améliore plus : "early stopping", déclenché à l'epoch 20 dans notre cas).

---

## 5. Rendre le modèle robuste — le problème du déséquilibre

**Constat initial** : certains attributs ont des catégories très peu représentées (ex. sur "Personnages", certaines classes n'ont que quelques exemples sur 1007 images). Un modèle standard privilégie les classes majoritaires et devient peu fiable sur les classes rares.

**Deux corrections apportées** :
1. **Fusion des classes rares** : les catégories avec moins de 15 exemples sont regroupées dans une catégorie "Autre" plutôt que d'être ignorées.
2. **Pondération de l'erreur par classe** : pendant l'entraînement, se tromper sur une classe rare "coûte" plus cher au modèle que se tromper sur une classe fréquente — ça force le modèle à apprendre les cas rares aussi.

**Résultat mesuré (avant / après, sur le même jeu de test)** :

| Attribut | F1-score avant | F1-score après |
|---|---|---|
| 1er point d'accroche | 0,19 | **0,80** |
| Couleur dominante | 0,23 | **0,60** |
| Présence de QR Code | 0,49 | **0,90** |
| Langue du claim | 0,45 | **0,80** |
| Mise en avant du prix | 0,60 | **0,92** |

*(F1-score = mesure combinant précision et rappel — plus fiable que le simple taux de bonnes réponses quand les catégories sont déséquilibrées.)*

**À dire clairement** : ce n'est pas l'architecture du modèle qui posait problème, c'était le déséquilibre des données. La correction a permis un gain net et mesurable sur (quasi) tous les attributs.

---

## 6. OCR — lecture automatique du texte

- Utilise **EasyOCR** (bibliothèque de reconnaissance de texte) pour extraire le texte réellement présent sur l'affiche.
- Sert à mesurer : la quantité de texte (proportion de la surface de l'affiche occupée par du texte), et à vérifier la cohérence avec l'attribut "Style de texte" prédit par le CNN.
- Détection de QR code faite séparément avec OpenCV (déterministe, sans IA) — sert de vérification croisée indépendante de ce que prédit le CNN pour cet attribut.
- *Remplace Tesseract (initialement prévu dans la bibliographie) : le binaire Tesseract n'est pas installé sur la machine de développement ; EasyOCR fait le même travail en pur Python.*

---

## 7. Vérification logo/marque

**Problème** : comment vérifier que le logo présent sur l'affiche correspond bien à la marque annoncée ? Il n'existe pas de base de données publique de logos couvrant les ~188 marques spécifiques de ce projet (Free, BUT, Cartier, Lacoste...).

**Solution retenue** : recadrer automatiquement la zone probable du logo (grâce à l'attribut déjà labellisé "Place du logo" — ex. "Bas droit"), puis comparer visuellement ce recadrage à une bibliothèque de référence construite à partir des propres images de chaque marque (139 marques couvertes, celles avec au moins 2 images).

**Limite honnête à présenter** : cette méthode fonctionne mécaniquement, mais sa précision reste modeste — testée sur IKEA (marque bien présente dans la référence), la similarité obtenue n'était que de 0,32 (jugée "incohérente" alors que l'image est bien une vraie affiche IKEA). C'est un recadrage approximatif comparé par similarité visuelle générale, pas un vrai détecteur de logo dédié. À présenter comme une première approche fonctionnelle, pas comme un résultat fiable à 100%.

---

## 8. Le score final et les recommandations — comment c'est calculé

**Important à expliquer clairement au directeur** : ce n'est **pas une IA qui décide du score**. C'est un **moteur de règles fixes**, écrit à la main, qui combine les résultats du CNN + OCR + logo selon une formule explicite et vérifiable.

**Étape 1** — les 20 attributs prédits par le CNN sont convertis en notes numériques fixes (exemples) :
- Contraste : "Très contrasté" → 1,0 / "Un peu contrasté" → 0,6 / "Pas contrasté" → 0,2
- Style de texte : "Majuscule gras" → 1,0 / "Minuscule sans gras" → 0,4
- Charge visuelle : "Moyen" → 1,0 (jugé optimal) / "Élevée" → 0,5 (jugé trop chargé)

**Étape 2** — ces notes sont combinées en 4 sous-scores :
- **Lisibilité** = Contraste (50%) + Style du texte (30%) + quantité de texte OCR (20%)
- **Contraste** = directement l'attribut prédit
- **Esthétique** = Charge visuelle (70%) + cohérence couleur/marque (30%)
- **Cohérence** = correspondance logo/marque (70%) + accord QR code prédit vs détecté (30%)

**Étape 3** — score global sur 100 :
```
Score = 40% × Lisibilité + 30% × Contraste + 20% × Esthétique + 10% × Cohérence
```
*(Cette pondération 40/30/20/10 vient du plan de mémoire initial — présentée comme un exemple illustratif au départ, **pas encore validée avec le directeur**. Point à trancher ensemble.)*

**Étape 4** — recommandations : des règles "si...alors..." sur ces mêmes valeurs (ex. *si Contraste = "Pas contrasté" → conseiller d'augmenter le contraste*).

**Pourquoi un système de règles et pas un LLM/IA générative pour le score** : reproductible (même image = même score à chaque fois), explicable (chaque note se justifie), cohérent avec l'objectif du mémoire (évaluation *objective*, pas subjective). Un LLM pourrait éventuellement reformuler les recommandations en phrases plus naturelles plus tard, mais ne doit pas calculer le score lui-même.

---

## 9. L'application (démonstration possible en direct)

Une interface (Streamlit) où on dépose une affiche et on obtient immédiatement : le score, les 4 sous-scores, les recommandations, la vérification du logo, le texte détecté, et le détail des 20 attributs. Un bouton permet aussi de télécharger un export Excel dans le même format que la base de données existante de l'équipe, pour intégration dans leur processus d'étiquetage.

---

## 10. Test sur de nouvelles images (2024)

Le système a été testé sur ~230 nouvelles images (2024, hors entraînement) fournies après coup : traitement automatique réussi, scores cohérents et différenciés (39 à 95/100 selon les affiches), démontrant que le pipeline fonctionne au-delà des données d'entraînement — pas seulement par cœur sur les images déjà vues.

---

## 11. Limites à assumer devant le directeur (transparence = force académique)

1. **La pondération du score (40/30/20/10) n'est pas validée scientifiquement** — reprise d'un exemple illustratif du plan initial. À discuter : la garder telle quelle avec justification, ou la faire évoluer avec le directeur/l'équipe métier.
2. **La vérification logo a une précision limitée**, y compris sur des marques connues (cf. IKEA) — approche pragmatique (pas de base de logos disponible), pas un détecteur de logo dédié.
3. **"Notoriété (base YouGov)" a été retirée** du périmètre — c'était une mesure externe de notoriété de marque, pas un critère de qualité visuelle intrinsèque, et mal modélisée en classification (score continu).
4. **Contrainte matérielle** : entraînement fait sur CPU avec peu de RAM → résolution d'image réduite (160×160 au lieu de 224×224). Un environnement avec GPU/plus de RAM permettrait probablement d'affiner encore les résultats.

---

## 12. Prochaines étapes

- Valider la pondération du score avec le directeur / l'équipe métier.
- Traiter le reste de la nouvelle base 2024 (~2300 images restantes).
- Éventuellement : reformulation des recommandations en langage naturel (LLM, optionnel, seulement pour l'affichage texte).
