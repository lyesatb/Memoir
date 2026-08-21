"""Score de qualité visuelle composite + recommandations, à partir des
attributs prédits par le CNN et des signaux OCR / logo.

Moteur de règles déterministe (pas de LLM) : chaque sous-score est
calculable et vérifiable manuellement, ce qui est plus adapté à un mémoire
de recherche qu'une boîte noire. Un LLM peut éventuellement reformuler ces
constats en langage naturel plus tard, côté application — il n'intervient
pas dans le calcul du score lui-même.

Pondération reprise du mémoire : Lisibilité 40 % / Contraste 30 % /
Esthétique 20 % / Cohérence 10 %. Les seuils ci-dessous sont des valeurs
de départ raisonnables, à valider avec l'équipe métier (cf. "Livrables
métier" du rapport de suivi).
"""

CONTRASTE_SCORE = {
    "Très contrasté": 1.0,
    "Un peu contrasté": 0.6,
    "Pas contrasté": 0.2,
}

STYLE_TEXTE_SCORE = {
    "Majuscule gras": 1.0,
    "Masjuscule sans gras": 0.8,  # orthographe telle quelle dans la base source
    "Minuscule gras": 0.6,
    "Minuscule sans gras": 0.4,
}

CHARGE_VISUELLE_SCORE = {
    "Faible (3 éléments)": 0.7,
    "Moyen (4-5)": 1.0,
    "Elevée (plus de 6)": 0.5,
}

COULEUR_MARQUE_SCORE = {
    "Dominante": 1.0,
    "Présente": 0.7,
    "Non": 0.4,
}

DEFAULT_SCORE = 0.5  # valeur neutre pour "Non renseigné", "Autre", ou une classe inconnue

# Bornes du ratio de surface occupée par le texte (issu de l'OCR)
OCR_RATIO_IDEAL_MIN = 0.01
OCR_RATIO_IDEAL_MAX = 0.30
OCR_RATIO_CLUTTER_MAX = 0.70  # au-delà, score de lisibilité "texte" à 0

LOGO_CONSISTENT_SCORE = 1.0
LOGO_INCONSISTENT_SCORE = 0.2


def _ocr_ratio_score(ratio):
    if ratio is None:
        return DEFAULT_SCORE
    if ratio < OCR_RATIO_IDEAL_MIN:
        return DEFAULT_SCORE
    if ratio <= OCR_RATIO_IDEAL_MAX:
        return 1.0
    span = OCR_RATIO_CLUTTER_MAX - OCR_RATIO_IDEAL_MAX
    return max(0.0, 1.0 - (ratio - OCR_RATIO_IDEAL_MAX) / span)


def score_poster(predictions, ocr_result=None, logo_result=None):
    """
    predictions : dict {attribut: valeur_prédite_décodée} (sorties du CNN, décodées via encoders.classes_)
    ocr_result : dict retourné par src.ocr.extract_text (optionnel)
    logo_result : dict retourné par src.logo_match.match_logo (optionnel)
    """
    findings = []

    # --- Contraste (30%) ---
    contraste_val = predictions.get("Contraste")
    contraste_score = CONTRASTE_SCORE.get(contraste_val, DEFAULT_SCORE)
    if contraste_score < 0.5:
        findings.append(
            "Contraste jugé insuffisant entre le texte/le sujet et le fond : "
            "envisager d'augmenter l'écart de luminosité ou de couleur pour la lisibilité à distance."
        )

    # --- Lisibilité (40%) : contraste + style de texte + OCR ---
    style_val = predictions.get("Style de texte \n(Lettres)")
    style_score = STYLE_TEXTE_SCORE.get(style_val, DEFAULT_SCORE)
    if style_score < 0.5:
        findings.append(
            "Le style typographique détecté (casse minuscule / sans graissage) réduit la lisibilité "
            "à distance ; privilégier une casse majuscule et un graissage plus marqué (norme AFNOR)."
        )

    ocr_ratio = ocr_result.get("ocr_text_area_ratio") if ocr_result else None
    ocr_score = _ocr_ratio_score(ocr_ratio)
    if ocr_ratio is not None and ocr_ratio > OCR_RATIO_IDEAL_MAX:
        findings.append(
            f"Le texte détecté par OCR occupe environ {ocr_ratio * 100:.0f}% du visuel : "
            "réduire la quantité de texte pour clarifier le message principal."
        )
    if ocr_result is not None:
        qte_texte_val = predictions.get("Quantité/Place du texte dans le visuel")
        if qte_texte_val == "Fort" and ocr_ratio is not None and ocr_ratio < OCR_RATIO_IDEAL_MIN:
            findings.append(
                "Écart entre l'attribut prédit 'Quantité de texte' (Fort) et le texte réellement "
                "détecté par OCR (quasi absent) : à vérifier manuellement."
            )

    lisibilite_score = 0.5 * contraste_score + 0.3 * style_score + 0.2 * ocr_score

    # --- Esthétique (20%) : charge visuelle + cohérence couleur de marque ---
    charge_val = predictions.get("Charge visuelle")
    charge_score = CHARGE_VISUELLE_SCORE.get(charge_val, DEFAULT_SCORE)
    if charge_val == "Elevée (plus de 6)":
        findings.append(
            "Charge visuelle élevée (plus de 6 éléments) : simplifier la composition peut "
            "améliorer l'impact visuel global (cf. Aesthetic Visual Analysis, MIT)."
        )

    couleur_marque_val = predictions.get("Couleur propre à la marque ? ")
    couleur_marque_score = COULEUR_MARQUE_SCORE.get(couleur_marque_val, DEFAULT_SCORE)

    esthetique_score = 0.7 * charge_score + 0.3 * couleur_marque_score

    # --- Cohérence (10%) : logo vs marque déclarée + QR code CNN vs détection ---
    logo_score = DEFAULT_SCORE
    if logo_result and logo_result.get("logo_is_consistent") is not None:
        is_consistent = logo_result["logo_is_consistent"]
        logo_score = LOGO_CONSISTENT_SCORE if is_consistent else LOGO_INCONSISTENT_SCORE
        if not is_consistent:
            findings.append(
                "Le logo détecté dans la zone attendue ne correspond pas au style visuel habituel "
                "de la marque déclarée : vérifier le fichier ou le rattachement de marque."
            )

    qr_score = DEFAULT_SCORE
    if ocr_result is not None and "qr_detected" in (ocr_result or {}):
        qr_detected = ocr_result["qr_detected"]
        qr_predicted = predictions.get("Présence d'un QR Code") == "Oui"
        agree = qr_detected == qr_predicted
        qr_score = 1.0 if agree else 0.3
        if not agree:
            findings.append(
                "Écart entre la présence de QR code annoncée par le modèle et détectée "
                "automatiquement dans l'image : à vérifier."
            )

    coherence_score = 0.7 * logo_score + 0.3 * qr_score

    global_score = 100 * (
        0.40 * lisibilite_score + 0.30 * contraste_score
        + 0.20 * esthetique_score + 0.10 * coherence_score
    )

    if not findings:
        findings.append("Aucun point d'attention majeur détecté sur les critères évalués.")

    return {
        "score_global": round(global_score, 1),
        "score_lisibilite": round(100 * lisibilite_score, 1),
        "score_contraste": round(100 * contraste_score, 1),
        "score_esthetique": round(100 * esthetique_score, 1),
        "score_coherence": round(100 * coherence_score, 1),
        "recommandations": findings,
    }
