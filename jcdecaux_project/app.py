"""Application Streamlit : dépose une affiche, obtiens un score de qualité
visuelle, ses points d'attention et des recommandations.

Couche fine au-dessus du pipeline déjà testé (src/inference.py, src/ocr.py,
src/logo_match.py, src/recommend.py) — aucune logique de scoring ici.

Lancement : python -m streamlit run app.py
"""
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from config.config import LOGO_REFERENCE_PATH
from src import logo_match
from src import ocr as ocr_module
from src import recommend
from src import team_export
from src.img_utils import safe_open_rgb
from src.inference import get_model, predict_image, guess_brand_from_filename

st.set_page_config(page_title="Score de qualité visuelle — affiches JCDecaux", page_icon="🖼️")


@st.cache_resource(show_spinner="Chargement du modèle...")
def load_resources():
    get_model()  # charge et met en cache le modèle + les encodeurs
    if LOGO_REFERENCE_PATH.exists():
        return logo_match.load_logo_reference()
    return None


def analyze_poster(image_path, declared_brand=None, reference=None):
    """Logique métier pure (sans Streamlit) : image -> prédictions + OCR + logo + score.
    Séparée de l'UI pour rester testable en dehors d'une session Streamlit."""
    predictions = predict_image(image_path)

    try:
        ocr_result = ocr_module.extract_text(image_path)
        ocr_result["qr_detected"] = ocr_module.detect_qr_code(image_path)
    except Exception as e:
        ocr_result = {"ocr_error": str(e)}

    logo_result = {}
    if reference:
        try:
            logo_result = logo_match.match_logo(
                image_path,
                declared_brand=declared_brand or None,
                place_du_logo_label=predictions.get("Place du logo"),
                reference=reference,
            )
        except Exception as e:
            logo_result = {"logo_error": str(e)}

    scoring = recommend.score_poster(predictions, ocr_result=ocr_result, logo_result=logo_result)
    return predictions, ocr_result, logo_result, scoring


reference = load_resources()

st.title("Score de qualité visuelle d'une affiche")
st.write(
    "Dépose une affiche publicitaire pour obtenir un score de qualité visuelle, "
    "ses points d'attention et des recommandations."
)

uploaded_file = st.file_uploader("Affiche", type=["jpg", "jpeg", "png", "bmp", "gif", "jfif"])

if uploaded_file is not None:
    guessed_brand = guess_brand_from_filename(Path(uploaded_file.name).stem)
    declared_brand = st.text_input(
        "Marque annoncée sur l'affiche (devinée depuis le nom du fichier — corrige si besoin)",
        value=guessed_brand,
    )
    suffix = Path(uploaded_file.name).suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    try:
        try:
            img_preview = safe_open_rgb(tmp_path)
        except Exception as e:
            st.error(f"Impossible de lire cette image : {e}")
            st.stop()

        st.image(img_preview, caption=uploaded_file.name)

        with st.spinner("Analyse en cours..."):
            try:
                predictions, ocr_result, logo_result, scoring = analyze_poster(
                    tmp_path, declared_brand=declared_brand, reference=reference
                )
            except Exception as e:
                st.error(f"Erreur lors de la prédiction : {e}")
                st.stop()

            df_single = pd.DataFrame([{
                "image": uploaded_file.name,
                "marque_declaree": declared_brand,
                **predictions,
            }])
            team_export.export_team_format(df_single)

        st.subheader(f"Score global : {scoring['score_global']:.0f} / 100")
        cols = st.columns(4)
        cols[0].metric("Lisibilité", f"{scoring['score_lisibilite']:.0f}")
        cols[1].metric("Contraste", f"{scoring['score_contraste']:.0f}")
        cols[2].metric("Esthétique", f"{scoring['score_esthetique']:.0f}")
        cols[3].metric("Cohérence", f"{scoring['score_coherence']:.0f}")

        st.subheader("Recommandations")
        for rec in scoring["recommandations"]:
            st.markdown(f"- {rec}")

        if logo_result.get("logo_best_match"):
            st.subheader("Logo")
            st.write(
                f"Marque la plus proche visuellement : **{logo_result['logo_best_match']}** "
                f"(similarité {logo_result['logo_best_score']:.2f})"
            )
            if logo_result.get("logo_declared_score") is not None:
                consistent = "cohérent" if logo_result["logo_is_consistent"] else "incohérent"
                st.write(
                    f"Comparaison à la marque déclarée ({declared_brand}) : **{consistent}** "
                    f"(similarité {logo_result['logo_declared_score']:.2f})"
                )
        elif reference is None:
            st.caption("Vérification de logo indisponible : aucune référence construite "
                       "(lancer `python -m src.logo_match`).")

        if ocr_result.get("ocr_text"):
            st.subheader("Texte détecté (OCR)")
            st.write(ocr_result["ocr_text"])
        elif ocr_result.get("ocr_error"):
            st.caption(f"OCR indisponible sur cette image : {ocr_result['ocr_error']}")

        with st.expander("Voir le détail des 20 attributs prédits"):
            df_pred = pd.DataFrame(list(predictions.items()), columns=["Attribut", "Valeur prédite"])
            st.table(df_pred)

        st.divider()
        with open(team_export.TEAM_FORMAT_PATH, "rb") as f:
            st.download_button(
                "Télécharger le fichier Excel (format équipe, mis à jour)",
                data=f.read(),
                file_name=team_export.TEAM_FORMAT_PATH.name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    finally:
        Path(tmp_path).unlink(missing_ok=True)
