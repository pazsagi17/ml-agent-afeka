# -*- coding: utf-8 -*-
"""
Smart Sentiment & Spam Analysis Agent — Streamlit demo for ML coursework.
Uses sklearn Pipelines: TfidfVectorizer + LogisticRegression on synthetic data.
"""

from __future__ import annotations

import random
from typing import List, Tuple, cast

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Synthetic datasets (immediate run — no external files)
# ---------------------------------------------------------------------------

def _spam_ham_data() -> Tuple[List[str], List[int]]:
    """0 = legitimate (ham), 1 = spam."""
    ham = [
        "היי, נתראה מחר ב־10 בבוקר בקמפוס",
        "שלום, שלחתי את המצגת למייל שלך",
        "Thanks for the notes from last class",
        "Can we reschedule the lab to Thursday?",
        "הפרויקט נראה מצוין, תודה על העדכון",
        "I'll upload the dataset to the shared drive",
        "פגישת צוות ביום שני אחרי ההפסקה",
        "Reminder: assignment due next week",
        "נשמח לעזור אם משהו לא ברור",
        "See you at Afeka for the workshop",
        "הקוד רץ אצלי בלי שגיאות",
        "Could you review section 3 of the report?",
        "שלחתי את הקובץ CSV בצירוף",
        "Let's meet in the library after lunch",
        "תודה על ההסבר בכיתה היום",
        "The model converged after 200 epochs",
        "אעדכן כשאסיים את הניסויים",
        "Group 4 presentation is ready",
        "נתאם לגבי מועד ההגשה",
        "I pushed the fix to the main branch",
        "הנתונים נקיים יחסית הפעם",
        "Office hours moved to 3pm",
        "שמחתי לשמוע שהבחינה עברה טוב",
        "Please confirm receipt of this message",
        "נשלח עדכון נוסף בהמשך השבוע",
    ]
    spam = [
        "WINNER!!! Click here NOW to claim your FREE iPhone!!!",
        "URGENT: You have won $1,000,000 — send bank details",
        "Buy cheap pills online 80% OFF click link bit.ly/fake123",
        "CONGRATULATIONS you are selected!!! Act fast limited time",
        "זכית בפרס!!! לחץ כאן מיד לאישור הזהות",
        "FREE crypto airdrop — connect wallet now",
        "Act now: luxury watches replica 90% discount",
        "You must verify your account immediately or it will be closed",
        "Hot singles in your area click this link",
        "מבצע בלעדי!!! רק היום קנה 2 קבל 5",
        "Earn $5000 per week from home — no experience needed",
        "Your PayPal is locked. Click to restore access",
        "לחץ כאן לקבלת בונוס מיידי בלי בדיקת אשראי",
        "100% guaranteed weight loss miracle pill",
        "Limited offer: cheap loans no credit check",
        "Your parcel is held — pay customs fee via this link",
        "ניצחת בהגרלה העולמית, השאר פרטי כרטיס אשראי",
        "Subscribe now for exclusive adult content",
        "FINAL NOTICE: invoice attached open document.exe",
        "Cheap SEO traffic bots buy followers Instagram",
        "לחץ לאישור זכייה בחופשה לדובאי",
        "You inherited money from unknown relative contact lawyer",
        "Double your bitcoin in 24 hours guaranteed",
        "מכירת פומבית — iPhone במחיר אפסי בלבד!!!",
        "Claim your reward card before midnight tonight!!!",
    ]
    texts = ham + spam
    labels = [0] * len(ham) + [1] * len(spam)
    return texts, labels


def _sentiment_data() -> Tuple[List[str], List[str]]:
    """Labels: negative, neutral, positive."""
    negative = [
        "This lecture was confusing and too fast",
        "I'm disappointed with my grade on the exam",
        "השירות היה איטי מאוד ולא מקצועי",
        "The app crashed three times during the demo",
        "לא אהבתי את הממשק, מבולגן וקשה לשימוש",
        "Worst customer support experience ever",
        "התוצאות של המודל היו גרועות מאוד",
        "Too many bugs, not ready for production",
        "ההסבר לא היה ברור ואיבדתי את הקשר",
        "The dataset is noisy and full of errors",
        "לא ממליץ על הספר הזה, מלא טעויות",
        "Training took forever and still didn't converge",
        "הבחינה הייתה קשה מדי ביחס לחומר",
        "Poor documentation, wasted a lot of time",
        "לא עבד אצלי בכלל על המחשב שלי",
    ]
    neutral = [
        "The class covered standard ML topics",
        "We used logistic regression for the baseline",
        "המצגת הייתה באורך סביר",
        "The assignment instructions were clear enough",
        "עבדנו עם pandas ו־sklearn",
        "Lab session was about text preprocessing",
        "The professor answered questions at the end",
        "הפרויקט דורש עיבוד טקסט ו־TF-IDF",
        "We split the data 80/20 train test",
        "The UI is built with Streamlit",
        "דיברנו על overfitting ו-regularization",
        "Submission is through the course portal",
        "הקוד צריך להישלח עד מועד ההגשה",
        "Group work of three students maximum",
        "We presented results in a short report",
    ]
    positive = [
        "Excellent course — learned a lot!",
        "ההסבר היה ברור ומעניין מאוד",
        "Great project idea and smooth demo",
        "אהבתי את הגישה הפרקטית ל־ML",
        "The professor's examples were very helpful",
        "ממש נהניתי מהעבודה על הסוכן החכם",
        "Streamlit made the UI easy and fast",
        "המודל עבד מעל הציפיות שלי",
        "Amazing teamwork and clean codebase",
        "תודה רבה על המשוב המפורט",
        "Best ML assignment I've done this year",
        "הוויזואליזציה של הביטחון עזרה להבין",
        "Really impressed with sklearn pipelines",
        "ממליץ בחום על הקורס לחברים",
        "Clear slides and good pacing overall",
    ]
    texts = negative + neutral + positive
    labels = (
        ["negative"] * len(negative)
        + ["neutral"] * len(neutral)
        + ["positive"] * len(positive)
    )
    return texts, labels


def build_spam_pipeline(random_state: int = 42) -> Tuple[Pipeline, float, float]:
    X, y = _spam_ham_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    return pipe, train_acc, test_acc


def build_sentiment_pipeline(random_state: int = 42) -> Tuple[Pipeline, float, float]:
    X, y = _sentiment_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
    max_iter=2000,
    random_state=42, # או ה-seed שמוגדר שם
    solver="lbfgs"
),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    train_acc = pipe.score(X_train, y_train)
    test_acc = pipe.score(X_test, y_test)
    return pipe, train_acc, test_acc


@st.cache_resource
def load_models(_seed: int = 42):
    """Cache fitted pipelines for snappy UI."""
    spam_pipe, spam_tr, spam_te = build_spam_pipeline(_seed)
    sent_pipe, sent_tr, sent_te = build_sentiment_pipeline(_seed)
    return {
        "spam": spam_pipe,
        "spam_train_acc": spam_tr,
        "spam_test_acc": spam_te,
        "sentiment": sent_pipe,
        "sent_train_acc": sent_tr,
        "sent_test_acc": sent_te,
    }


def render_proba_bars(
    labels: List[str],
    probs: np.ndarray,
    title: str,
) -> None:
    """Visual confidence scores from predict_proba."""
    st.markdown(f"**{title}**")
    order = np.argsort(-probs)
    for idx in order:
        lab = labels[idx]
        p = float(probs[idx])
        st.progress(
            min(max(p, 0.0), 1.0),
            text=f"{lab}: {p * 100:.1f}%",
        )


def main() -> None:
    st.set_page_config(
        page_title="סוכן ניתוח סנטימנט וספאם",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "input" not in st.session_state:
        st.session_state.input = ""
    force_analyze = bool(st.session_state.pop("force_analyze", False))

    # --- Sidebar (Hebrew + English) ---
    st.sidebar.markdown("## ⚙️ הגדרות / Settings")
    st.sidebar.markdown(
        "_פרויקט הדגמה — Afeka College ML Course_"
    )
    seed = st.sidebar.number_input(
        "Random seed (לאימון חוזר)",
        min_value=0,
        max_value=9999,
        value=42,
        step=1,
    )
    show_dataset = st.sidebar.checkbox("הצג דוגמאות מהמאגר הסינתטי", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**מודלים:** Pipeline של `TfidfVectorizer` + `LogisticRegression`"
    )
    st.sidebar.markdown(
        "- **ספאם:** סיווג בינארי (ham / spam)\n"
        "- **סנטימנט:** שלושה מחלקות (שלילי / נייטרלי / חיובי)"
    )

    models = load_models(int(seed))

    # --- Header ---
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.title("🤖 סוכן חכם: סנטימנט וספאם")
        st.caption("Smart Sentiment & Spam Analysis Agent — sklearn + Streamlit")
    with c2:
        st.metric("דיוק ספאם (בדיקה)", f"{models['spam_test_acc']:.0%}")
    with c3:
        st.metric("דיוק סנטימנט (בדיקה)", f"{models['sent_test_acc']:.0%}")

    st.markdown(
        "הזינו טקסט חופשי (עברית או אנגלית). המערכת מציגה **הסתברויות** "
        "לכל מחלקה באמצעות `predict_proba`."
    )

    if show_dataset:
        with st.expander("דוגמאות סינתטיות — ספאם / לא ספאם", expanded=False):
            sx, sy = _spam_ham_data()
            st.dataframe(
                pd.DataFrame({"טקסט": sx, "תווית (0=תקין, 1=ספאם)": sy}),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("דוגמאות סינתטיות — סנטימנט", expanded=False):
            tx, ty = _sentiment_data()
            st.dataframe(
                pd.DataFrame({"טקסט": tx, "סנטימנט": ty}),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
# 1. קודם כל יוצרים את העמודות והכפתורים
    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        analyze = st.button("🔍 נתח עכשיו", type="primary", use_container_width=True)
    with col_b:
        sample_spam = st.button("דוגמת ספאם", use_container_width=True)
    with col_c:
        sample_ok = st.button("דוגמה תקינה", use_container_width=True)

# 2. רק עכשיו בודקים אם לחצו עליהם (הלוגיקה)
    if sample_spam:
        texts_spam, labels_spam = _spam_ham_data()
        spam_pool = [t for t, y in zip(texts_spam, labels_spam) if y == 1]
        st.session_state.input = random.choice(spam_pool)
        st.session_state.force_analyze = True
        st.rerun()
    elif sample_ok:
        texts_spam, labels_spam = _spam_ham_data()
        ham_pool = [t for t, y in zip(texts_spam, labels_spam) if y == 0]
        st.session_state.input = random.choice(ham_pool)
        st.session_state.force_analyze = True
        st.rerun()

# 3. ורק בסוף מציגים את תיבת הטקסט
    st.text_area(
        "📝 טקסט לניתוח",
        value=st.session_state.input,
        height=160,
        key="input"
)

    texts_spam, labels_spam = _spam_ham_data()
    spam_pool = [t for t, y in zip(texts_spam, labels_spam) if y == 1]
    ham_pool = [t for t, y in zip(texts_spam, labels_spam) if y == 0]
    if sample_spam:
        st.session_state.input = random.choice(spam_pool)
        st.session_state.force_analyze = True
        st.rerun()
    elif sample_ok:
        st.session_state.input = random.choice(ham_pool)
        st.session_state.force_analyze = True
        st.rerun()
   
   

   # 1. קודם כל שואבים את הטקסט מתוך ה-session_state או מהווידג'ט
    text = st.session_state.input.strip() if "input" in st.session_state else ""

    # 2. עכשיו אפשר לבדוק אם הוא ריק
    if analyze or force_analyze:
        if not text: # עכשיו 'text' כבר קיים ולא תהיה שגיאה
            st.warning("נא להזין טקסט לפני הניתוח.")
            return
        left, right = st.columns(2)

        with left:
            st.subheader("🛡️ זיהוי ספאם")
            st.caption("Spam vs. Ham (legitimate)")
            spam_clf = cast(Pipeline, models["spam"])
            s_pred = int(spam_clf.predict([text])[0])
            s_proba = spam_clf.predict_proba([text])[0]
            classes = list(spam_clf.classes_)
            order_map = {c: i for i, c in enumerate(classes)}
            probs_ordered = np.array([s_proba[order_map[c]] for c in sorted(classes)])
            label_names = ["תקין (Ham)", "ספאם (Spam)"]
            verdict = "⚠️ זוהה כספאם" if s_pred == 1 else "✅ נראה כהודעה תקינה"
            st.markdown(f"### {verdict}")
            render_proba_bars(label_names, probs_ordered, "רמת ביטחון / Confidence")

        with right:
            st.subheader("💬 ניתוח סנטימנט")
            st.caption("Negative / Neutral / Positive")
            sent_clf = cast(Pipeline, models["sentiment"])
            t_pred = str(sent_clf.predict([text])[0])
            t_proba = sent_clf.predict_proba([text])[0]
            t_classes = list(sent_clf.classes_)
            he_map = {
                "negative": "שלילי (Negative)",
                "neutral": "נייטרלי (Neutral)",
                "positive": "חיובי (Positive)",
            }
            label_names_r = [he_map[c] for c in t_classes]
            st.markdown(f"### תחזית: **{he_map.get(t_pred, t_pred)}**")
            render_proba_bars(label_names_r, t_proba, "התפלגות הסתברות / Probability mix")

        st.divider()
        st.markdown(
            "**הערות מקצועיות:** המודלים אומנו על מאגר זעיר לצורך הדגמה בלבד; "
            "בפרודקשן יש לאסוף נתונים אמיתיים, להעריך מדדים נוספים (F1, ROC), "
            "ולשקול מודלים חזקים יותר לעברית."
        )


if __name__ == "__main__":
    main()
