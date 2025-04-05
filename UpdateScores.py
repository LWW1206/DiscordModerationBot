import sqlite3
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

try:
    vectorizer = joblib.load("vectorizer2.pkl")
    logreg_model = joblib.load("logistic_regression_model2.pkl")
    naive_bayes_model = joblib.load("naive_bayes_model2.pkl")
    logging.info("✅ Successfully loaded ML models.")
except FileNotFoundError:
    logging.error("❌ Trained models not found. Make sure all .pkl files exist.")
    exit(1)

# Define best thresholds (from F1 score analysis)
thresholds = {
    "distilbert": 0.72,
    "roberta": 0.65,
    "logreg": 0.48,
    "naive_bayes": 0.45
}

db_path = "moderation_data_updated.db"
logging.info(f"Connecting to database: {db_path}")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT id, message, distilbert_score, roberta_score FROM moderation")
rows = cursor.fetchall()
logging.info(f"Processing {len(rows)} rows...")

def get_logreg_proba(message):
    vectorized_message = vectorizer.transform([message])
    return logreg_model.predict_proba(vectorized_message)[0][1]

def get_naive_bayes_proba(message):
    vectorized_message = vectorizer.transform([message])
    return naive_bayes_model.predict_proba(vectorized_message)[0][1]

for row in rows:
    msg_id, message, distilbert_score, roberta_score = row

    logreg_score = get_logreg_proba(message)
    naive_bayes_score = get_naive_bayes_proba(message)

    flagged_distilbert = 1 if distilbert_score > thresholds["distilbert"] else 0
    flagged_roberta = 1 if roberta_score > thresholds["roberta"] else 0
    flagged_logreg = 1 if logreg_score > thresholds["logreg"] else 0
    flagged_naive_bayes = 1 if naive_bayes_score > thresholds["naive_bayes"] else 0

    cursor.execute("""
    UPDATE moderation
    SET 
        flagged_distilbert = ?, 
        flagged_roberta = ?, 
        flagged_logreg = ?, 
        flagged_naive_bayes = ?
    WHERE id = ?
    """, (flagged_distilbert, flagged_roberta, flagged_logreg, flagged_naive_bayes, msg_id))

conn.commit()
conn.close()
logging.info("✅ Moderation flags updated based on optimal F1 thresholds.")
