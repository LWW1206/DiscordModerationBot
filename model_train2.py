import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

df1 = pd.read_csv("train.csv")
df1["toxic_label"] = df1["toxic"].apply(lambda x: 1 if x > 0 else 0)

df2 = pd.read_csv("train2.csv")
df2["toxic_label"] = df2["target"].apply(lambda x: 1 if x > 0.5 else 0)

df_combined = pd.concat([
    df1[["comment_text", "toxic_label"]],
    df2[["comment_text", "toxic_label"]]
], ignore_index=True)

X_train, X_test, y_train, y_test = train_test_split(
    df_combined["comment_text"], 
    df_combined["toxic_label"], 
    test_size=0.2, 
    random_state=42
)

vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

logreg = LogisticRegression()
logreg.fit(X_train_tfidf, y_train)

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)

# Save models
joblib.dump(vectorizer, "vectorizer2.pkl")
joblib.dump(logreg, "logistic_regression_model2.pkl")
joblib.dump(nb, "naive_bayes_model2.pkl")

print("✅ All models trained and saved successfully!")
print(f"Total training samples: {len(df_combined)}")
print(f"Toxic ratio: {df_combined['toxic_label'].mean():.2%}")