import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle

# Load dataset
# You can replace this dataset with your own
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/sms.tsv", sep='\t', header=None)
df.columns = ['label', 'message']

# Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Create pipeline (TF-IDF + Naive Bayes)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Save model and vectorizer
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
