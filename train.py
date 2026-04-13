import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("Dataset1.csv")

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Separate features and labels
X = df.drop("diseases", axis=1)
y = df["diseases"]

print(f"Features: {X.shape[1]}")
print(f"Classes: {y.nunique()}")

# Label encoding (text → number)
le = LabelEncoder()
y = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")

# Save model and encoder
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("✅ Model trained and saved!")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# dataset load karo
df = pd.read_csv("Dataset1.csv")

print(df.columns)
# features aur label separate karo
X = df.drop("diseases", axis=1)
y = df["diseases"]

# label encoding (text → number)
le = LabelEncoder()
y = le.fit_transform(y)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model train karo
model = RandomForestClassifier()
model.fit(X_train, y_train)

# model save karo
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("✅ Model trained successfully!")