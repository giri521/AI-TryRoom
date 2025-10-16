# train_fashion_model.py - Adapted for StyleSync AI Fashion Data
import json
import os
import warnings
import numpy as np
import pandas as pd
from pprint import pprint

# Embeddings
# NOTE: The default models are good for general text. For fashion, consider fine-tuning
# or using a model trained on e-commerce/product descriptions if possible.
from sentence_transformers import SentenceTransformer

# Models & utils
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import joblib

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Optional libs (kept from original template)
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# -------------------------
# Helper functions
# -------------------------
def choose_label_column(df):
    """
    Chooses the target classification column for the fashion dataset.
    Prioritizes 'gender', 'style', and 'category'.
    """
    preferred = ['gender', 'style', 'category', 'combo_type']
    for c in preferred:
        if c in df.columns:
            # Check this is usable as a classification label
            nunique = df[c].nunique(dropna=True)
            if nunique >= 2:
                # We prioritize 'gender' or 'style' as better classification targets
                return c
    return None

def create_pseudo_labels(embeddings, n_clusters=4):
    """Create pseudo-labels using KMeans clustering on embeddings (if no true labels available)."""
    from sklearn.cluster import KMeans
    n_clusters = min(n_clusters, max(2, embeddings.shape[0] // 5))  # avoid too many clusters for tiny data
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels.astype(str)

def safe_cv_strategy(y, max_splits=5):
    """Return an appropriate CV splitter (StratifiedKFold if possible, else KFold)."""
    counts = pd.Series(y).value_counts()
    min_count = counts.min()
    if min_count >= 2:
        n_splits = min(max_splits, int(min_count))
        if n_splits < 2:
            n_splits = 2
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        # fallback to KFold (not stratified) if there's a class with a single sample
        n_splits = 2
        return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

# -------------------------
# 1. Load dataset
# -------------------------
DATA_PATH = "data.json" # Your product data from the previous step
assert os.path.exists(DATA_PATH), f"{DATA_PATH} not found. Put your dataset as 'data.json' in cwd."

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data_list = json.load(f)
data = pd.DataFrame(data_list)
print("✅ Loaded dataset:", data.shape)
print("Columns:", list(data.columns))

# -------------------------
# 2. Preprocess text (Feature Engineering for Embeddings)
# -------------------------
def to_text_field(row):
    """
    Combines core product attributes into a single text string for embedding.
    """
    name = row.get('name', '') or ''
    category = row.get('category', '') or ''
    color = row.get('color', '') or ''
    material = row.get('material', '') or ''
    style = row.get('style', '') or ''
    fit = row.get('fit', '') or ''
    gender = row.get('gender', '') or ''
    
    # Concatenate attributes, forming a comprehensive product description
    text = f"{name}. Category: {category}. Color: {color}. Material: {material}. Style: {style}. Fit: {fit}. Gender: {gender}."
    return text.strip()

data['text'] = data.apply(to_text_field, axis=1)
print("Sample text (first row):\n", data['text'].iloc[0][:500])

# -------------------------
# 3. Compute / load embeddings
# -------------------------
EMB_PATH = "fashion_embeddings.npy" # Changed name to avoid conflict with recipe data

if os.path.exists(EMB_PATH):
    recipe_embeddings = np.load(EMB_PATH)
    print("✅ Loaded existing embeddings from", EMB_PATH)
else:
    # Use general-purpose models for product embeddings
    model_names = ['all-mpnet-base-v2', 'all-MiniLM-L6-v2']
    bert_model = None
    for mname in model_names:
        try:
            print("🔄 Loading SentenceTransformer:", mname)
            bert_model = SentenceTransformer(mname)
            print("✅ Using embedding model:", mname)
            break
        except Exception as e:
            print("⚠️ Failed to load", mname, ":", e)
            continue
    if bert_model is None:
        raise RuntimeError("No sentence-transformers model could be loaded. Install sentence-transformers and try again.")
        
    recipe_texts = data['text'].tolist()
    recipe_embeddings = bert_model.encode(recipe_texts, show_progress_bar=True, convert_to_numpy=True)
    np.save(EMB_PATH, recipe_embeddings)
    print("✅ Saved embeddings to", EMB_PATH)

# -------------------------
# 4. Determine label (or create pseudo-labels)
# -------------------------
# We will explicitly try to classify by 'style' as it's the most challenging/useful feature
label_col = 'style' # Target classification for StyleSync AI
y = None

if label_col in data.columns:
    # Since 'style' is multi-label (e.g., 'Casual Sporty'), we simplify it
    # by using the FIRST style tag found for multi-class classification training.
    data['simplified_style'] = data[label_col].apply(lambda x: str(x).split()[0] if str(x).strip() else 'Unknown')
    y = data['simplified_style'].astype(str).values
    
    # Check if the simplified column is suitable
    nunique = pd.Series(y).nunique()
    if nunique >= 2:
        print(f"✅ Using simplified label column: '{label_col}' (first tag) with {nunique} classes.")
    else:
        # Fallback if simplification failed (e.g., all rows are 'Unknown')
        y = None 

if y is None:
    # create pseudo-labels using KMeans
    print("ℹ️ No suitable label column found or 'style' is uninformative. Creating pseudo-labels with clustering (unsupervised)...")
    n_clusters = 4
    y = create_pseudo_labels(recipe_embeddings, n_clusters=n_clusters)
    print("✅ Created pseudo-labels. Number of pseudo-classes:", len(np.unique(y)))

# Print label distribution
print("Label distribution (Targeting the first style tag for prediction):")
print(pd.Series(y).value_counts())

# -------------------------
# 5. Dimensionality reduction (optional)
# For this small dataset, we skip reduction.
# -------------------------

# -------------------------
# 6. Define candidate models
# -------------------------
models = {}

# Logistic (needs scaling)
models['LogisticRegression'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=RANDOM_STATE))
])

# SVM
models['SVC'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE))
])

# Random Forest
models['RandomForest'] = RandomForestClassifier(
    n_estimators=250, max_depth=None, class_weight='balanced', random_state=RANDOM_STATE
)

# XGBoost (if available)
if HAS_XGB:
    models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=200, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE
    )

# LightGBM (if available)
if HAS_LGB:
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=200, random_state=RANDOM_STATE)

print("Models to evaluate:", list(models.keys()))

# -------------------------
# 7. Cross-validated evaluation
# -------------------------
# Use the safer CV strategy based on the (potentially simplified) target label 'y'
cv = safe_cv_strategy(y, max_splits=5) 
results = []

for name, estimator in models.items():
    try:
        # Accuracy is the standard metric for multi-class classification
        scores = cross_val_score(estimator, recipe_embeddings, y, cv=cv, scoring='accuracy', n_jobs=-1)
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        results.append((name, mean_score, std_score))
        print(f"{name:12s} | Acc (cv mean ± std): {mean_score:.4f} ± {std_score:.4f}")
    except Exception as e:
        print(f"⚠️ Skipping {name} due to error during cross_val_score: {e}")

# Sort and show
results = sorted(results, key=lambda x: x[1], reverse=True)
print("\n== Cross-validated accuracy ranking ==")
for name, mean_score, std_score in results:
    print(f"{name:12s} : {mean_score:.4f} ± {std_score:.4f}")

if len(results) == 0:
    raise RuntimeError("No models could be evaluated. Check data & installed libraries.")

best_name, best_mean, best_std = results[0]
best_estimator = models[best_name]
print(f"\n🎉 Best model by CV accuracy: {best_name} ({best_mean:.4f} ± {best_std:.4f})")

# -------------------------
# 8. Cross-validated predictions & final metrics
# -------------------------
print("\n🔁 Generating cross-validated predictions (out-of-fold) for final metrics...")
y_pred_oof = cross_val_predict(best_estimator, recipe_embeddings, y, cv=cv, n_jobs=-1)

acc = accuracy_score(y, y_pred_oof)
print(f"\n🎯 Cross-validated Accuracy (overall, out-of-fold): {acc:.4f}\n")

print("📊 Classification Report (out-of-fold):")
print(classification_report(y, y_pred_oof, zero_division=0))

# Confusion Matrix setup
unique_labels = np.unique(y)
cm = confusion_matrix(y, y_pred_oof, labels=unique_labels)
print("\n🧮 Confusion Matrix (rows=actual, cols=predicted):")
print(cm)

# Plot confusion matrix heatmap
plt.figure(figsize=(max(8, len(unique_labels)), max(6, len(unique_labels))))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=unique_labels, yticklabels=unique_labels, cmap="Blues")
plt.title(f"Confusion Matrix - {best_name} (cross-validated, OOF)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -------------------------
# 9. Optionally run a quick RandomizedSearch on the best model (safeguarded)
# -------------------------
DO_HYPERPARAM_TUNING = True
if DO_HYPERPARAM_TUNING:
    print("\n⚙️ Running a short RandomizedSearchCV on the best model (limited iterations)...")
    # Only tune for certain known models
    param_distributions = None
    estimator_for_search = None

    if best_name == 'RandomForest':
        estimator_for_search = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
        param_distributions = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20, 40],
            "min_samples_split": [2, 5, 10]
        }
    elif best_name == 'LogisticRegression':
        estimator_for_search = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000))
        ])
        param_distributions = {
            "clf__C": [0.01, 0.1, 1, 10, 100]
        }
    elif best_name == 'XGBoost' and HAS_XGB:
        estimator_for_search = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
        param_distributions = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    elif best_name == 'LightGBM' and HAS_LGB:
        estimator_for_search = lgb.LGBMClassifier(random_state=RANDOM_STATE)
        param_distributions = {
            "n_estimators": [100, 200, 400],
            "max_depth": [ -1, 10, 20],
            "learning_rate": [0.01, 0.05, 0.1]
        }
    else:
        print("No tuned hyperparam search configured for", best_name)

    if estimator_for_search is not None and param_distributions is not None:
        n_iter = 16
        try:
            rs = RandomizedSearchCV(estimator_for_search, param_distributions, n_iter=n_iter,
                                    scoring='accuracy', cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
            rs.fit(recipe_embeddings, y)
            print("Best params from RandomizedSearchCV:")
            pprint(rs.best_params_)
            print("Best cross-val accuracy:", rs.best_score_)
            best_estimator = rs.best_estimator_
        except Exception as e:
            print("⚠️ Hyperparam tuning failed or too expensive:", e)

# -------------------------
# 10. Fit best estimator on full data and save model
# -------------------------
print(f"\n🔐 Fitting best model ({best_name}) on the full dataset and saving to disk...")
try:
    best_estimator.fit(recipe_embeddings, y)
    joblib.dump(best_estimator, "best_fashion_model.joblib")
    print("✅ Saved best model to best_fashion_model.joblib")
except Exception as e:
    print("⚠️ Failed to fit & save best estimator:", e)

# Save embeddings & label mapping for later use
np.save(EMB_PATH, recipe_embeddings)
pd.Series(y, name='label').to_csv("fashion_labels.csv", index=False)
print("✅ Saved embeddings and label file.")

print("\n🎯 DONE. Summary:")
print(f" - Best model: {best_name}")
print(f" - Cross-validated OOF accuracy: {acc:.4f}")
print(" - Classification report printed above and confusion matrix plotted.")
print("\nNext steps for StyleSync AI:")
print(" - Integrate this model to predict the *best fit style* for a user based on their profile and a new product's description.")
print(" - Use the prediction probabilities to recommend items matching the predicted style.")