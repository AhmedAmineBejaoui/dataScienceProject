import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Show the first few rows of the training data

# Summary of the dataset

# Summary statistics of numerical columns


# Load the dataset relative to this script so it works on any machine.
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "titanic"
train_data = pd.read_csv(DATA_DIR / "train.csv")
test_data = pd.read_csv(DATA_DIR / "test.csv")

print(train_data.head())
print(train_data.info())
print(train_data.describe(include="all"))

# Missing values are imputed inside the sklearn pipeline below.
# Do not impute the full dataset before train/test splitting or cross-validation.

# Définir les features numériques et catégorielles
numeric_features = ['Age', 'Pclass', 'SibSp', 'Parch']
categorical_features = ['Sex', 'Embarked']

# Créer les transformations pour les colonnes numériques et catégorielles
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Appliquer les transformations à l'aide d'un ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Créer le pipeline incluant le préprocesseur et le modèle
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Séparer les features et la cible
X = train_data.drop(columns=['Survived', 'Name', 'Ticket', 'Cabin', 'PassengerId'])
y = train_data['Survived']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle avec le pipeline
model_pipeline.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model_pipeline.predict(X_test)

# Calculer et afficher l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Validation croisée pour évaluer la robustesse du modèle
cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=5)
print(f'Cross-Validation Accuracy: {cv_scores.mean()}')

# Prétraiter les données du test et faire des prédictions
X_test_data = test_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])
predictions = model_pipeline.predict(X_test_data)

# Créer un DataFrame pour la soumission
submission = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)
