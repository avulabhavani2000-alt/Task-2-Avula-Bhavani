from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

print("=" * 60)
print("        AI DATA CLASSIFICATION USING MACHINE LEARNING")
print("=" * 60)

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

# Dataset Information
print("\nDATASET INFORMATION")
print("-" * 60)
print("Dataset Name      : Iris Dataset")
print("Total Samples     :", len(X))
print("Total Features    :", len(iris.feature_names))
print("Total Classes     :", len(iris.target_names))

print("\nFeature Names:")
for feature in iris.feature_names:
    print("•", feature)

print("\nTarget Classes:")
for index, flower in enumerate(iris.target_names):
    print(f"• Class {index}: {flower}")

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("DATA SPLITTING")
print("=" * 60)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# Create Classification Model
model = DecisionTreeClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# Make Predictions
predictions = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy Score : {accuracy * 100:.2f}%")

# Show Sample Predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

for i in range(10):
    actual_flower = iris.target_names[y_test[i]]
    predicted_flower = iris.target_names[predictions[i]]

    print(f"\nSample {i + 1}")
    print(f"Actual Flower    : {actual_flower}")
    print(f"Predicted Flower : {predicted_flower}")

# Classification Report
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions,
        target_names=iris.target_names
    )
)

# User Testing Section
print("\n" + "=" * 60)
print("CUSTOM FLOWER PREDICTION")
print("=" * 60)

print("\nEnter Flower Measurements:")

sepal_length = float(input("Sepal Length (cm): "))
sepal_width = float(input("Sepal Width (cm): "))
petal_length = float(input("Petal Length (cm): "))
petal_width = float(input("Petal Width (cm): "))

user_data = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = model.predict(user_data)

flower_name = iris.target_names[prediction[0]]

print("\nPredicted Flower Type :", flower_name)

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)