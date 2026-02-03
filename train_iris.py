from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import argparse


def main(test_size=0.2, random_state=42):
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'Accuracy: {acc:.4f}')
    print(classification_report(y_test, preds, target_names=iris.target_names))
    cm = confusion_matrix(y_test, preds)
    df_cm = pd.DataFrame(cm, index=iris.target_names, columns=iris.target_names)
    sns.heatmap(df_cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Iris classifier')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state (default: 42)')
    args = parser.parse_args()
    main(test_size=args.test_size, random_state=args.random_state)
