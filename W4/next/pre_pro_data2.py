import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def preprocess_and_save(json_path="W1/raw_cve_data.json", max_features=5000, test_size=0.2):
    """ Tiền xử lý dữ liệu và lưu vào file """
    with open(json_path, "r") as f:
        data = json.load(f)

    X, y = [], []
    for entry in data["vulnerabilities"]:
        cve = entry.get("cve", {})
        descriptions = cve.get("descriptions", [])
        desc_text = next((d["value"] for d in descriptions if d["lang"] == "en"), "")
        
        metrics = cve.get("metrics", {})
        severity_label = next((m["baseSeverity"] for m in metrics.get("cvssMetricV2", []) if "baseSeverity" in m), None)

        if desc_text and severity_label:
            X.append(desc_text)
            y.append(severity_label)

    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_tfidf = vectorizer.fit_transform(X)

    # Chia train-test
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=test_size, random_state=42, stratify=y)

    # Lưu dữ liệu vào file
    with open("processed_data.pkl", "wb") as f:
        pickle.dump((X_train, X_test, y_train, y_test), f)

    # Lưu vectorizer
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("Dữ liệu đã được xử lý và lưu lại!")

if __name__ == "__main__":
    preprocess_and_save()
