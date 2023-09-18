import csv
import tldextract
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import pickle


def extract_character_pairs(domain):
    extracted = tldextract.extract(domain)
    second_level_domain = extracted.domain
    second_level_domain = '<' + second_level_domain + '>'
    character_pairs = [second_level_domain[i:i + 2] for i in range(len(second_level_domain) - 1)]
    return character_pairs


def create_feature_vector(character_pairs):
    # Tạo một danh sách của tất cả các ký tự có thể (bao gồm chữ cái, chữ số, và ký tự đặc biệt)
    characters = list("abcdefghijklmnopqrstuvwxyz0123456789-_<>")
    possible_pairs = [a + b for a in characters for b in characters]

    # Tạo một vector đặc trưng với tất cả các giá trị ban đầu là 0
    feature_vector = np.zeros(len(possible_pairs), dtype=int)

    # Đếm số lần xuất hiện của mỗi cặp ký tự trong character_pairs
    for pair in character_pairs:
        if pair in possible_pairs:
            index = possible_pairs.index(pair)
            feature_vector[index] += 1

    return feature_vector

legitimate_domains = []
with open('top1000.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        legitimate_domains.append((row[1], 0))
DGA_domains = []
with open('DGA_domains_test.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        DGA_domains.append((row[1], 1))

data = legitimate_domains + DGA_domains

# Chuyển đổi tên miền thành dạng vector và gán nhãn
X = np.array([create_feature_vector(extract_character_pairs(domain)) for domain, _ in data])
y = np.array([label for _, label in data])

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Huấn luyện một mạng nơ-ron
clf = MLPClassifier(random_state=42, max_iter=300)
clf.fit(X_train, y_train)

# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Đánh giá kết quả
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
