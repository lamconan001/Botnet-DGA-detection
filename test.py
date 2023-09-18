import pickle
from ML import extract_character_pairs, create_feature_vector
from sklearn.metrics import classification_report
import numpy as np
import csv

with open('my_dumped_classifier.pkl', 'rb') as file:
    clf = pickle.load(file)

test = []
with open('DGA_domains.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        test.append((row[1], 1))

with open('legit_domains_test.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        test.append((row[1], 0))

X_test = np.array([create_feature_vector(extract_character_pairs(domain)) for domain, _ in test])
y_test = np.array([label for _, label in test])
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))