import json
import random
import copy
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

def load_data(file_path):
    emojis = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            emojis.append(entry["label"])
    return emojis

def count_emojis(emojis):
    emoji_counts = Counter(emojis)
    return emoji_counts

def print_emoji_counts(emoji_counts):
    for emoji, count in emoji_counts.items():
        print(f"{emoji}: {count}")

def load_data_train(file_path):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data.append(entry["text_no_emoji"])
            labels.append(entry["label"])  
    return data, labels

def load_data_test(file_path):
    data = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            entry = json.loads(line)
            data.append(entry["text_no_emoji"])
            labels.append(entry["ground_truth_label"])  
    return data, labels

def replace_labels(labels):
    label_mapping = {
        'beaming_face_with_smiling_eyes': 'faccina sorridente con occhi sorridenti',
        'smiling_face_with_sunglasses': 'faccina sorridente con occhiali da sole',
        'rose': 'rosa',
        'two_hearts': 'due cuori',
        'sparkles': 'scintille',
        'loudly_crying_face': 'faccina che piange rumorosamente',
        'face_screaming_in_fear': 'faccina che urla dalla paura',
        'smiling_face_with_smiling_eyes': 'faccina sorridente con occhi sorridenti',
        'face_blowing_a_kiss': 'faccina che manda un bacio',
        'rolling_on_the_floor_laughing': 'rotolare per le risate sul pavimento',
        'winking_face': 'faccina che fa l\'occhiolino',
        'grinning_face_with_sweat': 'faccina sorridente con sudore',
        'face_savoring_food': 'faccina che assapora il cibo',
        'thumbs_up': 'pollice su',
        'smiling_face_with_heart_eyes': 'faccina sorridente con occhi a cuore',
        'sun': 'sole',
        'blue_heart': 'cuore blu',
        'thinking_face': 'faccina pensante',
        'red_heart': 'cuore rosso',
        'grinning_face': 'faccina ghignante',
        'flexed_biceps': 'bicipite flesso',
        'TOP_arrow': 'freccia su',
        'kiss_mark': 'impronta del bacio',
        'face_with_tears_of_joy': 'faccina con lacrime di gioia',
        'winking_face_with_tongue': 'faccina che fa l\'occhiolino con la lingua'
    }
    return [label_mapping[label] for label in labels]

def write_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # Codice 1
    file_path = "22_itamoji_emojiprediction\ITAmoji_anonim\ITAmoji_2018_TRAINdataset_v1.ANON.list"
    emojis = load_data(file_path)
    emoji_counts = count_emojis(emojis)
    print_emoji_counts(emoji_counts)

    # Codice 2
    test_texts, test_labels = load_data_test("22_itamoji_emojiprediction\ITAmoji_anonim\ITAmoji_2018_TESTdataset_v1_withGroundTruth.ANON.list")
    train_texts, train_labels = load_data_train("22_itamoji_emojiprediction\ITAmoji_anonim\ITAmoji_2018_TRAINdataset_v1.ANON.list")
    test_labels = replace_labels(test_labels)
    train_labels = replace_labels(train_labels)
    X = train_texts + test_texts  
    Y = train_labels + test_labels
    classes = list(set(Y))
    label_to_number = {label: i for i, label in enumerate(classes)}
    y_train_numerical = [label_to_number[label] for label in Y]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_train_numerical, test_size=0.2, random_state=42)
    class_counts = Counter(y_train)
    max_samples = max(class_counts.values())
    target_counts = {label: max_samples for label in class_counts}
    oversampler = RandomOverSampler(sampling_strategy=target_counts)
    X_train, y_train = oversampler.fit_resample(X_train, y_train)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    #print("accuracy", accuracy, "f1_score", f1)
    data = []
    x = vectorizer.transform(test_texts)
    probs = classifier.predict_proba(x)
    for i in range(len(test_labels)):
        emoji = test_labels[i]
        dc = {}
        ch = [emoji]
        distractor_classes = copy.deepcopy(classes)
        while len(ch) < 4:
            choice = random.choices(distractor_classes, weights=probs[i], k=1)[0]
            if choice not in ch:
                ch.append(choice)
        random.shuffle(ch)
        dc['sentence'] = test_texts[i]
        dc['choices'] = ch
        dc['label'] = dc['choices'].index(emoji)
        data.append(dc)
    output_file = "itamoji_emojiprediction_test.jsonl"        
    write_jsonl(data, output_file)

    # Codice 3
    X = train_texts + test_texts  
    Y = train_labels + test_labels
    classes = list(set(Y))
    label_to_number = {label: i for i, label in enumerate(classes)}
    y_train_numerical = [label_to_number[label] for label in Y]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y_train_numerical, test_size=0.2, random_state=42)
    class_counts = Counter(y_train)
    max_samples = max(class_counts.values())
    target_counts = {label: max_samples for label in class_counts}
    oversampler = RandomOverSampler(sampling_strategy=target_counts)
    X_resampled, y_resampled = oversampler.fit_resample(X_train, y_train)
    undersampler = RandomUnderSampler(sampling_strategy=target_counts)
    X_train, y_train = undersampler.fit_resample(X_resampled, y_resampled)
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    #print("accuracy", accuracy, "f1_score", f1)
    data = []
    x = vectorizer.transform(train_texts)
    probs = classifier.predict_proba(x)
    for i in range(len(train_labels)):
        emoji = train_labels[i]
        dc = {}
        ch = [emoji]
        distractor_classes = copy.deepcopy(classes)
        while len(ch) < 4:
            choice = random.choices(distractor_classes, weights=probs[i], k=1)[0]
            if choice not in ch:
                ch.append(choice)
        random.shuffle(ch)
        dc['sentence'] = train_texts[i]
        dc['choices'] = ch
        dc['label'] = dc['choices'].index(emoji)
        data.append(dc)
    output_file = "itamoji_emojiprediction_train.jsonl"        
    write_jsonl(data, output_file)

if __name__ == "__main__":
    main()
