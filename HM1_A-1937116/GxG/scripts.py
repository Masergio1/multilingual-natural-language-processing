import re
import json

def process_train_file(input_file):
    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        docs = re.findall(r'<doc id="\d+" genre="[^"]+" gender="([MF])">(.*?)</doc>', content, re.DOTALL)
        for idx, (gender, text) in enumerate(docs, start=1):
            data.append({
                "id": idx,
                "text": text.strip(),
                "choices": ["female", "male"],
                "label": 1 if gender == "M" else 0
            })
    return data

def process_test_file(input_file, label_file, gold_code):
    labels = []
    with open(label_file, 'r') as label_file:
        for line in label_file:
            doc_id, gender = line.strip().split("\t")
            labels.append(gender)

    data = []
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
        docs = re.findall(r'<doc id="(\d+)" genre="[^"]+" gender="\?">(.*?)</doc>', content, re.DOTALL)
        i = 0
        for doc_id, text in docs:
            gender = labels[i]
            data.append({
                "id": int(doc_id),
                "text": text.strip(),
                "choices": ["femmina", "maschio"],
                "label": 1 if gender == "M" else 0
            })
            i += 1
    return data

def write_jsonl(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    tasks = ["Children", "Diary", "Journalism", "Twitter", "Youtube"]
    gold_codes = ["CH", "DI", "JO", "TW", "YT"]
    train_combined_data = []
    test_combined_data = []
    for task, gold_code in zip(tasks, gold_codes):
        train_input_file = "GxG\gxg-master\Data\Training\GxG_" + task + ".txt"
        train_data = process_train_file(train_input_file)
        train_combined_data.extend(train_data)

        test_input_file = "GxG\gxg-master\Data\Test\GxG_" + task + ".txt"
        test_label_file = "GxG\\gxg-master\\Data\\Gold\\test_" + gold_code + ".gold"
        test_data = process_test_file(test_input_file, test_label_file, gold_code)
        test_combined_data.extend(test_data)

    train_output_file = "GxG-task1-train-data.jsonl"
    test_output_file = "GxG-task1-test-data.jsonl"
    write_jsonl(train_combined_data, train_output_file)
    write_jsonl(test_combined_data, test_output_file)

if __name__ == "__main__":
    main()
