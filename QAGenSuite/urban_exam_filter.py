import json

def filter_questions(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    filtered_objects = []
    keywords = ["正确的", "错误的", "不准确的"]
    for obj in data:
        instruction = obj.get('instruction', '')
        if any(keyword in instruction for keyword in keywords):
            filtered_objects.append(obj)

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filtered_objects, file, ensure_ascii=False, indent=4)

input_filepath = 'urban_exam_sft.json'
output_filepath = 'urban_exam_sft_filtered.json'
filter_questions(input_filepath, output_filepath)
