'''
1、读取txt文件夹下全部txt文件
2、使用正则表达式分割文本为段落
3、如果段落过长，按照序号 1. 2. 3. 继续深入分割
4、清理文本数据：去除多余空格、回车及广告信息
5、将分割后的段落保存到csv文件存储到csv文件夹
'''
import os
import re
import csv


def clean_text(text):
    text = re.sub(r'<td.*?</td>', '', text, flags=re.DOTALL)
    text = re.sub(r'<table.*?</table>', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'2000人注规交流群432529468。欢迎学习交流，仅限注规考试', '', text)
    text = re.sub(r'(\n\s*){2,}', '\n', text)
    return text


def split_paragraphs(text, regex_pattern):
    text = clean_text(text)
    paragraphs = re.split(regex_pattern, text.strip())
    return [para.strip() for para in paragraphs if para]


def split_paragraph_further(paragraph, max_length=5000):
    if len(paragraph) <= max_length:
        return [paragraph]
    sub_paragraphs = re.split(r'\s(?=\d+\.)', paragraph)
    split_result = []
    current_part = sub_paragraphs[0]

    for part in sub_paragraphs[1:]:
        if len(current_part) + len(part) > max_length:
            split_result.append(current_part)
            current_part = part
        else:
            current_part += ' ' + part
    split_result.append(current_part)

    return split_result


def process_file(file_path, regex_pattern):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        paragraphs = split_paragraphs(content, regex_pattern)

        further_split_paragraphs = []
        for para in paragraphs:
            further_split_paragraphs.extend(split_paragraph_further(para))
        return further_split_paragraphs


def save_paragraphs_to_csv(paragraphs, csv_path):
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for paragraph in paragraphs:
            writer.writerow([paragraph])


def process_files_in_directory(directory, regex_pattern):
    csv_directory = os.path.join(directory, 'csv')
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            paragraphs = process_file(file_path, regex_pattern)

            csv_filename = f"{os.path.splitext(filename)[0]}.csv"
            csv_path = os.path.join(csv_directory, csv_filename)
            save_paragraphs_to_csv(paragraphs, csv_path)
            print(f"Processed '{filename}' and saved to '{csv_filename}'")


regex_pattern = r'\n(?=[一二三四五六七八九十]{1,}、)'
directory_path = 'D:\\python\\PDFprocess\\txt\\'
process_files_in_directory(directory_path, regex_pattern)
