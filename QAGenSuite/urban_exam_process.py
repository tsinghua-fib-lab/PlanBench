import json
import os

import pandas as pd


def transform(file):
    df = pd.read_excel(file)
    df = df[['题号', '问题', 'A', 'B', 'C', 'D', 'E', '答案', '解析']]
    df.columns = ['id', 'question', 'A', 'B', 'C', 'D', 'E', 'answer', 'explanation']
    df['id'] = df['id'] - 1
    if '原理' in file:
        df['type'] = '注册城乡规划师原理'
    elif '相关知识' in file:
        df['type'] = '注册城乡规划师相关知识'
    else:
        df['type'] = '注册城乡规划师管理与法规'
    return df


def generate_sft_data(df):
    data = []
    for i, row in df.iterrows():
        if row['id'] < 80:
            question_type = '单项选择题'
            instruction = f"{row['question']}\n\nA. {row['A']}\n\nB. {row['B']}\n\nC. {row['C']}\n\nD. {row['D']}"
        else:
            question_type = '多项选择题'
            instruction = f"{row['question']}\n\nA. {row['A']}\n\nB. {row['B']}\n\nC. {row['C']}\n\nD. {row['D']}\n\nE. {row['E']}"
        system_prompt = f"以下是中国关于{row['type']}考试的{question_type}, 请选出其中的正确答案并给出解析。\n\n"
        response = f"{row['answer']}\n\n{row['explanation']}\n"
        sample = {
            "instruction": system_prompt + instruction,
            "output": response,
        }
        data.append(sample)
    return data


if __name__ == '__main__':
    data_dir = 'ExamMaterial'
    # get all filenames in data_dir that are xlsx files
    files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    sft_data = []
    for file in files:
        df = transform('ExamMaterial'+'/'+file)
        sft_data.extend(generate_sft_data(df))
    # save the data to a json file
    with open('urban_exam_sft.json', 'w',encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
