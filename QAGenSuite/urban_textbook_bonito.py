'''
bonito模型调用，须在A100上运行
中文QA生产效果一般
'''

from pprint import pprint
import os
import csv
from datasets import Dataset
from vllm import SamplingParams
from transformers import set_seed

set_seed(2)


def convert_to_dataset(text):
    dataset = Dataset.from_list([{"input": text}])
    return dataset


folder_path = '/content/csv/'

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            pprint(f"文件：{filename}")
            count = 0
            for row in reader:
                if row:
                    count = count + 1
                    unannotated_paragraph = row[0]
                    sampling_params = SamplingParams(max_tokens=256, top_p=1, temperature=0.5, n=20)
                    synthetic_dataset = bonito.generate_tasks(
                        convert_to_dataset(unannotated_paragraph),
                        context_col="input",
                        task_type="qa",
                        sampling_params=sampling_params,
                    )
                    for i in range(0, len(synthetic_dataset)):
                        pprint("----无选择题问答----")
                        pprint(f'Input: {synthetic_dataset[i]["input"]}')
                        pprint(f'Output: {synthetic_dataset[i]["output"]}')

                    sampling_params = SamplingParams(max_tokens=256, top_p=1, temperature=0.05, n=20)
                    synthetic_dataset = bonito.generate_tasks(
                        convert_to_dataset(unannotated_paragraph),
                        context_col="input",
                        task_type="mcqa",
                        sampling_params=sampling_params,
                    )
                    for i in range(0, len(synthetic_dataset)):
                        pprint("----多项选择----")
                        pprint(f'Input: {synthetic_dataset[i]["input"]}')
                        pprint(f'Output: {synthetic_dataset[i]["output"]}')
                    sampling_params = SamplingParams(max_tokens=256, top_p=1, temperature=0.05, n=20)
                    synthetic_dataset = bonito.generate_tasks(
                        convert_to_dataset(unannotated_paragraph),
                        context_col="input",
                        task_type="exqa",
                        sampling_params=sampling_params,
                    )
                    for i in range(0, len(synthetic_dataset)):
                        pprint("----提取式问答----")
                        pprint(f'Input: {synthetic_dataset[i]["input"]}')
                        pprint(f'Output: {synthetic_dataset[i]["output"]}')
                    sampling_params = SamplingParams(max_tokens=256, top_p=1, temperature=0.05, n=20)
                    synthetic_dataset = bonito.generate_tasks(
                        convert_to_dataset(unannotated_paragraph),
                        context_col="input",
                        task_type="ynqa",
                        sampling_params=sampling_params,
                    )
                    for i in range(0, len(synthetic_dataset)):
                        pprint("----是非问答----")
                        pprint(f'Input: {synthetic_dataset[i]["input"]}')
                        pprint(f'Output: {synthetic_dataset[i]["output"]}')
                    pprint(f"文件：{filename}的第{count}段完成")
            pprint("")


