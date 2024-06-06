''' 读取csv中划分好的段落，调用open ai接口生成QA '''

import time
import requests
import os
import csv
import json
import re

folder_path = 'D:\\python\\PDFprocess\\csv\\'

url = "https://apikeyplus.com/v1/chat/completions"

headers = {
  'Content-Type': 'application/json',
  'Authorization': 'Bearer sk-VpMF4Tvbz3B3zTPlF6343e78E2Fb401e866bC5E3200471D5'
}

question_prompt = ''' #01 你是一个问答对数据集处理专家。
#02 你的任务是根据我的问题和我给出的内容，生成对应的问答对。
#03 生成的问题必须宏观、价值，不要生成特别细节的问题，不要太长。
#04 答案要全面，多使用我的信息，内容要更丰富。
#05 你必须根据我的问答对示例格式来生成：
  {
    "instruction": "问题",
    "output": "答案"
  }
按照
  {
    "instruction": "问题",
    "output": "答案"
  }
格式生成40个问答对，尽量是40个，每一个可以短一些，但最少够30个
'''

file_counter = 1
response_list = []

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            for row in reader:
                time.sleep(1)
                if row:
                    count += 1
                    input_question = "#6 我的内容如下" + row[0]
                    input_data = question_prompt + input_question

                    data = {
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": input_data}]
                    }

                    response = requests.post(url, headers=headers, json=data)
                    if response.status_code == 200:
                        response_content = response.json()['choices'][0]['message']['content']
                        processed_string = re.sub(r"\}\s+\{", "}, {", response_content.strip())
                        response_list.append(processed_string)

                    print(f"第'{count}'段写入完成")

                    # 每50行记录写入一个JSON文件并重置列表
                    if count % 50 == 0:
                        with open(f'{folder_path}{file_counter}.json', 'w', encoding='utf-8') as f:
                            json_string = json.dumps(response_list, indent=4, ensure_ascii=False)
                            f.write(json_string)
                        response_list = []
                        file_counter += 1
                        time.sleep(300)

# 写入最后剩余的记录到一个新的JSON文件
if response_list:
    with open(f'{folder_path}{file_counter}.json', 'w', encoding='utf-8') as f:
        json_string = json.dumps(response_list, indent=4, ensure_ascii=False)
        f.write(json_string)









