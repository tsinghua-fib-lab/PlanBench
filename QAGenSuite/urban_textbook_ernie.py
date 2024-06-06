''' 读取csv中划分好的段落，调用文心一言接口生成QA '''
# -*- coding: utf-8 -*-
import csv
import json
import os
import time

import requests

API_KEY = "eyAYJjaGJeFL7rrZNTKXfQIQ"
SECRET_KEY = "PVJO2ASeb13A9UQoUhlaH5DlAoAksyD8"

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
格式生成10个问答对，每一个可以短一些
'''


def ask_Q(question):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()

    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response


def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


folder_path = 'D:\\python\\PDFprocess\\csv\\'
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
                    ans = ask_Q(input_data)
                    if "error_msg" in ans.json():
                        print(f"error occur第'{count}'段")
                    else:
                        response_list.append(ans.json()['result'])
                        print(f"文件{filename}的第'{count}'段写入完成")
                    # 每50行记录写入一个JSON文件并重置列表
                    if count % 100 == 0:
                        with open(f'{folder_path}{file_counter}.json', 'w', encoding='utf-8') as f:
                            f.write(str(response_list))
                        response_list = []  # 重置列表
                        file_counter += 1
                        time.sleep(100)

if response_list:
    with open(f'{folder_path}{file_counter}.json', 'w', encoding='utf-8') as f:
        f.write(str(response_list))
        print(f"全部写入完成")



