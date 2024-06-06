
'''
读取urban_exam_sft_filtered.json中过滤得到的需要扩充的真题题目
调用文心一言接口生成QA
'''
import json
import requests

API_KEY = "eyAYJjaGJeFL7rrZNTKXfQIQ"
SECRET_KEY = "PVJO2ASeb13A9UQoUhlaH5DlAoAksyD8"

question_prompt = ''' 你是一个专门做以下工作的专家：{
    "instruction": "以下是中国关于注册城乡规划师相关知识考试的单项选择题, 请选出其中的正确答案并给出解析。\n\n下列关于西方古代建筑风格特点的表述，哪项是错误的( )\n\nA. 古埃及建筑追求雄伟、庄严、神秘、震撼人心的艺术效果\n\nB. 古希腊建筑风格特征为庄严、典雅、精致、有性格、有活力\n\nC. 巴洛克建筑应用纤巧的装饰，具有娇媚柔糜的贵族气息\n\nD. 古典主义建筑立面造型强调轴线对称和比例关系",
    "output": "C\n\n巴洛克建筑的风格特征包括：(1)追求新奇；(2)追求建筑形体和空间的动态，常用穿插的曲面和椭圆形空间；(3)喜好富丽的装饰，强烈的色彩，打破建筑与雕刻绘画的界限，使其相互渗透；(4)趋向自然，追求自由奔放的格调，表达世俗情趣，具有欢乐气氛。\n"
  },
  {
    "instruction": "以下是中国关于注册城乡规划师相关知识考试的单项选择题, 请选出其中的正确答案并给出解析。\n\n下列关于近现代西方建筑流派创作特征和建筑主张的表述，哪项是错误的( )\n\nA. 工艺美术运动热衷于手工艺的效果与自然材料的美\n\nB. 新艺术运动热衷于模仿自然界草木形状的曲线\n\nC. 维也纳分离派主张结合传统和地域文化\n\nD. 德意志制造联盟主张建筑和工业相结合",
    "output": "C\n\n维也纳分离派声称要和过去的传统决裂。他们主张造型简洁与集中装饰，但和新艺术运动的不同是装饰主题用直线和大片墙面以及简单的立方体，使建筑走向简洁的道路。\n"
  }
以上每一个json对象的instruction属性都是一道选择题，output属性为对应的答案和解析。请你根据题目和答案将错误选项构造为一道新的题目，要求不偏离本题目的内容，尽量根据解析来设置，构造的题目也可以是问答题，输出格式和上述格式一致。我给出范式：比如可以将上述json第一个对象构造为：
{
    "instruction": "古埃及建筑追求雄伟、庄严、神秘、震撼人心的艺术效果，该说法是否正确",
    "output": "正确"
  }，
{
    "instruction": "古希腊建筑风格特征是什么",
    "output": "古希腊建筑风格特征为庄严、典雅、精致、有性格、有活力"
  }，
{
    "instruction": "古典主义建筑立面造型强调什么？",
    "output": "古典主义建筑立面造型强调轴线对称和比例关系"
  }
现在请你按照你的职责，为以下内容生成四个问题，只需要四个，不能偏离本题目内容
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
    （Access Token）
    :return: access_token，or None
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))

filename = 'urban_exam_sft_filtered.json'
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)
count = 0
file_counter = 1
response_list = []
for item in data:
    count += 1
    question = question_prompt + str(item)
    ans = ask_Q(question)
    response_list.append(ans.json()['result'])
    print(f"'{count}'writing end")

with open(f'urban_exam_extra.json', 'w', encoding='utf-8') as f:
    f.write(str(response_list))
    print(f" All writing end")

