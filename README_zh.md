## 目录

- [文本处理与问答对生成](#文本处理与问答对生成)
- [模型微调](模型微调)
- [模型测试](#模型测试)
- [实验结果](#实验结果)
- [Licenses](#licenses)
- [引用](#引用)



## 文本处理与问答对生成

本文构造的问答对分别来自历年真题、历年真题扩展和教材，涉及城市规划原理、城市规划管理及法规、城市规划实务、城市规划相关知识四部分知识内容。

| Name                                        |  Words  | Samples |
| ------------------------------------------- | :-----: | :-----: |
| MCQ                                         | 619,810 |  2,397  |
| dialog                                      | 350,080 |  4,139  |
| Principles of urban planning                | 470,621 |  5,091  |
| Knowledge of urban planning                 | 457,236 |  9,307  |
| Urban planning management and regulations   | 313,246 |  4,347  |
| Urban planning practice                     | 120,155 |  3,589  |
| Detailed regulatory plan                    | 174,626 |   246   |
| History of urban construction in China      | 156,923 |   608   |
| Additional contents of urban planning exams | 60,371  |  1,610  |

#### 历年真题

切换到程序主目录`QAGenSuite`，执行`urban_exam_process.py`程序,读取`ExamMaterial`文件夹中`xlsx`文件进行文本处理,构造`Json`格式问答对存储到`urban_exam_sft.json`,共计`2397`个`samples`

```
cd QAGenSuite
python urban_exam_process.py
```

```
# core code

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
```

```
# QA samples

[
  {
    "instruction": "以下是中国关于注册城乡规划师相关知识考试的单项选择题, 请选出其中的正确答案并给出解析。\n\n下列关于中国古代木构建筑的表述，哪项是错误的( )\n\nA. 木构架体系包括抬梁式、穿斗式、井干式三种形式\n\nB. 木构架体系中承重的梁柱结构部分称为大木作\n\nC. 斗棋由矩形的斗和升、方形的拱、斜的昂组成\n\nD. 清代用“斗口”作为建筑的模数",
    "output": "C\n\n斗拱是我国木构架建筑特有的结构构件，由方形的斗、升和矩形的棋、斜的昂组成。\n"
  },
  {
    "instruction": "以下是中国关于注册城乡规划师相关知识考试的单项选择题, 请选出其中的正确答案并给出解析。\n\n下列关于中国古建空间度量单位的表述，哪项是错误的( )\n\nA. 平面布置以“间”和“步”为单位\n\nB. 正面两柱间的水平距离称为“开间”\n\nC. 屋架上的檩与檩中心线间的水平距离，称为“步”\n\nD. 各开间宽度的总和称为“通进深”",
    "output": "D\n\n各步距离的总和或侧面各开间宽度的总和称为“通进深”。\n"
  }
 ]
```

#### 历年真题扩展

过滤历年真题中(1) 下列说法正确的 (2)下列说法错误的 (3)下列说法不准确的等类似题目,针对每个选项文本构造新的问答对,避免知识遗忘

切换到程序主目录`QAGenSuite`，执行`urban_exam_filter.py`程序,读取`urban_exam_sft.json`过滤含有关键字的题目,然后执行`urban_exam_extension.py`对过滤后的问答对进行扩充,得到扩充样本集`urban_exam_extra.json`,共计样本`4139`个

```
cd QAGenSuite
python urban_exam_filter.py
python urban_exam_extension.py
```

```
# QA samples
[
    {
        "instruction": "在行政立法的效力等级不等式中，为什么法律大于行政法规？",
        "output": "行政立法的效力等级指行政法规在国家法律规范体系中所处的地位。宪法具有最高的法律效力；法律效力仅次于宪法，高于行政法规和规章；"
    },
    {
        "instruction": "在行政立法的效力等级不等式中，行政法规与地方性法规和规章相比，谁的效力更高？",
        "output": "行政法规大于地方性法规和规章"
    }
]
```

#### 教材

选择注册城市规划师考试官方四本教材、修订内容、《控制性详细规划》、《中国城市建设史》七本教材构造知识库,扩充问答对共计`24798`个

1. pdf 教材文本获取与文本处理

   执行`urban_pdf_to_txt.py`调用`pdfplumber`接口识别`pdf`文件夹下`pdf`格式教材内容,存储为`txt`格式,执行`urban_textbook_process.py`读取`txt`文件夹下全部`txt`文件,使用自定义正则表达式规则分割文本为段落,如果段落过长,按照序号 `1. 2. 3.` 继续深入分割,清理文本数据：去除多余空格回车及广告信息,将分割后的段落保存为`csv`格式文件并存储到`csv`文件夹

   ```
   cd QAGenSuite
   python urban_pdf_to_txt.py
   python urban_textbook_process.py
   ```

   ```
   # core code
   
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
               further_split_paragraphs.extend(split_paragraph_further(par))
           return further_split_paragraphs
   ```

2. 问答对生成

   方法一: 调用`Bonito `条件任务生成的开源模型

   克隆`repo`并安装依赖项

   ```
   !git clone https://github.com/BatsResearch/bonito.git
   
   !pip install -U bonito/
   ```

   使用以下命令创建环境并安装软件包：

   ```
   conda create -n bonito python=3.9
   conda activate bonito
   pip install -e .
   ```

   启动程序,读取`csv`文件中分割好的段落输入模型,得到输出问答对

   ```
   python urban_textbook_bonito.py
   ```

   ```
   # core code
   
   from bonito import Bonito
   from vllm import SamplingParams
   from datasets import load_dataset
   
   # Initialize the Bonito model
   
   bonito = Bonito("BatsResearch/bonito-v1")
   
   # load dataset with unannotated text
   
   unannotated_text = load_dataset(
       "BatsResearch/bonito-experiment",
       "unannotated_contract_nli"
   )["train"].select(range(10))
   
   # Generate synthetic instruction tuning dataset
   
   sampling_params = SamplingParams(max_tokens=256, top_p=0.95, temperature=0.5, n=1)
   synthetic_dataset = bonito.generate_tasks(
       unannotated_text,
       context_col="input",
       task_type="nli",
       sampling_params=sampling_params
   )
   ```

   方法二:调用`chatgpt api`接口,本文使用` HTTP requests` 请求的方式调用 `ChatGPT`

   启动程序,读取`csv`文件中分割好的段落输入模型,得到输出问答对

   ```
   python urban_textbook_gpt.py
   ```

   ```
   # core code
   
   # Construct request body
   
   url = "https://apikeyplus.com/v1/chat/completions"
   
   headers = {
     'Content-Type': 'application/json',
     'Authorization': 'Bearer sk-VpMF4Tvbz3B3zTPlF6343e78E2Fb401e866bC5E3200471D5'
   }
   
   # Preset prompt
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
   格式生成40个问答对
   '''
   
   file_counter = 1
   response_list = []
   
   # Traverse all CSV paragraphs, read the text content, concatenate it with the preset prompt to obtain the request body content, call the GPT API to send the request, obtain the response message, read the content field, and process it into JSON format to return a question and answer pair
   
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
   ```

   方法三:调用文心一言 `api`接口

   启动程序,读取`csv`文件中分割好的段落输入模型,得到输出问答对

   ```
   python urban_textbook_ernie.py
   ```

   核心代码与方法二类似,得到的问答对格式如下

   ```
   [
       {
           "instruction": "什么是法、法律与法律规范的关系？",
           "output": "法是一种社会规范，而法律是通过国家权力制定并强制执行的一种法律规范，是一种人们行为的准则，具有强制性和普遍性。"
       },
       {
           "instruction": "为什么法律规范具有普遍性？",
           "output": "法律规范具有普遍性是因为法律适用于所有的公民，不受其个人意志的干扰，是国家权力的具体表现。"
       }
   ]
   ```

## 模型微调

本文采用`LLaMA Factory `进行`lora`微调

#### 软硬件依赖

| 必需项       | 至少   | 推荐   |
| ------------ | ------ | ------ |
| python       | 3.8    | 3.10   |
| torch        | 1.13.1 | 2.2.0  |
| transformers | 4.37.2 | 4.40.1 |
| datasets     | 2.14.3 | 2.19.1 |
| accelerate   | 0.27.2 | 0.30.0 |
| peft         | 0.9.0  | 0.10.0 |
| trl          | 0.8.1  | 0.8.6  |

| 可选项       | 至少   | 推荐   |
| ------------ | ------ | ------ |
| CUDA         | 11.6   | 12.2   |
| deepspeed    | 0.10.0 | 0.14.0 |
| bitsandbytes | 0.39.0 | 0.43.1 |
| vllm         | 0.4.0  | 0.4.2  |
| flash-attn   | 2.3.0  | 2.5.8  |

#### 硬件依赖

| 方法              | 精度 | 7B    | 13B   | 30B   | 70B    | 110B   | 8x7B  | 8x22B  |
| ----------------- | ---- | ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| Full              | AMP  | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| Full              | 16   | 60GB  | 120GB | 300GB | 600GB  | 900GB  | 400GB | 1200GB |
| Freeze            | 16   | 20GB  | 40GB  | 80GB  | 200GB  | 360GB  | 160GB | 400GB  |
| LoRA/GaLore/BAdam | 16   | 16GB  | 32GB  | 64GB  | 160GB  | 240GB  | 120GB | 320GB  |
| QLoRA             | 8    | 10GB  | 20GB  | 40GB  | 80GB   | 140GB  | 60GB  | 160GB  |
| QLoRA             | 4    | 6GB   | 12GB  | 24GB  | 48GB   | 72GB   | 30GB  | 96GB   |
| QLoRA             | 2    | 4GB   | 8GB   | 16GB  | 24GB   | 48GB   | 18GB  | 48GB   |

#### 如何使用

##### 拉取仓库,下载依赖

```bash
git clone https://github.com/tsinghua-fib-lab/PlanBench/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]
```

##### 数据准备

将生成的`Json`格式问答对复制到`LLaMA-Factory/data`目录下,同时更新 `data/dataset_info.json` 文件,内容如下:

```
{
  # Q&A with only real questions
  "urban_exam_sft_zh": {
    "file_name": "urban_exam_sft.json",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  },
 # Textbook real questions, real questions, expanded question
  "urban_all_data": {
    "file_name": "urban_all_data.json",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  },
  # Q&A with only textbook content
  "urban_textbook": {
    "file_name": "urban_textbook.json",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  },
   # Q&A with only textbook content and original questions
  "urban_textbook_exam_sft": {
    "file_name": "urban_textbook_exam_sft.json",
    "columns": {
      "prompt": "instruction",
      "response": "output"
    }
  }
}
```

##### 使用 LLaMA Board 可视化界面（由 [Gradio](https://github.com/gradio-app/gradio) 驱动）

> LLaMA Board 可视化界面目前仅支持单 GPU 训练。

如果在 `Hugging Face` 模型和数据集的下载中遇到问题，通过下述方法使用魔搭社区

```
export USE_MODELSCOPE_HUB=1
```

在本地使用0号`GPU `在`7864`端口启动`LLaMA Board `可视化界面

```
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1  GRADIO_SERVER_PORT=7864 llamafactory-cli webui
```

在LLaMA Board 可视化界面选择模型,数据集,调整参数,启动微调,微调完成后适配器保存到`LLaMA-Factory/saves/`文件夹

## 模型测试

#### 为适配器添加配置文件

切换到`LLaMA-Factory/examples/lora_single_gpu` 目录

```
cd LLaMA-Factory/examples/lora_single_gpu
```

为适配器添加配置文件(针对单选题和多选题分别添加),以`planglm3_base_lora_eval.yaml`和`planglm3_base_lora_eval_m.yaml`为例,`planglm3_base_lora_eval.yaml`为单选题测试配置文件,`planglm3_base_lora_eval_m.yaml`为多选题测试配置文件,格式分别如下:

```
vim planglm3_base_lora_eval.yaml

# model
model_name_or_path: ZhipuAI/chatglm3-6b-base
adapter_name_or_path: saves/ChatGLM3-6B-Base/lora/train_2024-06-02-00-01-11

# method
finetuning_type: lora

# dataset
task: ceval
split: test
template: fewshot
lang: zh
n_shot: 0

# output
save_dir: saves/urban/planglm3_base_epoch3

# eval
batch_size: 4
```

```
vim planglm3_base_lora_eval_m.yaml

# model
model_name_or_path: ZhipuAI/chatglm3-6b-base
adapter_name_or_path: saves/ChatGLM3-6B-Base/lora/train_2024-06-02-00-01-11

# method
finetuning_type: lora

# dataset
task: cevalm
split: test
template: fewshot
lang: zh-m
n_shot: 0

# output
save_dir: saves/urban/planglm3_base_m_epoch3

# eval
batch_size: 4
```

#### 构建推理测试器

```
cd /LLaMA-Factory/src/llmtuner/eval
vim evaluator.py

#core code
class Evaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None, multi_choice: bool = False) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args.template)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self._multi_choice = multi_choice
        if not multi_choice:
            self._output_choices = CHOICES
        else:
            self._output_choices = MULTI_CHOICES
        self.choice_inputs = [
            self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in self._output_choices
        ]

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [self._output_choices[offset.item()] for offset in torch.argmax(choice_probs, dim=-1)]

    def eval(self) -> None:
        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, self.eval_args.task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
                kwargs = {"trust_remote_code": True}
            else:
                kwargs = {}

            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                **kwargs,
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[self.data_args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )

                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])

            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input)
                outputs += preds

            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)

    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join(
            [
                "{:>15}: {:.2f}% {:d}/{:d}".format(category_name, 100 * np.mean(category_correct), np.sum(category_correct), category_correct.size)
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=2)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)


def run_eval() -> None:
    Evaluator().eval()


def run_eval_multi() -> None:
    Evaluator(multi_choice=True).eval()
```

#### 重写cli.py函数,添加eval命令参数

```
cd LLaMA-Factory/src/llmtuner
vim cli.py

#core code
USAGE = """
Usage:
    llamafactory-cli api -h: launch an API server
    llamafactory-cli chat -h: launch a chat interface in CLI
    llamafactory-cli eval -h: do evaluation
    llamafactory-cli export -h: merge LoRA adapters and export model
    llamafactory-cli train -h: do training
    llamafactory-cli webchat -h: launch a chat interface in Web UI
    llamafactory-cli webui: launch LlamaBoard
    llamafactory-cli version: show version info
"""


@unique
class Command(str, Enum):
    API = "api"
    CHAT = "chat"
    EVAL = "eval"
    EVAL_M = "eval_m"
    EXPORT = "export"
    TRAIN = "train"
    WEBDEMO = "webchat"
    WEBUI = "webui"
    VERSION = "version"
    HELP = "help"


def main():
    command = sys.argv.pop(1)
    if command == Command.API:
        run_api()
    elif command == Command.CHAT:
        run_chat()
    elif command == Command.EVAL:
        run_eval()
    elif command == Command.EVAL_M:
        run_eval_multi()
    elif command == Command.EXPORT:
        export_model()
    elif command == Command.TRAIN:
        run_exp()
    elif command == Command.WEBDEMO:
        run_web_demo()
    elif command == Command.WEBUI:
        run_web_ui()
    elif command == Command.VERSION:
        print("Welcome to LLaMA Factory, version {}".format(__version__))
    elif command == Command.HELP:
        print(USAGE)
    else:
        raise NotImplementedError("Unknown command: {}".format(command))
```

#### 执行命令,进行lora推理测试

下面命令分别针对单选题和多选题对`ChatGLM3-6B-Base `模型进行` LoRA` 推理测试。

```bash
cd LLaMA-Factory
#将路径修改为第一步中配置文件路径
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval examples/lora_single_gpu/planglm3_base_lora_eval.yaml
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval_m examples/lora_single_gpu/planglm3_base_lora_eval_m.yaml
```

测试结果会保存到配置文件中指定的保存路径中

## 实验结果

#### **不同模型对比**

选取`ChatGLM3-6B-Base,ChatGLM3-6B-Chat,Baichuan2-7B-Base,Baichuan2-7B-Chat,LLaMA3-8B,LLaMA3-8B-chat,Qwen1.5-7B,Qwen1.5-7B-chat,Yi-6B,Yi-6B-chat`十种模型,分别通过构造的问答对进行微调,然后在测试集上分别针对单选题和多选题评估其在城市规划原理、城市规划管理及法规、城市规划相关知识三方面的理解能力

|                   | single-Average | single-Basic | single-Knowledge | single-Regulation | mul-Average | mul--Basic | mul--Knowledge | mul--Regulation |
| ----------------- | -------------- | ------------ | ---------------- | ----------------- | ----------- | ---------- | -------------- | --------------- |
| ChatGLM3-6B-Base  | 54.58%         | 50.00%       | 58.75%           | 55.00%            | 15.00%      | 5.00%      | 35.00%         | 5.00%           |
| ChatGLM3-6B-Chat  | 43.33%         | 35.00%       | 52.50%           | 42.50%            | 11.67%      | 5.00%      | 25.00%         | 5.00%           |
| Baichuan2-7B-Base | 48.75%         | 46.25%       | 56.25%           | 43.75%            | 8.33%       | 5.00%      | 5.00%          | 15.00%          |
| Baichuan2-7B-Chat | 40.42%         | 43.75%       | 46.25%           | 31.25%            | 8.33%       | 5.00%      | 20.00%         | 0.00%           |
| LLaMA3-8B         | 46.25%         | 40.00%       | 57.50%           | 41.25%            | 10.00%      | 5.00%      | 25.00%         | 0.00%           |
| LLaMA3-8B-chat    | 42.50%         | 42.50%       | 51.25%           | 33.75%            | 8.33%       | 10.00%     | 10.00%         | 5.00%           |
| Qwen1.5-7B        | 55.83%         | 50.00%       | 63.75%           | 53.75%            | 8.33%       | 10.00%     | 5.00%          | 10.00%          |
| Qwen1.5-7B-chat   | 54.17%         | 48.75%       | 63.75%           | 50.00%            | 10.00%      | 15.00%     | 5.00%          | 10.00%          |
| Yi-6B             | 62.92%         | 58.75%       | 68.75%           | 61.25%            | 6.67%       | 15.00%     | 0.00%          | 0.00%           |
| Yi-6B-chat        | 64.17%         | 62.50%       | 66.25%           | 63.75%            | 13.33%      | 0.00%      | 35.00%         | 5.00%           |

#### **不同数据对比**

选取实验一中性能最好的四种模型,分别通过构造不同规模和不同知识覆盖的问答对进行微调,然后在测试集上评估其表现,评估指标为城市规划原理、城市规划管理及法规、城市规划相关知识三部分知识在单选题和多选题准确率平均值

|                 | 原始模型 | 七本教材 | 七本教材+原来的真题 | 七本教材+原来真题+真题扩充 |
| --------------- | -------- | -------- | ------------------- | -------------------------- |
| Qwen1.5-7B      | 47.00%   | 46.33%   | 44.67%              | 46.33%                     |
| Qwen1.5-7B-chat | 45.00%   | 47.67%   | 47.00%              | 45.33%                     |
| Yi-6B           | 51.67%   | 51.67%   | 51.67%              | 51.67%                     |
| Yi-6B-chat      | 52.67%   | 50.00%   | 53.33%              | 54.00%                     |

#### **基础模型试验**

选取`ChatGLM3-6B-Base,ChatGLM3-6B-Chat,Baichuan2-7B-Base,Baichuan2-7B-Chat,LLaMA3-8B,LLaMA3-8B-chat,Qwen1.5-7B,Qwen1.5-7B-chat,Yi-6B,Yi-6B-chat`十种模型,不进行微调直接进行测试

|                   | single-Average | single-Basic | single-Knowledge | single-Regulation | mul-Average | mul--Basic | mul--Knowledge | mul--Regulation |
| ----------------- | -------------- | ------------ | ---------------- | ----------------- | ----------- | ---------- | -------------- | --------------- |
| ChatGLM3-6B-Base  | 52.50%         | 47.50%       | 60.00%           | 50.00%            | 11.67%      | 5.00%      | 25.00%         | 5.00%           |
| ChatGLM3-6B-Chat  | 43.75%         | 38.75%       | 51.25%           | 41.25%            | 13.33%      | 5.00%      | 30.00%         | 5.00%           |
| Baichuan2-7B-Base | 45.42%         | 50.00%       | 47.50%           | 38.75%            | 6.67%       | 5.00%      | 0.00%          | 15.00%          |
| Baichuan2-7B-Chat | 42.50%         | 36.25%       | 51.25%           | 40.00%            | 10.00%      | 5.00%      | 25.00%         | 0.00%           |
| LLaMA3-8B         | 44.58%         | 42.50%       | 53.75%           | 37.50%            | 10.00%      | 10.00%     | 20.00%         | 0.00%           |
| LLaMA3-8B-chat    | 47.92%         | 46.25%       | 58.75%           | 38.75%            | 13.33%      | 15.00%     | 25.00%         | 0.00%           |
| Qwen1.5-7B        | 55.83%         | 53.75%       | 60.00%           | 53.75%            | 11.67%      | 15.00%     | 10.00%         | 10.00%          |
| Qwen1.5-7B-chat   | 53.33%         | 47.50%       | 63.75%           | 48.75%            | 11.67%      | 15.00%     | 15.00%         | 5.00%           |
| Yi-6B             | 62.08%         | 61.25%       | 65.00%           | 60.00%            | 10.00%      | 15.00%     | 5.00%          | 10.00%          |
| Yi-6B-chat        | 62.92%         | 62.50%       | 70.00%           | 56.25%            | 11.67%      | 0.00%      | 30.00%         | 5.00%           |

#### **不同模型规模**

选取QWEN1.5模型的不同规模的版本,分别在不微调和sft后进行评估

 (一)不微调

|              | single-Average | single-Basic | single-Knowledge | single-Regulation | mul-Average | mul--Basic | mul--Knowledge | mul--Regulation |
| ------------ | -------------- | ------------ | ---------------- | ----------------- | ----------- | ---------- | -------------- | --------------- |
| Qwen1.5-0.5B | 33.75%         | 31.25%       | 36.25%           | 33.75%            | 6.67%       | 10.00%     | 10.00%         | 0.00%           |
| Qwen1.5-1.8B | 48.33%         | 38.75%       | 53.75%           | 52.50%            | 6.67%       | 10.00%     | 0.00%          | 10.00%          |
| Qwen1.5-4B   | 48.33%         | 43.75%       | 56.25%           | 45.00%            | 15.00%      | 15.00%     | 15.00%         | 15.00%          |
| Qwen1.5-7B   | 55.83%         | 53.75%       | 60.00%           | 53.75%            | 11.67%      | 15.00%     | 10.00%         | 10.00%          |
| Qwen1.5-14B  | 60.00%         | 48.75%       | 68.75%           | 62.50%            | 11.67%      | 5.00%      | 20.00%         | 10.00%          |
| Qwen1.5-32B  | 63.33%         | 65.00%       | 58.75%           | 66.25%            | 5.00%       | 5.00%      | 5.00%          | 5.00%           |

(二)SFT

|              | single-Average | single-Basic | single-Knowledge | single-Regulation | mul-Average | mul--Basic | mul--Knowledge | mul--Regulation |
| ------------ | -------------- | ------------ | ---------------- | ----------------- | ----------- | ---------- | -------------- | --------------- |
| Qwen1.5-0.5B | 37.50%         | 40.00%       | 40.00%           | 32.50%            | 6.67%       | 10.00%     | 10.00%         | 0.00%           |
| Qwen1.5-1.8B | 49.17%         | 38.75%       | 51.25%           | 57.50%            | 6.67%       | 10.00%     | 0.00%          | 10.00%          |
| Qwen1.5-4B   | 51.25%         | 48.75%       | 55.00%           | 50.00%            | 10.00%      | 10.00%     | 5.00%          | 15.00%          |
| Qwen1.5-7B   | 55.83%         | 50.00%       | 63.75%           | 53.75%            | 8.33%       | 10.00%     | 5.00%          | 10.00%          |
| Qwen1.5-14B  | 59.58%         | 51.25%       | 61.25%           | 66.25%            | 6.67%       | 5.00%      | 5.00%          | 10.00%          |
| Qwen1.5-32B  | 63.75%         | 60.00%       | 68.75%           | 62.50%            | 11.67%      | 10.00%     | 5.00%          | 20.00%          |

## Licenses

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)

本项目遵循 [MIT License](https://lbesson.mit-license.org/).



## 引用

