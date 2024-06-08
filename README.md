## Catalogue

- [Text processing and question answering pair generation](#Text-processing-and-question-answering-pair-generation)
- [Model fine-tuning](#Model-fine-tuning)
- [Model testing](#Model-testing)
- [Experimental result](#Experimental-result)
- [Licenses](#licenses)
- [quote](#quote)



## Text processing and question answering pair generation

The question and answer pairs constructed in this article come from past real questions, past real question extensions, and textbooks, covering four parts of knowledge content: urban planning principles, urban planning management and regulations, urban planning practice, and urban planning related knowledge.

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

#### Historical True Questions

Switch to the main directory of the program`QAGenSuite`, execute the `urban_exam_process. py` program, read the `xlsx``file in the `ExamMaterial` folder for text processing, construct a `Json`format Q&A pair, and store it in `urban_exam_sft. json`, totaling `2397`

```
cd QAGenSuite
python urban_exam_process.py
```

```
# core code

def generate_sft_data(df):
    data = []
    for i, row in df.iterrows():
        if row[`id`] < 80:
            question_type = `单项选择题`
            instruction = f"{row[`question`]}\n\nA. {row[`A`]}\n\nB. {row[`B`]}\n\nC. {row[`C`]}\n\nD. {row[`D`]}"
        else:
            question_type = `多项选择题`
            instruction = f"{row[`question`]}\n\nA. {row[`A`]}\n\nB. {row[`B`]}\n\nC. {row[`C`]}\n\nD. {row[`D`]}\n\nE. {row[`E`]}"
        system_prompt = f"以下是中国关于{row[`type`]}考试的{question_type}, 请选出其中的正确答案并给出解析。\n\n"
        response = f"{row[`answer`]}\n\n{row[`explanation`]}\n"
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

#### Extension of past exam questions

Filter past exam questions (1) if the following statements are correct (2) if the following statements are incorrect (3) if the following statements are inaccurate or similar, construct new question and answer pairs for each option text to avoid forgetting knowledge

Switch to the main directory of the program `QAGenSuite`, execute the `urban_exam filter. py` program, read the `urban_exam sft. json` to filter out questions containing keywords, and then execute the `urban_exam extension. py` to expand the filtered Q&A pairs, resulting in an expanded sample set of `urban_exam extra.json`, with a total of 4139 samples

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

#### Teaching material

Select four official textbooks for the Registered Urban Planner Examination, revise their content, and construct a knowledge base consisting of seven textbooks: "Controlled Detailed Planning" and "History of Urban Construction in China". Expand the Q&A pairs to a total of 24798

1. PDF textbook text acquisition and text processing

   Execute `urban_pdf_to_txt. py` to call the `pdfplumber` interface to recognize the `pdf` format textbook content in the `pdf` folder, store it in the `txt` format, execute `urban_textbook_process. py` to read all the `txt` files in the `txt` folder, use custom regular expression rules to split the text into segments, and if the segment is too long, follow the sequence number `1 2. 3. `Continue to delve deeper into segmentation, clean up text data: remove excess spaces, carriage returns, and advertising information, save the segmented paragraphs as a `csv` format file and store it in the `csv` folder

   ```
   cd QAGenSuite
   python urban_pdf_to_txt.py
   python urban_textbook_process.py
   ```

   ```
   # core code
   
   def clean_text(text):
       text = re.sub(r`<td.*?</td>`, ``, text, flags=re.DOTALL)
       text = re.sub(r`<table.*?</table>`, ``, text, flags=re.DOTALL)
       text = re.sub(r`<!--.*?-->`, ``, text, flags=re.DOTALL)
       text = re.sub(r`2000人注规交流群432529468。欢迎学习交流，仅限注规考试`, ``, text)
       text = re.sub(r`(\n\s*){2,}`, `\n`, text)
       return text
   
   
   def split_paragraphs(text, regex_pattern):
       text = clean_text(text)
       paragraphs = re.split(regex_pattern, text.strip())
       return [para.strip() for para in paragraphs if para]
   
   
   def split_paragraph_further(paragraph, max_length=5000):
       if len(paragraph) <= max_length:
           return [paragraph]
       sub_paragraphs = re.split(r`\s(?=\d+\.)`, paragraph)
       split_result = []
       current_part = sub_paragraphs[0]
   
       for part in sub_paragraphs[1:]:
           if len(current_part) + len(part) > max_length:
               split_result.append(current_part)
               current_part = part
           else:
               current_part += ` ` + part
       split_result.append(current_part)
   
       return split_result
   
   
   def process_file(file_path, regex_pattern):
       with open(file_path, `r`, encoding=`utf-8`) as file:
           content = file.read()
           paragraphs = split_paragraphs(content, regex_pattern)
   
           further_split_paragraphs = []
           for para in paragraphs:
               further_split_paragraphs.extend(split_paragraph_further(par))
           return further_split_paragraphs
   ```

2. Q&A generation

   Method 1: Call the open-source model generated by the `Bonito` conditional task

   Clone `repo` and install dependencies

   ```
   !git clone https://github.com/BatsResearch/bonito.git
   
   !pip install -U bonito/
   ```

   Create an environment and install software packages using the following command:

   ```
   conda create -n bonito python=3.9
   conda activate bonito
   pip install -e .
   ```

   Start the program, read the segmented paragraphs from the `csv` file and input them into the model to obtain the output question and answer pairs

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

   Method 2: Call the `chatgpt API` interface. This article uses the `HTTP requests` request method to call the` ChatGPT ``

   Start the program, read the segmented paragraphs from the `csv` file and input them into the model to obtain the output question and answer pairs

   ```
   python urban_textbook_gpt.py
   ```

   ```
   # core code
   
   # Construct request body
   
   url = "https://apikeyplus.com/v1/chat/completions"
   
   headers = {
     `Content-Type`: `application/json`,
     `Authorization`: `Bearer sk-VpMF4Tvbz3B3zTPlF6343e78E2Fb401e866bC5E3200471D5`
   }
   
   # Preset prompt
   question_prompt = ``` #01 你是一个问答对数据集处理专家。
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
   ​```
   
   file_counter = 1
   response_list = []
   
   # Traverse all CSV paragraphs, read the text content, concatenate it with the preset prompt to obtain the request body content, call the GPT API to send the request, obtain the response message, read the content field, and process it into JSON format to return a question and answer pair
   
   for filename in os.listdir(folder_path):
       if filename.endswith(`.csv`):
           file_path = os.path.join(folder_path, filename)
           with open(file_path, newline=``, encoding=`utf-8`) as csvfile:
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
                           response_content = response.json()[`choices`][0][`message`][`content`]
                           processed_string = re.sub(r"\}\s+\{", "}, {", response_content.strip())
                           response_list.append(processed_string)
   ```

   Method 3: Call the ERNIE Bot `api` interface

   Start the program, read the segmented paragraphs from the `csv` file and input them into the model to obtain the output question and answer pairs

   ```
   python urban_textbook_ernie.py
   ```

   The core code is similar to Method 2, and the format of the resulting question and answer pairs is as follows

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

## Model fine-tuning

This article uses` LLaMA Factory `for` lora `fine-tuning

#### Software dependencies

| Required     | at least | recommendation |
| ------------ | -------- | -------------- |
| python       | 3.8      | 3.10           |
| torch        | 1.13.1   | 2.2.0          |
| transformers | 4.37.2   | 4.40.1         |
| datasets     | 2.14.3   | 2.19.1         |
| accelerate   | 0.27.2   | 0.30.0         |
| peft         | 0.9.0    | 0.10.0         |
| trl          | 0.8.1    | 0.8.6          |

| Optional     | at least | recommendation |
| ------------ | -------- | -------------- |
| CUDA         | 11.6     | 12.2           |
| deepspeed    | 0.10.0   | 0.14.0         |
| bitsandbytes | 0.39.0   | 0.43.1         |
| vllm         | 0.4.0    | 0.4.2          |
| flash-attn   | 2.3.0    | 2.5.8          |

#### Hardware dependency

| method            | accuracy | 7B    | 13B   | 30B   | 70B    | 110B   | 8x7B  | 8x22B  |
| ----------------- | -------- | ----- | ----- | ----- | ------ | ------ | ----- | ------ |
| Full              | AMP      | 120GB | 240GB | 600GB | 1200GB | 2000GB | 900GB | 2400GB |
| Full              | 16       | 60GB  | 120GB | 300GB | 600GB  | 900GB  | 400GB | 1200GB |
| Freeze            | 16       | 20GB  | 40GB  | 80GB  | 200GB  | 360GB  | 160GB | 400GB  |
| LoRA/GaLore/BAdam | 16       | 16GB  | 32GB  | 64GB  | 160GB  | 240GB  | 120GB | 320GB  |
| QLoRA             | 8        | 10GB  | 20GB  | 40GB  | 80GB   | 140GB  | 60GB  | 160GB  |
| QLoRA             | 4        | 6GB   | 12GB  | 24GB  | 48GB   | 72GB   | 30GB  | 96GB   |
| QLoRA             | 2        | 4GB   | 8GB   | 16GB  | 24GB   | 48GB   | 18GB  | 48GB   |

#### How to use it

##### Pull the warehouse and download dependencies

```bash
git clone https://github.com/tsinghua-fib-lab/PlanBench/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[metrics]
```

##### Data preparation

Copy the generated `Json` format question and answer pairs to the `LLaMA Factory/data` directory, and update the `data/dataset_info.json` file as follows:

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

##### Using LLaMA Board Visualization Interface

> The LLaMA Board visualization interface currently only supports single GPU training.

If you encounter problems downloading the `Hugging Face` model and dataset, use the following methods to use the Magic Building Community

```
export USE_MODELSCOPE_HUB=1
```

Launch the LLaMA Board visualization interface on port 7864 using GPU number 0 locally

```
CUDA_VISIBLE_DEVICES=0 GRADIO_SHARE=1  GRADIO_SERVER_PORT=7864 llamafactory-cli webui
```

Select the model, dataset, adjust parameters, initiate fine-tuning in the LLaMA Board visualization interface, and save the adapter to the `LLaMA Factory/saves/` folder after fine-tuning is completed

## Model testing

#### Add a configuration file for the adapter

change to the dir `LLaMA-Factory/examples/lora_single_gpu` 

```
cd LLaMA-Factory/examples/lora_single_gpu
```

Add configuration files to the adapter (for both single choice and multiple choice questions). For example, use `plantlm3_base_lora_eval. yaml` and `plantlm3_base_lora_eval_m. yaml` as the configuration files for single choice and multiple choice questions. The formats are as follows:

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

#### Building an inference tester

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

#### Rewrite the cli.py function and add the eval command parameter

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

#### Execute commands and perform Lora inference testing

The following commands perform LoRA inference tests on the `ChatGLM3-6B Base` model for both single choice and multiple choice questions.

```bash
cd LLaMA-Factory
#Change the path to the configuration file path in the first step
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval examples/lora_single_gpu/planglm3_base_lora_eval.yaml
CUDA_VISIBLE_DEVICES=5 llamafactory-cli eval_m examples/lora_single_gpu/planglm3_base_lora_eval_m.yaml
```

The test results will be saved to the specified save path in the configuration file

## Experimental result 

#### Comparison of different models

Select `ChatGLM3-6B-Base,ChatGLM3-6B-Chat,Baichuan2-7B-Base,Baichuan2-7B-Chat,LLaMA3-8B,LLaMA3-8B-chat,Qwen1.5-7B,Qwen1.5-7B-chat,Yi-6B,Yi-6B-chat`, Fine tune the questions and answers constructed separately, and then evaluate their understanding ability in urban planning principles, urban planning management and regulations, and urban planning related knowledge on the test set for single choice and multiple choice questions

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

#### Comparison of different data

Select the four models with the best performance in Experiment 1, fine tune them by constructing question and answer pairs of different scales and knowledge coverage, and then evaluate their performance on the test set. The evaluation indicators are the average accuracy of urban planning principles, urban planning management and regulations, and urban planning related knowledge in multiple-choice and multiple-choice questions

|                 | 原始模型 | 七本教材 | 七本教材+原来的真题 | 七本教材+原来真题+真题扩充 |
| --------------- | -------- | -------- | ------------------- | -------------------------- |
| Qwen1.5-7B      | 47.00%   | 46.33%   | 44.67%              | 46.33%                     |
| Qwen1.5-7B-chat | 45.00%   | 47.67%   | 47.00%              | 45.33%                     |
| Yi-6B           | 51.67%   | 51.67%   | 51.67%              | 51.67%                     |
| Yi-6B-chat      | 52.67%   | 50.00%   | 53.33%              | 54.00%                     |

#### Basic model test

Select `ChatGLM3-6B-Base,ChatGLM3-6B-Chat,Baichuan2-7B-Base,Baichuan2-7B-Chat,LLaMA3-8B,LLaMA3-8B-chat,Qwen1.5-7B,Qwen1.5-7B-chat,Yi-6B,Yi-6B-chat`,Test directly without fine-tuning

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

#### **Different model scales**

Select versions of the `QWEN1.5 `model of different scales and evaluate them separately after no fine-tuning and SFT

 (一)No fine-tuning

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

This project follows the [MIT License]（ https://lbesson.mit-license.org/ ）



## quote

