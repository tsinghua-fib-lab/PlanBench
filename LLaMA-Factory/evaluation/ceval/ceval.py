# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import datasets
import pandas as pd


_CITATION = """\
@article{huang2023ceval,
  title={C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models},
  author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and Fu, Yao and Sun, Maosong and He, Junxian},
  journal={arXiv preprint arXiv:2305.08322},
  year={2023}
}
"""

_DESCRIPTION = """\
C-Eval is a comprehensive Chinese evaluation suite for foundation models. It consists of 13948 multi-choice questions spanning 52 diverse disciplines and four difficulty levels.
"""

_HOMEPAGE = "https://cevalbenchmark.com"

_LICENSE = "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"

_URL = "ceval.zip"

task_list = [
    "urban_and_rural_planner_basic",
    "urban_and_rural_planner_knowledge",
    "urban_and_rural_planner_regulation",
]


class CevalConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class Ceval(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        CevalConfig(
            name=task_name,
        )
        for task_name in task_list
    ]

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("int32"),
                "question": datasets.Value("string"),
                "A": datasets.Value("string"),
                "B": datasets.Value("string"),
                "C": datasets.Value("string"),
                "D": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "explanation": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URL)
        task_name = self.config.name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "test", f"{task_name}_test.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "val", f"{task_name}_val.csv"),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, "dev", f"{task_name}_dev.csv"),
                },
            ),
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath, encoding="utf-8")
        for i, instance in enumerate(df.to_dict(orient="records")):
            if "answer" not in instance.keys():
                instance["answer"] = ""
            if "explanation" not in instance.keys():
                instance["explanation"] = ""
            yield i, instance

