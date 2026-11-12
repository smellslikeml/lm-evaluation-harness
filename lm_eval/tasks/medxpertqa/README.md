# MedXpertQA

### Paper

Title: `MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding`

Abstract: [https://arxiv.org/abs/2501.18362](https://arxiv.org/abs/2501.18362)

MedXpertQA is an expert-level medical question-answering benchmark of 4,460 questions spanning
17 specialties and 11 body systems, built to evaluate expert-level medical knowledge and advanced
reasoning. Questions are sourced from real-world clinical and examination challenges and augmented
with authoritative medical resources. This task uses the **Text** subset (`Text` config) of the
dataset `TsinghuaC3I/MedXpertQA` (MIT license, publicly available on the Hugging Face Hub), which
contains 2,450 10-way multiple-choice questions (answer options A–J) in the `test` split, plus a
5-row `dev` split used as the few-shot source. Each question is additionally tagged with
`medical_task`, `body_system`, and `question_type` metadata.

Homepage: \
[https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA](https://huggingface.co/datasets/TsinghuaC3I/MedXpertQA)


#### Tasks

* `medxpertqa`: 10-way multiple-choice QA on the Text subset.

### Citation

```bibtex
@misc{zuo2025medxpertqa,
      title={MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding},
      author={Yuxin Zuo and Shang Qu and Yifei Li and Zhangren Chen and Xuekai Zhu and Ermo Hua and Kaiyan Zhang and Ning Ding and Bowen Zhou},
      year={2025},
      eprint={2501.18362},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2501.18362},
}
```
