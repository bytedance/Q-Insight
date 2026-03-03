<div align="center">
<h2>

Q-Insight Family
</h2>

  <a href="https://arxiv.org/abs/2503.22679">
    <img
      src="https://img.shields.io/badge/QInsight-Paper-red?logo=arxiv&logoColor=red"
      alt="Q-Insight Paper on arXiv"
    />
  </a>
  <a href="https://arxiv.org/abs/2506.18564">
    <img
      src="https://img.shields.io/badge/VQInsight-Paper-red?logo=arxiv&logoColor=red"
      alt="VQ-Insight Paper on arXiv"
    />
  </a>
    <a href="https://arxiv.org/abs/2510.11369">
    <img
      src="https://img.shields.io/badge/RALI-Paper-red?logo=arxiv&logoColor=red"
      alt="RALI Paper on arXiv"
    />
  </a>
<a href="https://huggingface.co/ByteDance/Q-Insight">
    <img 
        src="https://img.shields.io/badge/QInsightFamily-Model-yellow?logo=huggingface&logoColor=yellow" 
        alt="Q-Insight Family Model"
    />
</a>


</div>

## 🚩 Updates
- 2026.03.03 The complete code of RALI and VQ-Insight are released at [RALI](https://github.com/xuanyuzhang21/RALI) and [VQ-Insight](https://github.com/xuanyuzhang21/VQ-Insight).
- 2026.02.06 The code and pretrained model of RALI and VQ-Insight are released!
- 2026.02.06 RALI has been accepted at ICLR 2026 as an **oral** presentation!
- 2025.11.08 VQ-Insight has been accepted at AAAI 2026 as an **oral** presentation!
- 2025.09.19 Q-Insight has been accepted at NeurIPS 2025 as a **spotlight** (Top 3%)!
- 2025.05.30 Released training and testing code, along with the pretrained model.
- 2025.05.26 Released our v2 paper.
- 2025.03.28 Released the Q-Insight technical report.

## 🔥 Introduction
(✏️: * denotes equal contribution, # denotes project leader, † denotes corresponding author)

<h3>
Q-Insight: Understanding Image Quality via Visual Reinforcement Learning
</h3>

[Weiqi Li](https://scholar.google.com/citations?user=SIkQdEsAAAAJ), [Xuanyu Zhang](https://scholar.google.com/citations?user=Sq2q-E8AAAAJ&hl=zh-CN&oi=ao), Shijie Zhao<sup>#, †</sup>, Yabin Zhang, Junlin Li, Li Zhang and [Jian Zhang](https://jianzhang.tech/)<sup>†</sup>

PLCC comparisons between our proposed Q-Insight and existing IQA metrics (left) and three example applications of our Q-Insight (right) are presented. Q-Insight demonstrates significantly improved performance compared to existing methods, especially on out-of-domain datasets. Additionally, Q-Insight effectively supports quality score regression, image degradation perception, and zero-shot image comparison reasoning tasks.
<p align="center">
  <img src="assets/teaser.png">
</p>


<h3>
VQ-Insight: Teaching VLMs for AI-Generated Video Quality Understanding via Progressive Visual Reinforcement Learning
</h3>

[Xuanyu Zhang*](https://scholar.google.com/citations?user=Sq2q-E8AAAAJ&hl=zh-CN&oi=ao), [Weiqi Li*](https://scholar.google.com/citations?user=SIkQdEsAAAAJ), Shijie Zhao<sup>#,†</sup>, Junlin Li, Li Zhang, Jian Zhang<sup>†</sup>

We propose a reasoning-style vision-language model VQ-Insight, which accurately performs AIGC video preference comparison, AIGC video multi-dimension scoring, and natural video scoring, accompanied by detailed and reasonable reasoning processes. Our VQ-Insight can be applied to post-training of video generation models and zero-shot content repairing.

<p align="center">
  <img src="assets/teaser_vqinsight.png">
</p>

<h3>
Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment
</h3>

Shijie Zhao<sup>*,#</sup>, Xuanyu Zhang<sup>*</sup>, Weiqi Li, Junlin Li, Li Zhang, Tianfan Xue, Jian Zhang

We revisit the reasoning mechanism in MLLM-based IQA model (such as Q-Insight) and propose a CLIP-based lightweight image scorer RALI. We verifies that through RL training, MLLMs leverage their reasoning capability to convert redundant visual representations into compact, cross-domain aligned text representations. This conversion is the source of the generalization exhibited by these reasoning-based IQA models. RALI uses only about 4% of Q-Insight’s parameters and inference time, while achieving comparable accuracy.

<p align="center">
  <img src="assets/teaser_rali.png">
</p>


## 🔧 Dependencies and Installation
```bash
git clone https://github.com/bytedance/Q-Insight.git
bash setup.sh
```

To run VQ-Insight, install additional pacakages.
```bash
cd src/eval/qwen-vl-utils
pip install -e .[decord]
```

## ⚡ Quick Inference
### Demo for RALI
Please download the **RALI** pretrained weights from the [link](https://huggingface.co/ByteDance/Q-Insight/tree/main/RALI). After downloading, place the checkpoint under `Q-Insight/checkpoints`, so that the directory structure becomes:

```text
Q-Insight/
├── checkpoints/
│   ├── ckpt.pt
│   ├── pca.pkl
│   ├── basis.npz
│   └── best/
│       ├── config.json
│       ├── pytorch_model.bin (or *.safetensors)
│       ├── preprocessor_config.json
│       └── ...
├── src/
├── assets/
└── README.md
```
Then run the following code: 
```bash
cd src/eval/
python demo_rali_score.py
```

### Demo for VQ-Insight 
#### Natural Video Scoring
```bash
cd src/eval/
python demo_vqinsight_score.py \
  --video_path "../../assets/demo_natural.mp4" \
  --video_type natural
```

#### AIGC Video Multi-Dimension Scoring
```bash
cd src/eval/
python demo_vqinsight_score.py \
  --video_path "../../assets/demo_aigc.mp4" \
  --video_type aigc
```

#### AIGC Video Comparison
```bash
cd src/eval/
python demo_vqinsight_comp.py \
  --video_a "../../assets/demo_comp1.mp4" \
  --video_b "../../assets/demo_comp2.mp4" \
  --model_name_or_path Bytedance/Q-Insight

```

### Demo for Q-Insight 
#### Score Regression
```
cd src/eval/
python demo_score.py
``` 
#### Degradation Perception
```
cd src/eval/
python demo_dist.py
```
#### Image Comparison Reasoning
```
cd src/eval/
python demo_comparison.py
```

## 📖 Dataset Preparation for Training Q-Insight
#### Score Regression
Download meta files from [Data-DeQA-Score](https://huggingface.co/datasets/zhiyuanyou/Data-DeQA-Score/tree/main) and the source images from the [KONIQ](https://database.mmsp-kn.de/koniq-10k-database.html) dataset.
Arrange the folders in `./src/open-r1-multimodal/data`as follows:
```
|-- Data-DeQA-Score
  |-- KONIQ
    |-- images/*.jpg
    |-- metas
```
#### Degradation Perception
Download the `refA_sd_brief` subset from [KADIS-700K](https://modelscope.cn/datasets/zhiyuanyou/DataDepictQA/files).
Arrange the folders in `./src/open-r1-multimodal/data` as follows:
```
|-- KADIS-700K
  |-- refA_sd_brief
    |-- dist_imgs/*.jpg
    |-- metas/train_dist.json
```

#### Image Comparison Reasoning
Download the validation dataset of [DiffIQA](https://drive.google.com/drive/folders/1vZehlUPDyDfo6Mq1K8pAMe3pcjqdDRht).
Arrange the folders in `./src/open-r1-multimodal/data` as follows:
```
|-- DiffIQA
  |-- ValidationImage
    |-- images
    |-- train_comparison.json
```

## Training Q-Insight
#### Score Regression and Degradation Perception
```
cd src/open-r1-multimodal/
bash run_qinsight_score_and_dist.sh
```
#### Image Comparison Reasoning
```
cd src/open-r1-multimodal/
bash run_qinsight_comparison.sh
```

## Training VQ-Insight and RALI

The training code of VQ-Insight and RALI will be released at [RALI](https://github.com/xuanyuzhang21/RALI) and [VQ-Insight](https://github.com/xuanyuzhang21/VQ-Insight).

## ✏️ To Do List
- [x] Release the code and model of VQ-Insight
- [ ] Add support for LoRA fine-tuning
- [ ] Provide a Gradio demo
- [x] Release inference code and weights
- [x] Release training code
- [x] Release the paper

## Acknowledgement
We appreciate the releasing codes and data of [VLM-R1](https://github.com/om-ai-lab/VLM-R1), [DepictQA](https://github.com/XPixelGroup/DepictQA) and [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score).



## Citation
If Q-Insight Family is helpful, please help to ⭐ the repo.

If you find the code helpful in your research or work, please cite the following papers:
```
@article{li2025qinsight,
  title={Q-Insight: Understanding Image Quality via Visual Reinforcement Learning},
  author={Li, Weiqi and Zhang, Xuanyu and Zhao, Shijie and Zhang, Yabin and Li, Junlin and Zhang, Li and Zhang, Jian},
  journal={Proceedings of the Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```
```
@article{zhang2025vqinsight,
  title={VQ-Insight: Teaching VLMs for AI-Generated Video Quality Understanding via Progressive Visual Reinforcement Learning},
  author={Zhang, Xuanyu and Li, Weiqi and Zhao, Shijie and Li, Junlin and Zhang, Li and Zhang, Jian},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2026}
}
```
```
@article{zhao2025reasoning,
  title={Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment},
  author={Zhao, Shijie and Zhang, Xuanyu and Li, Weiqi and Li, Junlin and Zhang, Li and Xue, Tianfan and Zhang, Jian},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
