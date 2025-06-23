# A Minimalist Optimizer Design for LLM Pretraining

<a href="https://arxiv.org/abs/2506.16659">
  <img src="https://img.shields.io/static/v1?label=arXiv&message=2506.16659&color=b31b1b" />
</a>

Preliminary code release for our paper "A Minimalist Optimizer Design for LLM Pretraining", by Athanasios Glentis, Jiaxiang Li,  Andi Han and Mingyi Hong.

## Abstract

Training large language models (LLMs) typically relies on adaptive optimizers such as Adam, which require significant memory to maintain first- and second-moment matrices, known as optimizer states. While recent works such as GaLore, Fira, and APOLLO have proposed state-compressed variants to reduce memory consumption, a fundamental question remains: What is the minimal amount of optimizer state that is truly necessary to retain state-of-the-art performance in LLM pretraining? In this work, we systematically investigate this question using a bottom-up approach. We find that two memory- and compute-efficient optimization techniques are particularly effective: (1) column-wise gradient normalization significantly boosts the performance of plain SGD without requiring momentum; and (2) adding first-order momentum only to the output layer - where gradient variance is highest - yields performance competitive with fully adaptive methods such as Muon. Based on these insights, we propose SCALE (Stochastic Column-normalized Last-layer Momentum), a new optimizer that combines column-normalized SGD with last-layer momentum, where column normalization refers to normalizing the gradient along the output dimension. Across multiple LLaMA models (60M-1B), SCALE matches or exceeds the performance of Adam while using only 35-45% of the total memory. It also consistently outperforms memory-efficient optimizers such as GaLore, Fira, and APOLLO, making it a strong candidate for large-scale pretraining under memory constraints. For the LLaMA 7B model, SCALE outperforms the state-of-the-art method APOLLO in terms of both perplexity and memory consumption. In addition, our method serves as a minimalist baseline for more sophisticated optimizer design. 


## Usage

The SCALE optimizer code can be found in `mem_eff_pt/pt_scale/scale_optimizer.py`. The repository provides the scripts used in our experiments in the `scripts/` directory.

Example script (`350m_scale.sh`):

```bash
torchrun --standalone --nproc_per_node 4 torchrun_main_DDP.py \
    --model_name 350m_scale \
    --model_config configs/llama_350m.json \
    --optimizer scale \
    --lr 1e-3 \
    --momentum 0.9 \
    --weight_decay 0.0 \
    --batch_size 128 \
    --total_batch_size 512 \
    --num_training_steps 60000 \
    --warmup_steps 6000 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --save_every 99999 \
    --seed 42  \
    --scheduler cosine \
    --dataset_path /path/to/c4/en \
```


## Citation

If you find this work useful for your research, please cite our paper:
```bibtex
@article{glentis2025minimalist,
  title={A Minimalist Optimizer Design for LLM Pretraining},
  author={Glentis, Athanasios and Li, Jiaxiang and Han, Andi and Hong, Mingyi},
  journal={arXiv preprint arXiv:2506.16659},
  year={2025}
}
```
