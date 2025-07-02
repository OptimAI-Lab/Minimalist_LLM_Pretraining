# A Minimalist Optimizer Design for LLM Pretraining

<a href="https://arxiv.org/abs/2506.16659">
  <img src="https://img.shields.io/static/v1?label=arXiv&message=2506.16659&color=b31b1b" />
</a>

Preliminary code release for our paper "A Minimalist Optimizer Design for LLM Pretraining", by Athanasios Glentis, Jiaxiang Li,  Andi Han and Mingyi Hong.

## SCALE Optimizer

We introduce our proposed optimizer, SCALE, detailed in Algorithm 1. The design of SCALE is motivated by empirical insights from our experiments, which highlighted the importance of stabilizing updates in the last layer and controlling gradient scale via column-wise normalization. Accordingly, SCALE integrates two components: 

- column-wise normalization of gradients
- first order momentum restricted to the last layer

Despite its minimalist design, SCALE is highly effective, outperforming baselines such as Adam and achieving performance competitive with state-of-the-art Stable-SPAM and Muon, while using substantially less memory.

<div align="center">
  <img src="imgs/scale_algorithm.png" alt="Image 1" style="width: 600px; margin: 0 auto;">
</div>



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
