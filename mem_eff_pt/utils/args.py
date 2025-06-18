import argparse
import torch

from mem_eff_pt.eff_pretraining import args_utils, training_utils


def parse_args(args):
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--debug", action='store_true')
    
    parser.add_argument("--dataset_path", type=str, default=None)
    
    parser.add_argument("--eval_at_begining", default=False, action="store_true")
    
    parser.add_argument("--start_tokenizing_idx", type=int, default=0)
    
    parser.add_argument("--no_slice", default=False, action="store_true")
    
    parser.add_argument("--adj_lr_muon", action='store_true')

    parser.add_argument("--keep_only_last_model", default=False, action="store_true")
    
    parser.add_argument("--adam_lr", type=float, default=2e-2) 
    parser.add_argument("--momentum", type=float, default=0.9) 
    parser.add_argument("--adam_beta_1", type=float, default=0.9) 
    parser.add_argument("--adam_beta_2", type=float, default=0.999) 
    
    # stream huggingface dataset instead of local
    parser.add_argument("--hf_dataset", default=False, action="store_true")
    
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts","cosine_quick_recovery"],
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for sgd
    parser.add_argument("--beta1", type=float, default=0.0)
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")

    parser.add_argument(
        "--distributed_type", type=str, default="ddp", choices=["fsdp", "ddp"]
    )

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)

    return args
