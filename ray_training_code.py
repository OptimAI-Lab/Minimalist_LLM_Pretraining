import sys
import os
import time
import random
import argparse

from dataclasses import dataclass

from safetensors.torch import load_file
import json

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
from transformers import logging as hf_logging
hf_logging.set_verbosity_error() 

import torch.distributed as dist

from torch.utils.data import IterableDataset, get_worker_info

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
import datasets
from datasets import load_dataset

from mem_eff_pt.peft_pretraining.modeling_llama import LlamaForCausalLM
from mem_eff_pt.utils.train_utils import build_optimizer

import wandb
import numpy as np

from mem_eff_pt.peft_pretraining.training_utils import get_scheculer

import ray
from ray import train
from ray.train import Checkpoint
from ray.train.torch import TorchTrainer, get_device, prepare_model
from ray.air import RunConfig
from ray.air.config import ScalingConfig

# Hugging Face logging
transformers.logging.set_verbosity_error()

# (Keep your SDP toggles)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

 


# -------------------------
# Dataset utilities
# -------------------------
def _get_ray_rank_and_world():
    """Returns (rank, world_size) if under Ray Train; otherwise (0, 1)."""
    try:
        ctx = train.get_context()
        if ctx is None:
            return 0, 1
        return ctx.get_world_rank(), ctx.get_world_size()
    except Exception:
        return 0, 1
    

class PreprocessedIterableDataset(IterableDataset):
    """
    Streaming dataset wrapper that applies tokenizer + batching.
    Sharding is handled outside (via HF '.shard()').
    """
    def __init__(self, data, tokenizer, batch_size, max_length, start_tokenizing_idx=None):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.start_tokenizing_idx = start_tokenizing_idx
        self.k = 0

    def __iter__(self):   
        iter_data = iter(self.data)
        batch = []
        for example in iter_data:
            if self.start_tokenizing_idx is None or self.k > self.start_tokenizing_idx :
                tokenized_example = self.tokenizer(
                    example["text"],
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                batch.append(tokenized_example)
            else:
                batch.append(0)

            if len(batch) == self.batch_size:
                yield self._format_batch(batch)
                batch = []
                self.k += 1

        if batch:
            yield self._format_batch(batch)

    def _format_batch(self, batch):
        if self.start_tokenizing_idx is None or self.k > self.start_tokenizing_idx:
            input_ids = torch.stack([item["input_ids"].squeeze(0) for item in batch])
            attention_mask = torch.stack([item["attention_mask"].squeeze(0) for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            return 0

def collate_fn(batch_list):
    batch = {
        "input_ids": torch.stack([torch.Tensor(example["input_ids"]).long() for example in batch_list]),
        "attention_mask": torch.stack([torch.Tensor(example["attention_mask"]).long() for example in batch_list]),
    }
    return batch


def batch_fn(dataset, batch_size):
    batch = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            batch = collate_fn(batch)
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch


@torch.no_grad()
def evaluate_model(model, preprocess_batched, pad_idx, device, batch_size, args):
    if not args.hf_dataset:
        data_files_val= {"validation": [f"{args.dataset_path}/c4-validation.{str(i).zfill(5)}-of-00008.json.gz" for i in range(0,8)]}
        val_data = datasets.load_dataset(path=args.dataset_path,  data_files=data_files_val, split="validation", streaming=True)
    else:
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        ) 

    val_data = val_data.shuffle(seed=42) 

    rank, world_size = _get_ray_rank_and_world()
    

    val_data = datasets.distributed.split_dataset_by_node(
            val_data, rank=rank, world_size=world_size
        )
    

    val_data_mapped = val_data.map(
        preprocess_batched,
        batched=True,
        remove_columns=["text", "timestamp", "url"],
    )
    val_data_mapped.batch = lambda batch_size: batch_fn(val_data_mapped, batch_size)

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0, device=device)
    
    
    total_batches = 1  # set to 1 to follow GaLore codebase
    
    if args.disable_glr_eval:
        total_batches = 0
    
    for batch in val_data_mapped.batch(batch_size=batch_size):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        
        if args.amp and not args.eval_in_fp32:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = model(**batch, labels=labels).loss
        else:
            loss = model(**batch, labels=labels).loss        

 
        total_loss += loss.detach()
        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size # * world_size !!


    total_loss = total_loss / total_batches
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens



# -------------------------
# Argparse stays (used on driver)
# -------------------------
def parse_args(args):
    parser = argparse.ArgumentParser()


     
    parser.add_argument("--sweep", action='store_true')  

    parser.add_argument("--opt_model", default=False, action="store_true")
    
    parser.add_argument("--gpt_model", default=False, action="store_true")
 
    parser.add_argument("--start_tokenizing_idx", type=int, default=None)
    
    parser.add_argument("--keep_only_last_model", default=False, action="store_true")
    
    parser.add_argument("--save_dir", type=str, default=None)
    
    parser.add_argument("--save_every", type=int, default=999999)
    
    parser.add_argument("--continue_from", type=str, default=None)
    
    parser.add_argument("--qwen_model", action='store_true') 
    
    parser.add_argument("--eval_in_fp32", action='store_true') 

    parser.add_argument("--adj_lr_full", action='store_true') 
     
    parser.add_argument("--disable_glr_eval", action='store_true') 
  
    parser.add_argument("--save_grads", action='store_true') 
     
    parser.add_argument("--cycle_length", type=int, default=None, help="Number of steps per cycle for cosine scheduler", )
    parser.add_argument( "--recovery_steps", type=int, default=10,  help="Number of steps for cosine restarts (only used for cosine_restarts)",)    
    
    parser.add_argument( "--scheduler",  type=str,  default="cosine",  choices=["linear", "cosine", "cosine_restarts","cosine_quick_recovery", "warmup_constant", "wsd_quick_recovery", "exp_cosine"], )
    
    parser.add_argument("--decay_steps", type=int, default=1000)

    parser.add_argument("--amp", action="store_true", help="Enable PyTorch AMP mixed precision training")
    parser.add_argument("--grad_clipping_norm", type=float, default=0.0)

    parser.add_argument("--wandb_entity", type=str)

    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)

    parser.add_argument("--adam_lr", type=float, default=None)
    
    parser.add_argument("--momentum", type=float, default=0.9) 
    parser.add_argument("--adam_beta_1", type=float, default=0.9) 
    parser.add_argument("--adam_beta_2", type=float, default=0.999) 

    parser.add_argument("--compile_mode", default="default")
    parser.add_argument("--compile_model", action='store_true')  
    parser.add_argument("--optimizer", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--wandb_project_name", type=str)
    parser.add_argument("--hf_dataset", default=False, action="store_true")

    parser.add_argument("--dataset_path", type=str, default="/c4/en")
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int )
    parser.add_argument("--eval_every", type=int )
    parser.add_argument("--num_training_steps", type=int,
                        help="Number of **update steps** to train for.")
    parser.add_argument("--dtype", type=str,
                        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=8, help="PyTorch DataLoader workers per Ray worker")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clipping", type=float, default=0.0)

    # Ray
    parser.add_argument("--ray_num_workers", type=int, default=2, help="Ray Train workers (processes)")
    parser.add_argument("--ray_use_gpu", action="store_true", help="Set if each Ray worker should use 1 GPU")
    parser.add_argument("--ray_cpus_per_worker", type=int, default=2, help="CPUs per Ray worker for DataLoader")
    return parser.parse_args(args)


# -------------------------
# Ray Train worker loop
# -------------------------
def training_loop_per_worker(config):
    # Pull config into a simple namespace-ish dict
    args = argparse.Namespace(**config)

    # Seeding (per worker)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Device from Ray Train (binds correct local_rank / CUDA)
    device = get_device()
    
    rank, world_size = _get_ray_rank_and_world()


    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert (
                args.total_batch_size % world_size == 0
            ), "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (
                args.batch_size * world_size
            )
            assert (
                args.gradient_accumulation > 0
            ), "gradient_accumulation must be greater than 0"
    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"
    


    # -------------------------
    # Data: streaming C4
    # -------------------------

    is_rank0 = train.get_context().get_world_rank() == 0
    
    
    args.is_rank0_flag = is_rank0
    

    if is_rank0:
        if args.sweep:
            wd_entity = args.wandb_run_entity
            wd_project = args.wandb_run_project
            run_id =args.wandb_run_id        
            wandb.init( entity=wd_entity, project=wd_project, id=run_id, resume="must" )
            wb_config = wandb.config
            wandb.config.update(config, allow_val_change=True)
        else:
            wandb.init(entity=args.wandb_entity, project=args.wandb_project_name, name=args.model_name)
            wandb.config.update(config, allow_val_change=True)            

    
    if not args.hf_dataset:
        data_files_train = {"train": [f"{args.dataset_path}/c4-train.{str(i).zfill(5)}-of-01024.json.gz" for i in range(0,1024)]}
        data = load_dataset(path=args.dataset_path,  data_files=data_files_train, split="train", streaming=True)
        data = data.shuffle(seed=42, buffer_size=100_000) 
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True ) 
        data = data.shuffle(seed=42, buffer_size=100_000) 

    data = data.shard(num_shards=world_size, index=rank)


    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=args.max_length)

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch


    dataset = PreprocessedIterableDataset(
        data, tokenizer, batch_size=args.batch_size, max_length=args.max_length, start_tokenizing_idx = args.start_tokenizing_idx
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=None, num_workers=args.workers
    )
    

    rank = train.get_context().get_world_rank()


    # -------------------------
    # Model
    # -------------------------

    model_config = AutoConfig.from_pretrained(args.model_config)
    
    if args.use_hf_model:
        model = AutoModelForCausalLM.from_config(model_config)   
        print(model)
    else:
        model = LlamaForCausalLM(model_config)
        print(model)
        
        
    if args.compile_model:
        model = torch.compile(model, mode=args.compile_mode)
        print(f"compiled model, mode = {args.compile_mode}")


    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()

    # dtype & device
    if args.dtype in ["bf16", "bfloat16"]:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)


    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = build_optimizer(model, trainable_params, args)
    
    scheduler = get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,  # Use total steps
            warmup_steps=args.warmup_steps, 
            min_lr_ratio=args.min_lr_ratio,
            cycle_length=args.cycle_length,  # Restart interval
            recovery_steps=args.recovery_steps,
            force_step=args.force_step,
            force_lr=args.force_lr,
            decay_steps=args.decay_steps,
        )   
    

    global_step = 0
    update_step = 0
    tokens_seen = 0
  


    model = prepare_model(model)

    pad_idx = tokenizer.pad_token_id
    local_step = 0

    max_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    # Progress bar only on rank 0
    is_rank0 = train.get_context().get_world_rank() == 0


    pbar = None
    if is_rank0:
        from tqdm import tqdm

        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps",
                ncols=80, leave=True, position=0,
                dynamic_ncols=False, ascii=True, file=sys.stdout)


    # -------------------------
    # TRAINING LOOP
    # -------------------------


    if is_rank0 and args.save_ptl:
        
        token_loss_log = []
        
        vocab_size = tokenizer.vocab_size
        token_id_to_idx = {i: i for i in range(vocab_size)}  # simple identity mapping
        loss_sum_array = np.zeros(vocab_size, dtype=np.float32)
        count_array = np.zeros(vocab_size, dtype=np.int32)
        
        id_to_token = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]

        token_loss_log.append(id_to_token)
    
    

    for batch_idx, batch in enumerate(dataloader):

        global_step += 1
        local_step += 1

        if update_step >= args.num_training_steps:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == pad_idx] = -100
        tokens_seen += (batch["input_ids"] != pad_idx).sum().item()




        if args.amp:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss # [batch_size, seq_len - 1]
        else:
            outputs = model(**batch, labels=labels)
            loss = outputs.loss

        scaled_loss = loss / args.gradient_accumulation
        scaled_loss.backward()
        

        if global_step % args.gradient_accumulation != 0:
            continue
        
 
        if args.grad_clipping_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping_norm)

        optimizer.step()
        scheduler.step()
        

        optimizer.zero_grad()
        update_step += 1

        if  is_rank0 and pbar is not None:
            pbar.update(1)

        lr = optimizer.param_groups[0]["lr"]

        if is_rank0 and wandb.run is not None :
            wandb.log(
                {
                    "lr": lr,
                    "loss": loss.item(),
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "max_memory": max_memory,
                },
                step=update_step
            )


        # save checkpoint by save_every
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
            and is_rank0
        ):
            if args.keep_only_last_model:
                current_model_directory = f"{args.save_dir}/model_last"
            else:
                current_model_directory = f"{args.save_dir}/model_{update_step}"
            print(
                f"Saving model and optimizer to {current_model_directory}, update step {update_step}"
            )
            os.makedirs(args.save_dir, exist_ok=True)
            
            if args.qwen_model:
            
                model.module.save_pretrained(
                    current_model_directory, max_shard_size="100GB", safe_serialization=False
                )
            else:
                model.module.save_pretrained(
                    current_model_directory, max_shard_size="100GB", safe_serialization=False
                )                

            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "wandb": wandb.run.dir,
                "dtype": args.dtype,
                # "config": run_config,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

            training_state_checkpoint = {
                "global_step": global_step,
                "update_step": update_step,
                "tokens_seen": tokens_seen,
            }
            with open(f"{current_model_directory}/training_state.json", "w") as f:
                json.dump(training_state_checkpoint, f, indent=4)

            # save wandb related info
            wandb_info = {
                "wandb_id": wandb.run.id,
            }
            with open(f"{args.save_dir}/wandb.json", "w") as f:
                json.dump(wandb_info, f, indent=4) 
 

        # Evaluation
        if args.eval_every > 0 and ((update_step % args.eval_every == 0) or (update_step == args.num_training_steps) ): 
                     
            total_loss, evaluated_on_tokens = evaluate_model(
                model, preprocess_batched, pad_idx, device, args.batch_size, args,
            )
 
            if is_rank0 and ( wandb.run is not None):
                print(f"[Eval Step {update_step}] Loss: {total_loss:.4f}, PPL: {np.exp(total_loss):.2f}", flush=True)
                wandb.log(
                                {
                                    "final_eval_loss": total_loss,
                                    "final_eval_perplexity": np.exp(total_loss),
                                    "final_eval_tokens": evaluated_on_tokens,
                                },
                                step=update_step
                            ) 
                

    # Close progress bar
    if is_rank0 and pbar is not None:
        pbar.close()

    # Cleanup
    del loss, optimizer, scheduler
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    if is_rank0 and update_step == args.num_training_steps and wandb.run is not None:
        #wandb.finish() 
        wandb.finish(exit_code=0, quiet=True)
        wandb.teardown()
        time.sleep(5) 


def main_driver(args):
    # Build Ray Train config dictionary passed to each worker
    worker_config = vars(args)
    
    if args.sweep:

        run = wandb.init(entity=args.wandb_entity, project=args.wandb_project_name, name=args.model_name)
        wb_config = wandb.config

        print("Driver wandb config : ", wb_config)    
        worker_config["lr"] = float(wb_config.lr)
    
        worker_config["wandb_run_id"] = run.id 
        worker_config["wandb_run_entity"] = run.entity
        worker_config["wandb_run_project"] = run.project

        # Immediately disable W&B in the driver
        wandb.finish()

        # turn off syncing
        wandb.teardown()

    # Start Ray locally if not connected to a cluster
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)  # connects to cluster if RAY_ADDRESS set / head running

    scaling = ScalingConfig(
        num_workers=args.ray_num_workers,
        use_gpu=args.ray_use_gpu,
        resources_per_worker={"CPU": args.ray_cpus_per_worker},  # optional
    )

    trainer = TorchTrainer(
        training_loop_per_worker,
        train_loop_config=worker_config,
        scaling_config=scaling,
        run_config=RunConfig(name="llama_c4_stream_train"),
    )

    result = trainer.fit()
    """
    if result.checkpoint:
        ckpt_path = "ray_ckpt.pt"
        state = result.checkpoint.to_dict()
        torch.save(state, ckpt_path)
        print(f"Saved final checkpoint to {ckpt_path}")
    """


if __name__ == "__main__":
    print("Starting Ray script")
    cli_args = parse_args(None)
    main_driver(cli_args)
