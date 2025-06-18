import torch
import torch.nn as nn
from mem_eff_pt.pt_scale import SCALE

def build_model(model, args):
    return model


def build_optimizer(model, trainable_params, args):   
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
            
    elif args.optimizer.lower() == "scale":

        main_params = []
        oned_params = []
        secondary_params = []
        main_modules_list = ["attn", "mlp","attention", "embed_tokens"]

        print(f"MAIN MODULES = {main_modules_list} !!!")
    
        id_to_name_main_params = {}
        id_to_name_secondary_params = {}
        id_to_name_oned_params = {}
    
        for module_name, module in model.named_modules():
            if not (isinstance(module, nn.Linear) or isinstance(module, nn.Embedding)):
                continue
            if not any(target_key in module_name for target_key in main_modules_list): 
                continue
            
            main_params.append(module.weight)
            id_to_name_main_params[id(module.weight)] = module_name
            
        for param_name, p in model.named_parameters():
            if id(p) in id_to_name_main_params:
                continue
            
            if p.ndim == 1:
                oned_params.append(p)
                id_to_name_oned_params[id(p)] = param_name
            else:
                secondary_params.append(p)
                id_to_name_secondary_params[id(p)] = param_name

        for module_name, module in model.named_modules():
            if hasattr(module, 'weight'):
                p = module.weight
                if id(p) in id_to_name_main_params:
                    print("Main module: ", module_name)
                if id(p) in id_to_name_oned_params:    
                    print("1D module: ", module_name)    
                if id(p) in id_to_name_secondary_params:    
                    print("Secondary module: ", module_name)          

        id_to_name = {**id_to_name_main_params, **id_to_name_secondary_params, **id_to_name_oned_params}
            
        # Create the optimizer
        optimizer = SCALE(        
                            lr=args.lr,
                            wd=args.weight_decay,
                            main_params=main_params,
                            secondary_params=secondary_params, 
                            oned_params=oned_params, 
                            id_to_name=id_to_name,
                            debug=args.debug,
                            momentum=args.momentum,
                            adam_lr=args.adam_lr,
                            adamw_betas=(args.adam_beta_1, args.adam_beta_2),
                            adamw_eps=1e-8,      
                            )        
        
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(
            trainable_params, lr=args.lr, weight_decay=args.weight_decay
        )
        
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.beta1,
        ) 
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    return optimizer


