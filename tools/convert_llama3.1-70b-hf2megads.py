#!/usr/bin/env python

import os
import sys
import torch
import pickle
from safetensors import safe_open
from pathlib import Path
import torch.distributed as dist

# Hard-coded paths matching launch script
HOME = os.environ["HOME"]
MEGATRON_PATH = os.path.join(HOME, "Megatron-DeepSpeed-ORNL")
HF_PATH = os.path.join(HOME, "models/llama-3.1-70b")
SAVE_PATH = os.path.join(HOME, "models/llama-3.1-70b-megads")

# Add Megatron to path
sys.path.append(MEGATRON_PATH)

# Hard-coded Llama-3.1-70B configuration
MODEL_ARGS = {
    "num_layers": 80,
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "seq_length": 8192,
    "max_position_embeddings": 8192,
    "ffn_hidden_size": 28672,
    "true_vocab_size": 128000,
    "padded_vocab_size": 128256,
    "make_vocab_size_divisible_by": 128,
    "tensor_model_parallel_size": 8,
    "pipeline_model_parallel_size": 5,
}

def load_weights():
    """Load all Llama weights from HF safetensors
    
    The output has keys:
    dict_keys(['model.embed_tokens.weight', 
     'model.layers.0.input_layernorm.weight', 'model.layers.0.mlp.down_proj.weight', 'model.layers.0.mlp.gate_proj.weight', 
     'model.layers.0.mlp.up_proj.weight', 'model.layers.0.post_attention_layernorm.weight', 'model.layers.0.self_attn.k_proj.weight', 
     'model.layers.0.self_attn.o_proj.weight', 'model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.v_proj.weight', 
     'model.layers.1.mlp.gate_proj.weight', ...
     'model.layers.79.input_layernorm.weight', 'model.layers.79.mlp.down_proj.weight', 'model.layers.79.mlp.gate_proj.weight', 
     'model.layers.79.mlp.up_proj.weight', 'model.layers.79.post_attention_layernorm.weight', 'model.layers.79.self_attn.k_proj.weight', 
     'model.layers.79.self_attn.o_proj.weight', 'model.layers.79.self_attn.q_proj.weight', 'model.layers.79.self_attn.v_proj.weight', 
     'model.norm.weight', 'lm_head.weight'])

    model.embed_tokens.weight shape: torch.Size([128256, 8192])
    model.layers.0.input_layernorm.weight shape: torch.Size([8192])
    model.layers.0.mlp.down_proj.weight shape: torch.Size([8192, 28672])
    model.layers.0.mlp.gate_proj.weight shape: torch.Size([28672, 8192])
    model.layers.0.mlp.up_proj.weight shape: torch.Size([28672, 8192])
    model.layers.0.post_attention_layernorm.weight shape: torch.Size([8192])
    model.layers.0.self_attn.k_proj.weight shape: torch.Size([1024, 8192])
    model.layers.0.self_attn.o_proj.weight shape: torch.Size([8192, 8192])
    model.layers.0.self_attn.q_proj.weight shape: torch.Size([8192, 8192])
    model.layers.0.self_attn.v_proj.weight shape: torch.Size([1024, 8192])
    ...
    model.norm.weight shape: torch.Size([8192])
    lm_head.weight shape: torch.Size([128256, 8192])
     
    A few notes on this:
     - The self-attention and MLP weights define a transformer layer---there are 80 of these.
     - QKV are not symmetric due to grouped-query attention.
     - The output layer projects to the logits over the vocabulary with matrix lm_head.weight. 
        In some architectures this uses a transposed version of the embedding matrix, but LLAMA-3.1-70B 
        unties these, using a separate learned matrix. In other words, the output layer is untied from the embedding layer.
    """

    # Load from safetensors
    weights = {}
    safetensor_files = sorted(Path(HF_PATH).glob("*.safetensors"))
    if not safetensor_files:
        raise ValueError(f"No .safetensors files found in {HF_PATH}")

    print(f"Loading {len(safetensor_files)} model shards...")
    for shard_file in safetensor_files:
        print(f"Loading shard {shard_file}")
        with safe_open(shard_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                weights[k] = f.get_tensor(k)
    
    return weights

def prepare_attention_weights(weights, layer_idx):
    """Prepare QKV weights preserving GQA structure."""
    prefix = f"model.layers.{layer_idx}"
    
    # Get Q,K,V weights
    q = weights[f"{prefix}.self_attn.q_proj.weight"]
    k = weights[f"{prefix}.self_attn.k_proj.weight"]
    v = weights[f"{prefix}.self_attn.v_proj.weight"]
    
    # Simply concatenate Q,K,V without repeating K/V
    # Megatron-DeepSpeed will handle the GQA routing internally # CHECK THIS
    return torch.cat([q, k, v], dim=0)

def get_layer_weights(weights, layer_idx):
    """Extract all weights for a given transformer layer."""
    prefix = f"model.layers.{layer_idx}"
    layer_weights = {}
    
    # Add weights with flattened keys
    layer_weights[f"layers.{layer_idx}.input_layernorm.weight"] = weights[f"{prefix}.input_layernorm.weight"]
    layer_weights[f"layers.{layer_idx}.self_attention.query_key_value.weight"] = prepare_attention_weights(weights, layer_idx)
    layer_weights[f"layers.{layer_idx}.self_attention.dense.weight"] = weights[f"{prefix}.self_attn.o_proj.weight"]
    layer_weights[f"layers.{layer_idx}.post_attention_layernorm.weight"] = weights[f"{prefix}.post_attention_layernorm.weight"]
    layer_weights[f"layers.{layer_idx}.mlp.dense_h_to_4h.weight"] = torch.cat([
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights[f"{prefix}.mlp.up_proj.weight"]
    ], dim=0)
    layer_weights[f"layers.{layer_idx}.mlp.dense_4h_to_h.weight"] = weights[f"{prefix}.mlp.down_proj.weight"]
    
    return layer_weights

def shard_tensor(tensor, dim, chunks):
    """Split a tensor into chunks along specified dimension."""
    return torch.chunk(tensor, chunks, dim=dim)

def shard_vocab(tensor, tp_size, tp_rank):
    """Shard vocabulary across tensor parallel ranks.
    For vocab size V and tp_size T, each rank gets V/T entries."""
    vocab_size = tensor.shape[0]
    
    # Calculate shard size
    shard_size = vocab_size // tp_size
    
    # Get this rank's slice
    start_idx = tp_rank * shard_size
    end_idx = start_idx + shard_size
    return tensor[start_idx:end_idx]

def save_checkpoint(weights, pp_rank, tp_rank):
    """Save a specific pipeline-parallel and tensor-parallel shard."""
    # Calculate layer range for this pipeline stage
    layers_per_pp = MODEL_ARGS["num_layers"] // MODEL_ARGS["pipeline_model_parallel_size"]
    start_layer = pp_rank * layers_per_pp
    end_layer = start_layer + layers_per_pp
    
    # Create state dict for this shard
    state_dict = {
        "iteration": 1,
        "model": {"language_model": {"encoder": {}}}
    }
    
    # Add embedding layer if first pipeline stage
    if pp_rank == 0:
        # Shard vocabulary for this rank
        word_emb_shard = shard_vocab(weights["model.embed_tokens.weight"], MODEL_ARGS["tensor_model_parallel_size"], tp_rank)
        state_dict["model"]["language_model"]["embedding"] = {
            "word_embeddings": {"weight": word_emb_shard}
        }
    
    # Add transformer layers for this pipeline stage
    for local_idx, global_idx in enumerate(range(start_layer, end_layer)):
        layer_weights = get_layer_weights(weights, global_idx)
        
        # Convert global layer indices to local indices in the keys
        local_weights = {}
        for k, v in layer_weights.items():
            local_k = k.replace(f"layers.{global_idx}", f"layers.{local_idx}")
            local_weights[local_k] = v
        
        # Shard the weights that need tensor parallelism
        qkv_weight = local_weights[f"layers.{local_idx}.self_attention.query_key_value.weight"]
        dense_weight = local_weights[f"layers.{local_idx}.self_attention.dense.weight"]
        mlp_h_to_4h = local_weights[f"layers.{local_idx}.mlp.dense_h_to_4h.weight"]
        mlp_4h_to_h = local_weights[f"layers.{local_idx}.mlp.dense_4h_to_h.weight"]
        
        # Shard QKV and MLP weights
        qkv_shards = torch.chunk(qkv_weight, MODEL_ARGS["tensor_model_parallel_size"], dim=0)
        dense_shards = torch.chunk(dense_weight, MODEL_ARGS["tensor_model_parallel_size"], dim=1)
        mlp_h_to_4h_shards = torch.chunk(mlp_h_to_4h, MODEL_ARGS["tensor_model_parallel_size"], dim=0)
        mlp_4h_to_h_shards = torch.chunk(mlp_4h_to_h, MODEL_ARGS["tensor_model_parallel_size"], dim=1)
        
        local_weights[f"layers.{local_idx}.self_attention.query_key_value.weight"] = qkv_shards[tp_rank]
        local_weights[f"layers.{local_idx}.self_attention.dense.weight"] = dense_shards[tp_rank]
        local_weights[f"layers.{local_idx}.mlp.dense_h_to_4h.weight"] = mlp_h_to_4h_shards[tp_rank]
        local_weights[f"layers.{local_idx}.mlp.dense_4h_to_h.weight"] = mlp_4h_to_h_shards[tp_rank]
        
        state_dict["model"]["language_model"]["encoder"].update(local_weights)
    
    # Add final layernorm and output layer if last pipeline stage
    if pp_rank == MODEL_ARGS["pipeline_model_parallel_size"] - 1:
        # add the final layernorm (actually gamma for RMSNorm)
        state_dict["model"]["language_model"]["encoder"]["final_layernorm.weight"] = weights["model.norm.weight"]

        # Shard output layer vocabulary
        output_layer_shard = shard_vocab(
            weights["lm_head.weight"],
            MODEL_ARGS["tensor_model_parallel_size"],
            tp_rank
        )
        state_dict["model"]["language_model"]["output_layer"] = {
            "weight": output_layer_shard
        }
    
    # Save the checkpoint
    save_dir = os.path.join(SAVE_PATH, f"iter_0000001/mp_rank_{tp_rank:02d}_{pp_rank:03d}")
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(save_dir, "model_optim_rng.pt"))

# Create save directory
os.makedirs(SAVE_PATH, exist_ok=True)

# Load weights
print("Loading weights...")
weights = load_weights()

# print(weights.keys())

# print(f"model.embed_tokens.weight shape: {weights['model.embed_tokens.weight'].shape}")
# print(f"model.layers.0.input_layernorm.weight shape: {weights['model.layers.0.input_layernorm.weight'].shape}")
# print(f"model.layers.0.mlp.down_proj.weight shape: {weights['model.layers.0.mlp.down_proj.weight'].shape}")
# print(f"model.layers.0.mlp.gate_proj.weight shape: {weights['model.layers.0.mlp.gate_proj.weight'].shape}")
# print(f"model.layers.0.mlp.up_proj.weight shape: {weights['model.layers.0.mlp.up_proj.weight'].shape}")
# print(f"model.layers.0.post_attention_layernorm.weight shape: {weights['model.layers.0.post_attention_layernorm.weight'].shape}")
# print(f"model.layers.0.self_attn.k_proj.weight shape: {weights['model.layers.0.self_attn.k_proj.weight'].shape}")
# print(f"model.layers.0.self_attn.o_proj.weight shape: {weights['model.layers.0.self_attn.o_proj.weight'].shape}")
# print(f"model.layers.0.self_attn.q_proj.weight shape: {weights['model.layers.0.self_attn.q_proj.weight'].shape}")
# print(f"model.layers.0.self_attn.v_proj.weight shape: {weights['model.layers.0.self_attn.v_proj.weight'].shape}")
# print(f"model.norm.weight shape: {weights['model.norm.weight'].shape}")
# print(f"lm_head.weight shape: {weights['lm_head.weight'].shape}")

# Save each shard
for pp_rank in range(MODEL_ARGS["pipeline_model_parallel_size"]):
    for tp_rank in range(MODEL_ARGS["tensor_model_parallel_size"]):
        print(f"Saving shard pp={pp_rank}, tp={tp_rank}")
        save_checkpoint(weights, pp_rank, tp_rank)

# Save latest checkpoint file
with open(os.path.join(SAVE_PATH, "latest_checkpointed_iteration.txt"), "w") as f:
    f.write("1")

print("Done!")
