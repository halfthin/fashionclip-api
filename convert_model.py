#!/usr/bin/env python3
"""Convert OpenCLIP checkpoint to HuggingFace CLIP format."""

from safetensors.torch import load_file, save_file
from pathlib import Path
import torch

SRC = "data/models/clip-laion2b/open_clip_model.safetensors"
DST = "data/models/models--laion--CLIP-ViT-B-16-laion2B-s34B-b88K/snapshots/default/model.safetensors"

state = load_file(SRC)
new_state = {}

# ===================== TEXT ENCODER =====================
# positional_embedding: [77, 512] -> text_model.embeddings.position_embedding
new_state["text_model.embeddings.position_embedding"] = state["positional_embedding"]

# token_embedding.weight: [49408, 512] -> text_model.embeddings.token_embedding.weight
new_state["text_model.embeddings.token_embedding.weight"] = state["token_embedding.weight"]

# ln_final -> text_model.final_layer_norm
new_state["text_model.final_layer_norm.bias"] = state["ln_final.bias"]
new_state["text_model.final_layer_norm.weight"] = state["ln_final.weight"]

# transformer.resblocks -> text_model.encoder.layers
for i in range(12):
    prefix = f"transformer.resblocks.{i}."
    # in_proj: [1536, 512] -> q_proj [512, 512], k_proj [512, 512], v_proj [512, 512]
    w = state[f"{prefix}attn.in_proj_weight"]  # [1536, 512]
    b = state[f"{prefix}attn.in_proj_bias"]    # [1536]
    d = 512
    new_state[f"text_model.encoder.layers.{i}.self_attn.q_proj.weight"] = w[:d].clone()
    new_state[f"text_model.encoder.layers.{i}.self_attn.q_proj.bias"] = b[:d].clone()
    new_state[f"text_model.encoder.layers.{i}.self_attn.k_proj.weight"] = w[d:2*d].clone()
    new_state[f"text_model.encoder.layers.{i}.self_attn.k_proj.bias"] = b[d:2*d].clone()
    new_state[f"text_model.encoder.layers.{i}.self_attn.v_proj.weight"] = w[2*d:].clone()
    new_state[f"text_model.encoder.layers.{i}.self_attn.v_proj.bias"] = b[2*d:].clone()
    # out_proj
    new_state[f"text_model.encoder.layers.{i}.self_attn.out_proj.weight"] = state[f"{prefix}attn.out_proj.weight"]
    new_state[f"text_model.encoder.layers.{i}.self_attn.out_proj.bias"] = state[f"{prefix}attn.out_proj.bias"]
    # ln_1, ln_2
    new_state[f"text_model.encoder.layers.{i}.layer_norm1.weight"] = state[f"{prefix}ln_1.weight"]
    new_state[f"text_model.encoder.layers.{i}.layer_norm1.bias"] = state[f"{prefix}ln_1.bias"]
    new_state[f"text_model.encoder.layers.{i}.layer_norm2.weight"] = state[f"{prefix}ln_2.weight"]
    new_state[f"text_model.encoder.layers.{i}.layer_norm2.bias"] = state[f"{prefix}ln_2.bias"]
    # mlp
    new_state[f"text_model.encoder.layers.{i}.mlp.fc1.weight"] = state[f"{prefix}mlp.c_fc.weight"]
    new_state[f"text_model.encoder.layers.{i}.mlp.fc1.bias"] = state[f"{prefix}mlp.c_fc.bias"]
    new_state[f"text_model.encoder.layers.{i}.mlp.fc2.weight"] = state[f"{prefix}mlp.c_proj.weight"]
    new_state[f"text_model.encoder.layers.{i}.mlp.fc2.bias"] = state[f"{prefix}mlp.c_proj.bias"]

# text_projection
new_state["text_projection"] = state["text_projection"]

# ===================== VISION ENCODER =====================
# class_embedding
new_state["vision_model.class_embedding"] = state["visual.class_embedding"]
# conv1 -> patch_embedding
new_state["vision_model.embeddings.patch_embedding.weight"] = state["visual.conv1.weight"]
# positional_embedding [197, 768] -> position_embedding
new_state["vision_model.embeddings.position_embedding"] = state["visual.positional_embedding"]
# ln_pre
new_state["vision_model.pre_layernorm.weight"] = state["visual.ln_pre.weight"]
new_state["vision_model.pre_layernorm.bias"] = state["visual.ln_pre.bias"]
# ln_post
new_state["vision_model.post_layernorm.weight"] = state["visual.ln_post.weight"]
new_state["vision_model.post_layernorm.bias"] = state["visual.ln_post.bias"]

# visual.transformer.resblocks -> vision_model.encoder.layers
for i in range(12):
    prefix = f"visual.transformer.resblocks.{i}."
    # in_proj: [2304, 768] -> q, k, v each [768, 768]
    w = state[f"{prefix}attn.in_proj_weight"]
    b = state[f"{prefix}attn.in_proj_bias"]
    d = 768
    new_state[f"vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = w[:d].clone()
    new_state[f"vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = b[:d].clone()
    new_state[f"vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = w[d:2*d].clone()
    new_state[f"vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = b[d:2*d].clone()
    new_state[f"vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = w[2*d:].clone()
    new_state[f"vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = b[2*d:].clone()
    # out_proj
    new_state[f"vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = state[f"{prefix}attn.out_proj.weight"]
    new_state[f"vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = state[f"{prefix}attn.out_proj.bias"]
    # ln_1, ln_2
    new_state[f"vision_model.encoder.layers.{i}.layer_norm1.weight"] = state[f"{prefix}ln_1.weight"]
    new_state[f"vision_model.encoder.layers.{i}.layer_norm1.bias"] = state[f"{prefix}ln_1.bias"]
    new_state[f"vision_model.encoder.layers.{i}.layer_norm2.weight"] = state[f"{prefix}ln_2.weight"]
    new_state[f"vision_model.encoder.layers.{i}.layer_norm2.bias"] = state[f"{prefix}ln_2.bias"]
    # mlp
    new_state[f"vision_model.encoder.layers.{i}.mlp.fc1.weight"] = state[f"{prefix}mlp.c_fc.weight"]
    new_state[f"vision_model.encoder.layers.{i}.mlp.fc1.bias"] = state[f"{prefix}mlp.c_fc.bias"]
    new_state[f"vision_model.encoder.layers.{i}.mlp.fc2.weight"] = state[f"{prefix}mlp.c_proj.weight"]
    new_state[f"vision_model.encoder.layers.{i}.mlp.fc2.bias"] = state[f"{prefix}mlp.c_proj.bias"]

# visual.proj -> visual_projection
new_state["visual_projection"] = state["visual.proj"]

# logit_scale
new_state["logit_scale"] = state["logit_scale"]

print(f"Converted {len(state)} keys to {len(new_state)} HuggingFace keys")
save_file(new_state, DST)
print(f"Saved to {DST}")
