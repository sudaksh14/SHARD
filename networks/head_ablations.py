import torch

# =========================
# Head Importance (Shard)
# =========================
@torch.no_grad()
def compute_head_importance_shard(attn):
    # attn.set_level_use(attn.max_level())

    max_heads = attn.num_heads
    hs_max = attn.embed_dim // attn.num_heads

    w_in = attn.in_proj_weight.data.view(3, max_heads, hs_max, attn.embed_dim)

    importance = torch.zeros(max_heads, device=w_in.device)

    for h in range(max_heads):
        q = w_in[0, h]
        k = w_in[1, h]
        v = w_in[2, h]

        importance[h] = q.norm() + k.norm() + v.norm()

    return importance


# =========================
# Reorder Input Projection
# =========================
@torch.no_grad()
def reorder_in_proj(attn, perm):
    max_heads = attn.num_heads
    hs_max = attn.embed_dim // attn.num_heads

    # weights
    w = attn.in_proj_weight.data.view(3, max_heads, hs_max, attn.embed_dim)
    w = w[:, perm, :, :]
    attn.in_proj_weight.data = w.reshape(3 * attn.embed_dim, attn.embed_dim)

    # bias
    b = attn.in_proj_bias.data.view(3, max_heads, hs_max)
    b = b[:, perm, :]
    attn.in_proj_bias.data = b.reshape(3 * attn.embed_dim)


# =========================
# Reorder Output Projection
# =========================
@torch.no_grad()
def reorder_out_proj(attn, perm):
    max_heads = attn.num_heads
    hs_max = attn.embed_dim // attn.num_heads

    w = attn.out_proj.weight.data.view(attn.embed_dim, max_heads, hs_max)
    w = w[:, perm, :]
    attn.out_proj.weight.data = w.reshape(attn.embed_dim, attn.embed_dim)
    # bias unchanged


# =========================
# Apply to One Attention
# =========================
@torch.no_grad()
def reorder_heads_shard_attention(attn):
    # attn.set_level_use(attn.max_level())

    importance = compute_head_importance_shard(attn)
    perm = torch.argsort(importance, descending=True)

    reorder_in_proj(attn, perm)
    reorder_out_proj(attn, perm)

    return perm


# =========================
# Apply to Whole Model
# =========================
@torch.no_grad()
def reorder_all_attention_heads(model, verbose=True):
    perms = {}

    for i, blk in enumerate(model.encoder.layers):
        attn = blk.self_attention
        perm = reorder_heads_shard_attention(attn)
        perms[f"layer_{i}"] = perm

        if verbose:
            print(f"[Layer {i}] Head importance order: {perm.tolist()}")

    return perms


# =========================
# Apply Random Permutation
# =========================
@torch.no_grad()
def reorder_heads_random(attn, seed=42):
    max_heads = attn.num_heads

    if seed is not None:
        torch.manual_seed(seed)

    # generate random permutation
    perm = torch.randperm(max_heads, device=attn.in_proj_weight.device)

    reorder_in_proj(attn, perm)
    reorder_out_proj(attn, perm)

    return perm


# =========================
# Apply Random to Whole Model
# =========================
@torch.no_grad()
def reorder_all_attention_heads_random(model, seed=42, verbose=True):
    perms = {}

    for i, blk in enumerate(model.encoder.layers):
        attn = blk.self_attention

        # optional: make per-layer deterministic but different
        layer_seed = None if seed is None else seed + i

        perm = reorder_heads_random(attn, seed=layer_seed)
        perms[f"layer_{i}"] = perm

        if verbose:
            print(f"[Layer {i}] Random head order: {perm.tolist()}")

    return perms

# =========================
# USAGE
# =========================
# AFTER loading pretrained weights:
#
# model = ...
# model.load_state_dict(...)
#
# reorder_all_attention_heads(model)
#
# Done ✅