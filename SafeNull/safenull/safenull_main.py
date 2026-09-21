"""
SafeNull: Preserving Safety Alignment in Continual Knowledge Editing via Dual Null-Space Constraints.
Main core implementation based on locate-then-edit with dual null-space projection.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast
from AlphaEdit.compute_ks import compute_ks
from AlphaEdit.compute_z import compute_z, get_module_input_output_at_words

from .safety_extractor import get_safety_covariance

CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def get_safe_project(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    hparams: Any,
    alpha: float = 0.5,
    force_recompute: bool = False,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Constructs the Dual Null-Space Projection Matrix P_safe:
    1. Retrieves/computes commonsense covariance C_wiki = E[k_wiki k_wiki^T]
    2. Computes safety anchor covariance C_safe = E[k_safe k_safe^T]
    3. Builds joint manifold C_joint = C_wiki + alpha * C_safe
    4. Computes SVD on C_joint and extracts null-space eigenvectors:
       P_safe = U_null @ U_null^T
    """
    from AlphaEdit.AlphaEdit_main import get_cov

    # 1. Commonsense covariance from Wikipedia
    c_wiki = get_cov(
        model,
        tok,
        hparams.rewrite_module_tmp.format(layer),
        hparams.mom2_dataset,
        hparams.mom2_n_samples if not force_recompute else hparams.mom2_n_samples // 10,
        hparams.mom2_dtype,
        force_recompute=force_recompute,
    ).cpu()

    # 2. Safety anchor covariance
    c_safe = get_safety_covariance(
        model,
        tok,
        layer,
        hparams,
        device=device
    ).cpu()

    # 3. Joint preservation manifold
    # Normalize scale to ensure balanced preservation
    norm_wiki = torch.linalg.norm(c_wiki) + 1e-8
    norm_safe = torch.linalg.norm(c_safe) + 1e-8
    c_joint = (c_wiki / norm_wiki) + alpha * (c_safe / norm_safe)

    # 4. SVD on joint covariance
    U, S, _ = torch.linalg.svd(c_joint, full_matrices=False)
    threshold = getattr(hparams, "nullspace_threshold", 2e-2)

    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    print(f"[SafeNull Layer {layer}] Joint singular values < {threshold}: {len(small_singular_indices)} / {S.shape[0]}")

    if len(small_singular_indices) == 0:
        # Fallback to smallest 10% singular vectors if threshold is too strict
        k = max(1, int(S.shape[0] * 0.1))
        small_singular_indices = torch.argsort(S)[:k]
        print(f"[SafeNull Layer {layer}] Warning: Fallback to {len(small_singular_indices)} smallest vectors")

    U_null = U[:, small_singular_indices]
    P_safe = U_null @ U_null.T
    return P_safe


def apply_safenull_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: Any,
    cache_template: Optional[str] = None,
    cache_c: Optional[torch.Tensor] = None,
    P_safe: Optional[torch.Tensor] = None,
    alpha: float = 0.5,
    return_orig_weights_device: Optional[str] = None,
    mock: bool = False
) -> Tuple[AutoModelForCausalLM, torch.Tensor]:
    """
    Executes the SafeNull update algorithm:
    Solves parameter perturbation constrained by P_safe:
    Delta W = solve( P_safe @ (K K^T + cache_c) + lambda * I, P_safe @ K @ resid^T )
    """
    if mock:
        print("[SafeNull Mock] Running in mock mode, skipping full tensor updates.")
        return model, cache_c

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        cur_z = compute_z(
            model,
            tok,
            request,
            hparams,
            z_layer,
            context_templates,
        )
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, layer in enumerate(hparams.layers):
        print(f"[SafeNull] Processing LAYER {layer}")
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T

        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)

        # Dual Null-space projection operator
        P_curr = P_safe[i, :, :].to(device)
        cache_c_curr = cache_c[i, :, :].to(device) if cache_c is not None else torch.zeros_like(P_curr)

        lhs = P_curr @ (layer_ks @ layer_ks.T + cache_c_curr) + hparams.L2 * torch.eye(
            layer_ks.shape[0], dtype=torch.float, device=device
        )
        rhs = P_curr @ layer_ks @ resid.T

        upd_matrix = torch.linalg.solve(lhs, rhs)

        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.to(weights[weight_name].device)

        del layer_ks, cur_zs, targets, upd_matrix, P_curr, cache_c_curr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Update sequential cache
    if cache_c is not None:
        for i, layer in enumerate(hparams.layers):
            layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
            cache_c[i, :, :] += (layer_ks.cpu() @ layer_ks.cpu().T)

    return model, cache_c


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(f"Matrix shape {matrix.shape} cannot match target weight shape {shape}")


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE
    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=2,
                    max_out_len=10,
                )
            ]
            for length, n_gen in [(10, 2)]
        ]
    return CONTEXT_TEMPLATES_CACHE
