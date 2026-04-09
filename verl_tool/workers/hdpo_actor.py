"""
HDPO-enabled Actor implementation with dual-loss policy update.

This module provides a clean, inheritance-based implementation of HDPO
instead of relying on monkey patching.
"""
import torch
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.py_functional import append_to_dict
from verl.utils.device import get_device_id
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.profiler import GPUMemoryLogger
import logging
import os

logger = logging.getLogger(__name__)


class HDPODataParallelPPOActor(DataParallelPPOActor):
    """
    HDPO (Hybrid Advantage with Policy Optimization) Actor.
    
    Extends DataParallelPPOActor to support dual-loss training:
    - Accuracy loss: optimizes answer correctness
    - Tool efficiency loss: optimizes tool usage (only for correct answers)
    
    The final policy loss is: pg_loss = w_acc * loss_acc + w_tool * loss_tool
    """
    
    def get_actor_info(self):
        """Return actor information for debugging/verification."""
        return {
            "actor_class": self.__class__.__name__,
            "actor_type": "HDPO",
            "dual_loss_enabled": True,
            "w_acc": self.config.get("w_acc", 1.0),
            "w_tool": self.config.get("w_tool", 0.0),
        }
    
    @GPUMemoryLogger(role="hdpo_actor", logger=logger)
    def update_policy(self, data: DataProto):
        """
        Update policy using HDPO dual-loss mechanism.
        
        Args:
            data: Training batch containing:
                - advantages: accuracy advantages
                - tool_advantages: tool efficiency advantages (from conditional GRPO)
        
        Returns:
            dict: Training metrics including both accuracy and tool losses
        """
        import time
        t_start = time.time()
        timing_breakdown = {}
        
        # Verify HDPO-specific data is present
        has_tool_advantages = "tool_advantages" in data.batch.keys()
        
        if not has_tool_advantages:
            print(f"[HDPO Actor] ⚠️  WARNING: tool_advantages not found in batch! HDPO will not work correctly.", flush=True)
        
        # Make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",      # Accuracy advantages
            "tool_advantages", # Tool efficiency advantages (HDPO-specific)
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    t_micro_start = time.time()
                    
                    t_prep_start = time.time()
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    
                    # HDPO-specific: extract both advantage types
                    advantages = model_inputs["advantages"]
                    tool_advantages = model_inputs["tool_advantages"]
                    
                    # Skip detailed micro-batch logging to reduce noise
                    
                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation
                    t_prep_end = time.time()
                    
                    # Forward pass
                    t_forward_start = time.time()
                    calculate_entropy = entropy_coeff != 0
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    t_forward_end = time.time()
                    
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]
                    
                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # ===== HDPO DUAL LOSS =====
                    t_loss_start = time.time()
                    # Compute accuracy loss (uses all samples)
                    loss_acc, acc_clipfrac, acc_kl, acc_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )

                    # Compute tool efficiency loss (L_tool).
                    # Normalized only over qualifying samples (correct answer + used tools +
                    # >=2 qualifying in group). Combined as: L = w_acc·L_acc + w_tool·L_tool
                    tool_has_signal = (tool_advantages != 0).any(dim=-1, keepdim=True)  # [bsz, 1]
                    tool_response_mask = response_mask * tool_has_signal.float()
                    
                    # Qualifying ratio for monitoring (not used in loss computation)
                    rho_val = tool_has_signal.float().mean().item()
                    
                    # Fallback: if no qualifying samples, use original mask (loss=0 anyway)
                    if tool_response_mask.sum() == 0:
                        tool_response_mask = response_mask
                    
                    loss_tool, tool_clipfrac, tool_kl, tool_clipfrac_lower = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=tool_advantages,
                        response_mask=tool_response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )

                    # Weighted combination: L = w_acc·L_acc + w_tool·L_tool
                    w_acc = self.config.get("w_acc", 1.0)
                    w_tool = self.config.get("w_tool", 0.0)
                    pg_loss = w_acc * loss_acc + w_tool * loss_tool
                    
                    # Compute metrics for monitoring
                    with torch.no_grad():
                        loss_acc_val = loss_acc.detach().item()
                        loss_tool_val = loss_tool.detach().item()
                        # Effective gradient ratio: (w_tool * loss_tool) / (w_acc * loss_acc)
                        if abs(loss_acc_val) > 1e-8:
                            effective_tool_ratio = (w_tool * loss_tool_val) / (w_acc * loss_acc_val + 1e-8)
                        else:
                            effective_tool_ratio = 0.0
                    t_loss_end = time.time()
                    
                    # Calculate key metrics for HDPO monitoring
                    acc_adv_nonzero_ratio = (advantages != 0).float().mean().item()
                    tool_adv_nonzero_ratio = (tool_advantages != 0).float().mean().item()
                    
                    # Calculate advantage statistics
                    acc_adv_mean = advantages.mean().item()
                    acc_adv_std = advantages.std().item()
                    tool_adv_mean = tool_advantages.mean().item()
                    tool_adv_std = tool_advantages.std().item()
                    
                    # Only calculate nonzero stats if there are nonzero values
                    if acc_adv_nonzero_ratio > 0:
                        acc_adv_nonzero_values = advantages[advantages != 0]
                        acc_adv_mean_nonzero = acc_adv_nonzero_values.mean().item()
                        acc_adv_std_nonzero = acc_adv_nonzero_values.std().item() if len(acc_adv_nonzero_values) > 1 else 0.0
                    else:
                        acc_adv_mean_nonzero = 0.0
                        acc_adv_std_nonzero = 0.0
                    
                    if tool_adv_nonzero_ratio > 0:
                        tool_adv_nonzero_values = tool_advantages[tool_advantages != 0]
                        tool_adv_mean_nonzero = tool_adv_nonzero_values.mean().item()
                        tool_adv_std_nonzero = tool_adv_nonzero_values.std().item() if len(tool_adv_nonzero_values) > 1 else 0.0
                    else:
                        tool_adv_mean_nonzero = 0.0
                        tool_adv_std_nonzero = 0.0

                    # Add entropy loss
                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = pg_loss - entropy_loss * entropy_coeff
                    else:
                        policy_loss = pg_loss
                    
                    # Add KL loss if enabled
                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef
                    
                    # Backprop
                    t_backward_start = time.time()
                    if self.config.use_dynamic_bsz:
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    loss.backward()
                    t_backward_end = time.time()
                    
                    t_micro_end = time.time()
                    
                    # Accumulate timing
                    timing_breakdown.setdefault('prep', []).append(t_prep_end - t_prep_start)
                    timing_breakdown.setdefault('forward', []).append(t_forward_end - t_forward_start)
                    timing_breakdown.setdefault('loss', []).append(t_loss_end - t_loss_start)
                    timing_breakdown.setdefault('backward', []).append(t_backward_end - t_backward_start)
                    timing_breakdown.setdefault('micro_total', []).append(t_micro_end - t_micro_start)

                    # Record HDPO metrics (these will be logged to wandb)
                    micro_batch_metrics.update({
                        # === Loss Components ===
                        "actor/pg_loss": pg_loss.detach().item() * loss_scale_factor,
                        "actor/pg_loss/acc": loss_acc.detach().item() * loss_scale_factor,
                        "actor/pg_loss/tool": loss_tool.detach().item() * loss_scale_factor,
                        
                        # === Loss Weights & Ratio ===
                        "actor/weights/w_acc": w_acc,
                        "actor/weights/w_tool": w_tool,
                        "actor/qualifying_ratio_rho": rho_val,  # ρ: fraction of qualifying samples
                        "actor/effective_tool_ratio": effective_tool_ratio,  # (w_tool*ρ*loss_tool)/(w_acc*loss_acc)
                        
                        # === Clip Fractions ===
                        "actor/pg_clipfrac/acc": acc_clipfrac.detach().item(),
                        "actor/pg_clipfrac/tool": tool_clipfrac.detach().item(),
                        
                        # === Accuracy Advantage Stats ===
                        "actor/advantage/acc_mean": acc_adv_mean,
                        "actor/advantage/acc_std": acc_adv_std,
                        "actor/advantage/acc_nonzero_ratio": acc_adv_nonzero_ratio,
                        "actor/advantage/acc_mean_nonzero": acc_adv_mean_nonzero,
                        "actor/advantage/acc_std_nonzero": acc_adv_std_nonzero,
                        
                        # === Tool Advantage Stats ===
                        "actor/advantage/tool_mean": tool_adv_mean,
                        "actor/advantage/tool_std": tool_adv_std,
                        "actor/advantage/tool_nonzero_ratio": tool_adv_nonzero_ratio,
                        "actor/advantage/tool_mean_nonzero": tool_adv_mean_nonzero,
                        "actor/advantage/tool_std_nonzero": tool_adv_std_nonzero,
                    })
                    append_to_dict(metrics, micro_batch_metrics)

                t_optim_start = time.time()
                grad_norm = self._optimizer_step()
                t_optim_end = time.time()
                timing_breakdown.setdefault('optim_step', []).append(t_optim_end - t_optim_start)
                
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
                
        self.actor_optimizer.zero_grad()
        
        t_end = time.time()
        total_time = t_end - t_start
        
        # Print timing breakdown
        if timing_breakdown:
            avg_prep = sum(timing_breakdown['prep']) / len(timing_breakdown['prep'])
            avg_forward = sum(timing_breakdown['forward']) / len(timing_breakdown['forward'])
            avg_loss = sum(timing_breakdown['loss']) / len(timing_breakdown['loss'])
            avg_backward = sum(timing_breakdown['backward']) / len(timing_breakdown['backward'])
            avg_optim = sum(timing_breakdown['optim_step']) / len(timing_breakdown['optim_step']) if 'optim_step' in timing_breakdown else 0
            
            avg_acc_loss = sum(metrics["actor/pg_loss/acc"]) / len(metrics["actor/pg_loss/acc"]) if "actor/pg_loss/acc" in metrics else 0
            avg_tool_loss = sum(metrics["actor/pg_loss/tool"]) / len(metrics["actor/pg_loss/tool"]) if "actor/pg_loss/tool" in metrics else 0
            avg_rho = sum(metrics["actor/qualifying_ratio_rho"]) / len(metrics["actor/qualifying_ratio_rho"]) if "actor/qualifying_ratio_rho" in metrics else 0
            print(f"[HDPO] Forward={avg_forward:.2f}s, Backward={avg_backward:.2f}s, Total={total_time:.2f}s, acc_loss={avg_acc_loss:.4f}, tool_loss={avg_tool_loss:.4f}, ρ={avg_rho:.3f}", flush=True)
        
        return metrics
