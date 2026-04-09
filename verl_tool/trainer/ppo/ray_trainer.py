import ray
import uuid
import torch
import os
import json
import numpy as np
from copy import deepcopy
from collections import defaultdict
from typing import Optional
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    pad_dataproto_to_divisor,
    unpad_dataproto,
    apply_kl_penalty,
    compute_response_mask,
    process_validation_metrics
) # for train and validate
from verl.trainer.ppo.ray_trainer import (
    DataProto,
) # for init
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.utils.debug import marked_timer
from verl.trainer.ppo.core_algos import agg_loss
from tqdm import tqdm
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from pprint import pprint

##############################################################################
#### Replace the original classes/functions with verl-tool customized ones ####
import verl.experimental.agent_loop
from verl_tool.agent_loop import AgentLoopManager
import verl.trainer.ppo.ray_trainer
from .reward import compute_reward, compute_reward_async
from verl_tool.workers.rollout.vllm_rollout.vllm_async_server import VerlToolvLLMHttpServer
import verl.workers.rollout.vllm_rollout.vllm_async_server
from .metric_util import compute_data_metrics, process_validation_metrics
verl.experimental.agent_loop.AgentLoopManager = AgentLoopManager
verl.trainer.ppo.ray_trainer.compute_reward = compute_reward
verl.trainer.ppo.ray_trainer.compute_reward_async = compute_reward_async
verl.trainer.ppo.ray_trainer.compute_data_metrics = compute_data_metrics
verl.trainer.ppo.ray_trainer.process_validation_metrics = process_validation_metrics
verl.workers.rollout.vllm_rollout.vllm_async_server.vLLMHttpServer = VerlToolvLLMHttpServer

# Replace compute_advantage with HDPO-enabled version (supports dual advantages)
import verl.trainer.ppo.core_algos
import verl_tool.trainer.ppo.hdpo_algos
from .hdpo_algos import AdvantageEstimator
from .hdpo_algos import hdpo_compute_advantage as compute_advantage
verl.trainer.ppo.core_algos.AdvantageEstimator = AdvantageEstimator
verl.trainer.ppo.ray_trainer.compute_advantage = compute_advantage
### ======================================================================= ###

from qwen_vl_utils import vision_process
from verl_tool.agent_loop.vision_process import smart_resize
vision_process.smart_resize=smart_resize
##############################################################################


class AgentRayPPOTrainer(RayPPOTrainer):
    
    def _print_grpo_statistics(self, batch: DataProto):
        """Print GRPO group statistics before computing advantages."""
        if not hasattr(batch, 'non_tensor_batch') or 'accuracy' not in batch.non_tensor_batch:
            return
        
        batch_size = len(batch)
        accuracies = batch.non_tensor_batch.get('accuracy', None)
        format_scores = batch.non_tensor_batch.get('format_score', None)
        final_scores = batch.non_tensor_batch.get('score', None)
        
        if accuracies is None or final_scores is None:
            return
        
        # Try to find uid field for GRPO grouping
        possible_index_fields = ['uid', 'index', 'prompt_id', 'data_id', 'unique_id']
        indices = None
        for field in possible_index_fields:
            if field in batch.non_tensor_batch:
                indices = batch.non_tensor_batch[field]
                break
        
        if indices is None:
            return
        
        unique_indices, counts = np.unique(indices, return_counts=True)
        if len(set(counts)) != 1:
            return  # Not uniform GRPO groups
        
        samples_per_prompt = counts[0]
        if samples_per_prompt <= 1:
            return  # Not GRPO
        
        num_groups = len(unique_indices)
        
        print("=" * 80)
        print(f"[GRPO] Total samples: {batch_size} | Groups: {num_groups} | Samples/group: {samples_per_prompt}")
        
        # Group-wise statistics
        group_max_scores = []
        group_min_scores = []
        group_all_correct = []
        group_all_wrong = []
        group_partial_correct = []
        group_zero_score = []
        
        for idx in unique_indices:
            mask = indices == idx
            group_scores = np.array(final_scores)[mask]
            group_accuracies = np.array(accuracies)[mask]
            
            group_max_scores.append(group_scores.max())
            group_min_scores.append(group_scores.min())
            
            # Answer correctness
            num_correct_in_group = (group_accuracies > 0).sum()
            if num_correct_in_group == samples_per_prompt:
                group_all_correct.append(1)
                group_all_wrong.append(0)
                group_partial_correct.append(0)
            elif num_correct_in_group == 0:
                group_all_correct.append(0)
                group_all_wrong.append(1)
                group_partial_correct.append(0)
            else:
                group_all_correct.append(0)
                group_all_wrong.append(0)
                group_partial_correct.append(1)
            
            # Check if all samples in group got 0 score
            if (group_scores == 0).all():
                group_zero_score.append(1)
            else:
                group_zero_score.append(0)
        
        # Score statistics
        print(f"[GRPO] Score - Max: {np.mean(group_max_scores):.3f} | Min: {np.mean(group_min_scores):.3f} | Gap: {np.mean(np.array(group_max_scores) - np.array(group_min_scores)):.3f}")
        
        # Group correctness distribution
        num_all_correct = sum(group_all_correct)
        num_all_wrong = sum(group_all_wrong)
        num_partial = sum(group_partial_correct)
        num_zero_score = sum(group_zero_score)
        
        print(f"[GRPO] Answer - All Correct: {num_all_correct}/{num_groups} ({num_all_correct/num_groups*100:.1f}%) | "
              f"All Wrong: {num_all_wrong}/{num_groups} ({num_all_wrong/num_groups*100:.1f}%) | "
              f"Partial: {num_partial}/{num_groups} ({num_partial/num_groups*100:.1f}%)")
        print(f"[GRPO] Score  - Zero Score Groups: {num_zero_score}/{num_groups} ({num_zero_score/num_groups*100:.1f}%) (all samples got 0 score)")
        print("=" * 80)
    
    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_attention_mask = test_output_gen_batch.batch["attention_mask"][:, test_output_gen_batch.batch["prompts"].shape[1]:]
            output_texts = [self.tokenizer.decode(ids[output_attention_mask[i]==1], skip_special_tokens=False) for i, ids in enumerate(output_ids)]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    
            tool_interact_info = test_batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if tool_interact_info:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                if "tool_interact_info" not in reward_extra_infos_dict:
                    reward_extra_infos_dict["tool_interact_info"] = []
                if "traj_stop_reason" not in reward_extra_infos_dict:
                    reward_extra_infos_dict["traj_stop_reason"] = []
                reward_extra_infos_dict["tool_interact_info"].extend(tool_interact_info)
                reward_extra_infos_dict["traj_stop_reason"].extend(
                    test_batch.non_tensor_batch.get("traj_stop_reason", [None] * reward_tensor.shape[0])
                )
                reward_extra_infos_dict["verl_tool_metrics"].extend(
                    test_batch.non_tensor_batch.get("verl_tool_metrics", [None] * reward_tensor.shape[0])
                )

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
        if "tool_interact_info" in reward_extra_infos_dict:
            # remove if after dump
            reward_extra_infos_dict.pop("tool_interact_info")
        if "traj_stop_reason" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("traj_stop_reason")
        if "verl_tool_metrics" in reward_extra_infos_dict:
            reward_extra_infos_dict.pop("verl_tool_metrics")

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs_attention_masks = batch.batch['attention_mask'][:, :batch.batch['prompts'].shape[1]]
            outputs_attention_masks = batch.batch['attention_mask'][:, batch.batch['prompts'].shape[1]:]
            inputs = [self.tokenizer.decode(batch.batch["prompts"][i][inputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["prompts"].shape[0])]
            outputs = [self.tokenizer.decode(batch.batch["responses"][i][outputs_attention_masks[i]==1], skip_special_tokens=False) for i in range(batch.batch["responses"].shape[0])]
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )
            
            tool_interact_info = batch.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            if tool_interact_info:
                for tool_interact in tool_interact_info:
                    if "image" in tool_interact:
                        if isinstance(tool_interact['image'], list):
                            tool_interact['image'] = [x[:50] for x in tool_interact['image']]  # crop the image to first 50 characters
                        elif isinstance(tool_interact['image'], str):
                            tool_interact['image'] = tool_interact['image'][:50] # for debug
                reward_extra_infos_to_dump.update({
                    "tool_interact_info": tool_interact_info,
                    "traj_stop_reason": batch.non_tensor_batch.get("traj_stop_reason", None),
                    "verl_tool_metrics": batch.non_tensor_batch.get("verl_tool_metrics", None),
                })

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    
                    # Check if rm_scores exists from agent_loop
                    has_rm_scores = "rm_scores" in batch.batch.keys()
                    has_trm_scores = "trm_scores" in batch.batch.keys()
                    print(f"\n{'='*80}")
                    print(f"[Trainer] Step {self.global_steps} - Before reward computation")
                    print(f"  Batch size: {len(batch)}")
                    print(f"  Has rm_scores: {has_rm_scores}")
                    print(f"  Has trm_scores: {has_trm_scores}")
                    if has_rm_scores:
                        print(f"  rm_scores shape: {batch.batch['rm_scores'].shape}")
                    print(f"{'='*80}\n")

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Only compute rewards if they don't already exist from agent_loop
                        need_compute_reward = (has_rm_scores and not has_trm_scores) or (not has_rm_scores)
                        
                        if need_compute_reward:
                            print(f"[Trainer] Computing rewards (rm_scores exists: {has_rm_scores}, trm_scores exists: {has_trm_scores})")
                            if self.config.reward_model.launch_reward_fn_async:
                                # Use compute_reward_async (creates new process each time, memory friendly)
                                future_reward = compute_reward_async.remote(
                                    data=batch, config=self.config, tokenizer=self.tokenizer
                                )
                            else:
                                reward_tensor, tool_reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                        else:
                            print(f"[Trainer] ✓ Skipping reward computation - using pre-computed rewards from agent_loop")
                            future_reward = None

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)
                    
                    if "rollout_log_probs" in batch.batch.keys():
                        # TODO: we may want to add diff of probs too.
                        from verl.utils.debug.metrics import calculate_debug_metrics
                        
                        metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    print(f"[Trainer] Starting advantage computation...")
                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            if future_reward is not None:
                                print(f"[Trainer] Waiting for async reward computation...")
                                reward_tensor, tool_reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                print(f"[Trainer] Got reward results")
                            else:
                                print(f"[Trainer] Using pre-computed rewards directly from batch")
                                reward_tensor = batch.batch["rm_scores"]
                                tool_reward_tensor = batch.batch.get("trm_scores", torch.zeros_like(reward_tensor))
                                # Create reward_extra_infos_dict from batch non_tensor_batch
                                reward_extra_infos_dict = {}
                        batch.batch["token_level_scores"] = reward_tensor
                        batch.batch["tool_reward_tensor"] = tool_reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout importance sampling weights centrally (once per batch)
                        # This corrects for mismatch between rollout policy and training policy
                        # Also computes mismatch metrics (KL, PPL, etc.)
                        batch, is_metrics = self.compute_rollout_importance_weights_and_add_to_batch(batch)
                        # IS and mismatch metrics already have mismatch/ prefix
                        metrics.update(is_metrics)

                        # === GRPO GROUP STATISTICS (before computing advantage) ===
                        self._print_grpo_statistics(batch)
                        
                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        
                        # Removed: Detailed HDPO advantage verification (system confirmed working)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            # Optimization: Reduce tensor data before Ray serialization
                            # Only select necessary fields to minimize data transfer
                            select_critic_batch = getattr(self.config.actor_rollout_ref.rollout, "select_critic_batch_before_update", True)
                            
                            if select_critic_batch:
                                # Select only necessary fields for critic update
                                critic_select_keys = ["input_ids", "responses", "response_mask", "attention_mask", "position_ids", "values", "returns"]
                                has_multi_modal_inputs = "multi_modal_inputs" in batch.non_tensor_batch.keys()
                                non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
                                
                                critic_batch = batch.select(batch_keys=critic_select_keys, non_tensor_batch_keys=non_tensor_select_keys)
                                # Preserve meta_info from original batch
                                critic_batch.meta_info.update(batch.meta_info)
                                critic_output = self.critic_wg.update_critic(critic_batch)
                            else:
                                critic_output = self.critic_wg.update_critic(batch)
                        
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            
                            # Optimization: Reduce tensor data before Ray serialization
                            # Only select fields needed by update_policy to minimize serialization overhead (reduces by ~30-50%)
                            select_actor_batch = getattr(self.config.actor_rollout_ref.rollout, "select_actor_batch_before_update", True)
                            
                            if select_actor_batch:
                                # Select only necessary fields for actor update
                                select_keys = [
                                    "responses",
                                    "response_mask",
                                    "input_ids",
                                    "attention_mask",
                                    "position_ids",
                                    "old_log_probs",
                                    "advantages",
                                ]
                                # Safely check for optional fields
                                if getattr(self.config.actor_rollout_ref.actor, "use_kl_loss", False):
                                    select_keys.append("ref_log_prob")
                                if "rollout_is_weights" in batch.batch.keys():
                                    select_keys.append("rollout_is_weights")
                                if "rollout_log_probs" in batch.batch.keys():
                                    select_keys.append("rollout_log_probs")
                                # HDPO-specific fields
                                if "tool_advantages" in batch.batch.keys():
                                    select_keys.append("tool_advantages")
                                
                                has_multi_modal_inputs = "multi_modal_inputs" in batch.non_tensor_batch.keys()
                                non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
                                
                                actor_batch = batch.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
                                # Preserve meta_info from original batch
                                actor_batch.meta_info.update(batch.meta_info)
                                
                                # Log optimization effect (only on first step)
                                if self.global_steps == 1:
                                    original_fields = list(batch.batch.keys())
                                    selected_fields = list(actor_batch.batch.keys())
                                    removed_fields = [f for f in original_fields if f not in selected_fields]
                                    print(f"[Optimization] update_actor field reduction:")
                                    print(f"  Original: {len(original_fields)} fields - {original_fields}")
                                    print(f"  Selected: {len(selected_fields)} fields - {selected_fields}")
                                    print(f"  Removed: {len(removed_fields)} fields - {removed_fields}")
                                
                                actor_output = self.actor_rollout_wg.update_actor(actor_batch)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(batch)
                        
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)
                        
                        # Print HDPO metrics if using HDPO
                        if self.config.algorithm.adv_estimator == "hdpo":
                            acc_loss = actor_output_metrics.get('actor/pg_loss/acc', 0)
                            tool_loss = actor_output_metrics.get('actor/pg_loss/tool', 0)
                            acc_adv_nonzero = actor_output_metrics.get('actor/advantage/acc_nonzero_ratio', 0)
                            tool_adv_nonzero = actor_output_metrics.get('actor/advantage/tool_nonzero_ratio', 0)
                            
                            print(f"[HDPO] Step {self.global_steps}: acc_loss={acc_loss:.4f}, tool_loss={tool_loss:.4f}, "
                                  f"acc_adv_nonzero={acc_adv_nonzero:.2%}, tool_adv_nonzero={tool_adv_nonzero:.2%}")
                    else:
                        # Still in critic warmup phase
                        print(f"[Trainer] Step {self.global_steps}: Skipping actor update (critic_warmup: {self.config.trainer.critic_warmup})")

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)
                    
                    # Print timing summary for GPU efficiency analysis
                    print(f"\n{'='*80}")
                    print(f"[Trainer] Step {self.global_steps} - Timing Summary:")
                    print(f"  gen:              {timing_raw.get('gen', 0):.2f}s")
                    print(f"  reward:           {timing_raw.get('reward', 0):.2f}s")
                    print(f"  old_log_prob:     {timing_raw.get('old_log_prob', 0):.2f}s")
                    print(f"  adv:              {timing_raw.get('adv', 0):.2f}s")
                    print(f"  update_actor:     {timing_raw.get('update_actor', 0):.2f}s")
                    print(f"  dump_rollout:     {timing_raw.get('dump_rollout_generations', 0):.2f}s")
                    total_time = sum([timing_raw.get(k, 0) for k in ['gen', 'reward', 'old_log_prob', 'adv', 'update_actor']])
                    print(f"  TOTAL:            {total_time:.2f}s")
                    
                    # GPU efficiency metrics
                    batch_size = len(batch)
                    total_tokens = torch.sum(batch.batch["attention_mask"]).item()
                    print(f"\n[GPU Efficiency]")
                    print(f"  Batch size:       {batch_size}")
                    print(f"  Total tokens:     {total_tokens:,}")
                    print(f"  Tokens/sec:       {total_tokens/total_time:,.0f}")
                    old_log_prob_time = timing_raw.get('old_log_prob', 1)
                    update_actor_time = timing_raw.get('update_actor', 1)
                    print(f"  old_log_prob throughput: {total_tokens/old_log_prob_time:,.0f} tokens/s")
                    print(f"  update_actor throughput: {total_tokens/update_actor_time:,.0f} tokens/s")
                    print(f"{'='*80}\n")

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)