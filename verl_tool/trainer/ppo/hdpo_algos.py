import torch
import numpy as np
from enum import Enum
from collections import defaultdict
from typing import Optional

from verl.protocol import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.core_algos import register_adv_est, compute_grpo_outcome_advantage

from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.trainer.ppo.ray_trainer import compute_advantage as original_compute_advantage

class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    HDPO = "hdpo"
    MT_GRPO = "mt_grpo"  # Multi-Turn GRPO with turn-level credit assignment
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"

# ======================== HDPO =================================#

def compute_grpo_conditional_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Conditional GRPO: computes advantage only among qualifying samples (score > 0)
    within each group. Qualifying = correct answer AND used tools (num_turns >= 1).

    Key behavior:
    - Groups with >=2 qualifying samples: standard within-group GRPO comparison.
    - Groups with 0 or 1 qualifying sample: advantage = 0 (no valid comparison).
      Unlike standard GRPO (which uses mean=0 for single samples), we cannot use
      mean=0 here because that would blindly reinforce any tool count. And cross-group
      baselines are not meaningful since different questions have different tool needs.
    - To improve data efficiency when qualifying samples are sparse, increase n
      (rollouts per prompt) rather than using a cross-group baseline.
    """
    scores = token_level_rewards.sum(dim=-1)
    
    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        
        # Pass 1: Group only qualifying samples (score > 0 means correct AND used tools)
        for i in range(bsz):
            if scores[i] > 0.0:
                id2score[index[i]].append(scores[i])
        
        # Pass 2: Compute mean and std for each group
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # Only 1 qualifying sample: no within-group comparison possible.
                # Set mean = score itself so advantage = 0.
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = scores_tensor[0]
                id2std[idx] = torch.tensor(1.0)
            else:  # len(id2score[idx]) >= 2
                # Multiple qualifying samples: standard GRPO comparison
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)

        # Pass 3: Compute advantage
        advantages = torch.zeros_like(scores)
        for i in range(bsz):
            if scores[i] > 0.0:
                idx = index[i]
                adv_val = scores[i] - id2mean[idx]
                if norm_adv_by_std_in_grpo:
                    adv_val = adv_val / (id2std[idx] + epsilon)
                advantages[i] = adv_val
        
        advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages


@register_adv_est(AdvantageEstimator.HDPO)
def compute_hdpo_advantage(
    token_level_rewards: torch.Tensor,
    tool_reward_tensor: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
):
    # 1. Accuracy reward: use standard GRPO advantage.
    acc_advantages, acc_returns = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    
    # 2. Tool reward: use conditional GRPO advantage.
    tool_advantages, tool_returns = compute_grpo_conditional_advantage(
        token_level_rewards=tool_reward_tensor,
        response_mask=response_mask,
        index=index,
        epsilon=epsilon,
        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        config=config,
    )
    
    # 3. Return two independent, correctly computed advantages
    return {
        'acc_advantages': acc_advantages,
        'tool_advantages': tool_advantages,
    }, acc_returns

# ======================== HDPO =================================#

def hdpo_compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """
    A wrapper function that handles the 'hdpo' advantage estimator specially.
    For all other estimators, it delegates the call to the original function.
    """
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    if adv_estimator == AdvantageEstimator.HDPO:
        hdpo_calculation_mask = data.batch["response_mask"]

        adv_results, returns = compute_hdpo_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            tool_reward_tensor=data.batch['tool_reward_tensor'],
            response_mask=hdpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            epsilon=1e-6,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=config,
        )

        data.batch['advantages'] = adv_results['acc_advantages']
        data.batch['tool_advantages'] = adv_results['tool_advantages']
        data.batch['returns'] = returns
        
        return data
    else:
        return original_compute_advantage(
            data, adv_estimator, gamma, lam, num_repeat, norm_adv_by_std_in_grpo, config
        )
