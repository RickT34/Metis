import re
import os
import torch
import time
import numpy as np
from collections import defaultdict
from verl import DataProto
from verl.workers.reward_manager.registry import register
from openai import OpenAI


API_RETRY_ATTEMPTS = 2
API_RETRY_DELAY = 1

# Configure the judge model endpoint via environment variables.
# During RL training, an OpenAI-compatible server (e.g., vLLM) evaluates
# answer correctness. Set these to point to your judge model deployment.
API_KEY = os.environ.get("JUDGE_API_KEY", "EMPTY")
BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8000/v1")


SYSTEM_PROMPT = """
You are a meticulous and impartial AI evaluator. Your task is to judge whether a predicted answer is correct based on a given question and a ground truth answer.
Your response MUST be a single word: either 'CORRECT' or 'INCORRECT'. Do not provide any explanations, reasoning, or any other text.
"""

def create_user_prompt(question, answer, predict):
    return f"""
Please evaluate the following 'Prediction' based on the 'Question' and the 'Ground Truth Answer'.

**Evaluation Criteria:**
- **CORRECT:** The 'Prediction' accurately and completely answers the 'Question'. It must be semantically equivalent to the 'Ground Truth Answer'. Minor differences in wording or formatting are acceptable as long as the core meaning is identical. The prediction should not contain any factual errors, hallucinations, or refuse to answer.
- **INCORRECT:** The 'Prediction' is factually wrong, does not answer the question, provides a partial or incomplete answer, hallucinates information, or is semantically different from the 'Ground Truth Answer'.

**Data for Evaluation:**

[Question]:
{question}

[Ground Truth Answer]:
{answer}

[Prediction to Evaluate]:
{predict}

**Your judgment (ONLY 'CORRECT' or 'INCORRECT'):**
"""


def extract_answer(text: str):
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)

    patterns_to_extract = [
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\\text\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\$\$(.*?)\$\$',
        r'\$(.*?)\$',
        r'\\\[(.*?)\\\]',
        r'\\\((.*?)\\\)',
        r'\*\*(.*?)\*\*',
    ]
    for pattern in patterns_to_extract:
        while re.search(pattern, text, re.DOTALL):
            text = re.sub(pattern, r'\1', text, flags=re.DOTALL)

    text = re.sub(r'(?<=\d),(?=\d)', '', text)
    
    return text.strip().lower()


def format_reward(predict_str: str, ground_truth: str, extra_info=None) -> float:
    
    format_correct = True

    count_think_1 = predict_str.count("<reason>")
    count_think_2 = predict_str.count("</reason>")
    if count_think_1 != count_think_2:
        format_correct = False
    
    count_search_1 = predict_str.count("<tool_call>")
    count_search_2 = predict_str.count("</tool_call>")
    if count_search_1 != count_search_2:
        format_correct = False

    predict_no_think = predict_str.split('</reason>')[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        format_correct = False
    if count_answer_1 == 0 or count_answer_2 == 0:
        format_correct = False

    return 1.0 if format_correct else 0.0


def answer_reward(client: OpenAI, model_name: str, question: str, solution: str, ground_truth: str, extra_info=None) -> float:
    """Use an LLM judge to evaluate answer correctness. Returns 1.0 for correct, 0.0 otherwise."""
    if isinstance(ground_truth, list):
        scores = [answer_reward(client, model_name, question, solution, gt, extra_info) for gt in ground_truth]
        return max(scores) if scores else 0.0
    
    if not all([question, solution, ground_truth]):
        return 0.0

    user_prompt = create_user_prompt(question, ground_truth, solution)
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                timeout=30.0
            )
            response = completion.choices[0].message.content
            decision = response.strip().upper()

            if decision == 'CORRECT':
                return 1.0
            elif decision == 'INCORRECT':
                return 0.0
            else:
                print(f"AI Judge Warning: Unexpected response '{response}'. Treating as INCORRECT.")
                return 0.0
        
        except Exception as e:
            print(f"AI Judge Error: API call failed (attempt {attempt + 1}/{API_RETRY_ATTEMPTS}): {e}")
            if attempt < API_RETRY_ATTEMPTS - 1:
                time.sleep(API_RETRY_DELAY)
    
    print(f"AI Judge Error: All API retries failed for the current sample. Treating as INCORRECT.")
    return 0.0



@register("metis")
class MetisRewardManager:
    """
    HDPO dual-reward manager for Metis.

    Computes two reward signals per trajectory:
      - Accuracy reward: r_acc = 0.9 * answer_score + 0.1 * format_score
      - Tool efficiency reward: r_tool = 1/(T+1) for correct answers, 0 otherwise

    These are kept separate so that HDPO can compute independent advantages
    for each objective without gradient entanglement.
    """
    
    name = "metis"
    
    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', **kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.reward_fn_key = reward_fn_key
        
        try:
            self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
            self.model_name = self.client.models.list().data[0].id
            print(f"AI Judge Client initialized successfully. Using model: {self.model_name}")
        except Exception as e:
            print(f"FATAL: Failed to initialize AI Judge Client: {e}")
            print("Please set JUDGE_API_KEY and JUDGE_BASE_URL environment variables.")
            raise e
    
    def __call__(self, data: DataProto, return_dict: bool = False):

        # Use pre-computed rewards if available (e.g., from async reward computation)
        if "rm_scores" in data.batch.keys() and "trm_scores" in data.batch.keys():
            print(f"[MetisRewardManager] Using pre-computed rewards, skipping computation for {len(data)} samples")
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "tool_reward_tensor": data.batch["trm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"], data.batch["trm_scores"]
        elif "rm_scores" in data.batch.keys():
            print(f"[MetisRewardManager] Using pre-computed rm_scores, skipping computation for {len(data)} samples")
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            return data.batch["rm_scores"]
        
        print(f"[MetisRewardManager] No pre-computed scores found, computing rewards for {len(data)} samples...")
        batch_size = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        tool_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(batch_size):
            score = {}
            data_item = data[i]
            
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch.get('reward_model', {}).get('ground_truth', None)

            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # Accuracy reward: r_acc = 0.9 * answer_score + 0.1 * format_score
            extracted_answer = extract_answer(response_str)
            answer = extracted_answer if extracted_answer else ""
            format_score = format_reward(response_str, ground_truth)
            answer_score = answer_reward(self.client, self.model_name, prompt_str, answer, ground_truth)
            
            score['answer_score'] = answer_score
            score['format_score'] = format_score
            score['accuracy'] = 1. if answer_score > 0 else 0.
            score['score'] = 0.9 * answer_score + 0.1 * format_score
            
            # Tool efficiency reward: r_tool = 1/(T+1) for correct, 0 otherwise
            tool_interact_info = data_item.non_tensor_batch.get('tool_interact_info', None)
            if isinstance(tool_interact_info, np.ndarray):
                tool_interact_info = tool_interact_info.tolist()
            num_turns = len(tool_interact_info) if tool_interact_info else 0
            
            if answer_score > 0 and num_turns >= 0:
                tool_score = 1.0 / (num_turns + 1)
            else:
                tool_score = 0.0
            score['tool_score'] = tool_score
            score['num_turns'] = num_turns

            if score['accuracy'] > 0:
                reward_extra_info['correct_response_length'].append(valid_response_length)
            else:
                reward_extra_info['wrong_response_length'].append(valid_response_length)
            
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
                if self.num_examine == 1:
                    reward = score["accuracy"]
            else:
                if self.num_examine == 1:
                    reward = score if score > 0 else 0.0
                else:
                    reward = score
            
            reward_tensor[i, valid_response_length - 1] = reward
            tool_reward_tensor[i, valid_response_length - 1] = tool_score
            
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[Num Turns]", num_turns)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)
           
        correct_response_length_mean = np.mean(reward_extra_info['correct_response_length']) if reward_extra_info.get('correct_response_length') else 0.0
        wrong_response_length_mean = np.mean(reward_extra_info['wrong_response_length']) if reward_extra_info.get('wrong_response_length') else 0.0
        reward_extra_info['correct_response_length'] = [correct_response_length_mean] * batch_size
        reward_extra_info['wrong_response_length'] = [wrong_response_length_mean] * batch_size

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "tool_reward_tensor": tool_reward_tensor,
                "reward_extra_info": dict(sorted(reward_extra_info.items())),
            }
        else:
            return reward_tensor, tool_reward_tensor
