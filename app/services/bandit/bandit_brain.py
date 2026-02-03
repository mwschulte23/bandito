import asyncio
import numpy as np
from typing import Literal
from app.schemas.bandit import BanditStateRead
from app.services.bandit.utils.feature_prep import FeatureTransformer
from app.services.bandit.utils.data_helper import get_full_bandit, add_event, get_event, update_event, update_state
from app.services.bandit.utils.rewards import calculate_reward
# from app.schemas.bandit_actions import RewardInput


async def pull_arm(
    bandit_id: int,
    user_id: int,
    user_query: str,
    gamma: float = 1.0,
    beta: float = 1.0
):
    bandit = await get_full_bandit(bandit_id, user_id)

    user_query
    context = {
        "hour_of_day": np.random.randint(0, high=24), 
        "is_weekend": 0,
        # "query_complexity": 
    }
    mapper = FeatureTransformer(bandit.arms)
    ts_theta = bandit.state.theta_hat + beta * (bandit.state.cholesky_l_inv @ np.random.standard_normal(bandit.state.dimensions))

    best_arm_id, max_score, best_features = None, -float('inf'), None # init 
    for arm in bandit.arms:
        if not arm.is_active:
            continue
        features = mapper.transform_to_vector(arm, context)
        score = features @ ts_theta
        if score > max_score:
            max_score, best_arm_id, chosen_model, chosen_prompt = score, arm.id, arm.model_name, arm.system_prompt
    
    event_id = await add_event(bandit_id, best_arm_id, max_score, context, user_query)
    return {
        'event_id': event_id,
        'bandit_arm': {'model': chosen_model, 'system_prompt': chosen_prompt},
        'context': context
    }


async def update_on_reward(
    bandit_id: int, user_id: int, event_id: int,
    llm_output: str | dict,
    reward: float,
    cost: float = None,
    latency: float = None,
    is_human_reward: bool = False
):
    bandit = await get_full_bandit(bandit_id, user_id)
    target_event = await get_event(event_id)
    chosen_arm = [arm for arm in bandit.arms if arm.id == target_event.arm_id][0]
    state = bandit.state

    mapper = FeatureTransformer(bandit.arms)
    features = mapper.transform_to_vector(chosen_arm, target_event.context)

    if is_human_reward:
        if target_event.immediate_reward is not None:
            # TODO: figure out storing raw reward vs adj vs residual 
            residual_reward = calculate_reward(reward, cost, latency) - target_event.immediate_reward
            target_event.human_reward = reward
            
            state.b = features * residual_reward
    else:
        adj_reward = calculate_reward(reward, cost, latency)
        target_event.immediate_reward = adj_reward
        state.a += np.outer(features, features)
        state.b += features * reward
    
    A_inv = np.linalg.inv(state.a)
    state.theta_hat = A_inv @ state.b
    state.cholesky_l_inv = np.linalg.cholesky(A_inv)

    if isinstance(llm_output, str):
        llm_output = {'response': llm_output}
    
    target_event.llm_output = llm_output
    target_event.cost = cost
    target_event.latency = latency

    await update_event(target_event.id, target_event)
    await update_state(state.id, state)



# TEMP
async def main(bandit_id: int, user_id: int, user_q: str):
    import requests
    import json
    import time
    
    pull = await pull_arm(bandit_id, user_id, user_q)

    start = time.perf_counter() # TODO: figure out how to measure this...without forcing user??
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer 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",
        },
        data=json.dumps({
            "model": pull['bandit_arm']['model'],
            "messages": [
                {"role": "user", "content": user_q},
                {"role": "system", "content": pull['bandit_arm']['system_prompt']}
            ]
        })
    )
    # "PACKAGING" FOR UPDATE
    output = response.json()
    llm_response =  output['choices'][0]['message']['content']
    cost = output['usage']['cost']
    latency_ms = (time.perf_counter() - start) * 1000

    await update_on_reward(
        bandit_id, user_id, pull.get('event_id'),
        llm_response,
        reward=0.75 if len(llm_response) < 300 else 0.25,
        cost=cost,
        latency=latency_ms,
        is_human_reward=False
    )
    return {
        'pull': pull,
        'llm_response': llm_response,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    args = parser.parse_args()
    pull = asyncio.run(main(1, 1, args.query))

    print(pull['pull'])
    print('\n\n')
    print(pull['llm_response'])

