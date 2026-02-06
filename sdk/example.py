"""
Example usage of the Bandito SDK.

This shows a typical agent workflow:
1. Authenticate
2. Pull an arm to get model/prompt config
3. Call the LLM (simulated here)
4. Submit reward with results
5. View leaderboard
"""
import os
import time
import json
import requests
from bandito import BanditoClient
from dotenv import load_dotenv
load_dotenv()


def make_llm_request(user_query: str, model_name: str, system_prompt: str, ):
    # TODO: switch to pydantic AI for full experiment testing/evaluation (e.g text -> SQL, text categorization, etc)
    start = time.perf_counter() # TODO: how to do this w/o forcing user to...decorator? context manager?
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        },
        data=json.dumps({
            "model": model_name,
            "messages": [
                {"role": "user", "content": query},
                {"role": "system", "content": system_prompt}
            ]
        })
    )
    output = response.json()
    llm_response =  output['choices'][0]['message']['content']
    cost = output['usage']['cost']
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        'response': llm_response,
        'cost': cost,
        'latency': latency_ms
    }


client = BanditoClient("http://localhost:8000")
client.login("mike@mike.com", "Mike2123!")

bandit_id = 3
query = """
Multi-armed bandits (MABs) shine when decisions must be made while you’re still learning, and the world won’t pause so you can run a clean randomized trial.

At their core, bandits balance exploration vs. exploitation: you repeatedly choose among “arms” (ads, treatments, recommendations), observe rewards, and adapt choices to favor what’s working. The system learns online and changes behavior as evidence accumulates.

Compared with A/B/n testing, bandits are more resource-efficient and faster to reward: instead of sending fixed traffic to bad variants until the test ends, they rapidly shift traffic toward better options. This means higher cumulative reward over the test period, not just a better final decision. They also avoid the garden of statistical sins common in A/B testing (peeking, early stopping, p-hacking) because they’re designed to be sequential and adaptive from the start.

Relative to full reinforcement learning, bandits are simpler and safer. They assume a stationary environment and no long-term state dynamics: each choice has an immediate reward, and today’s choice doesn’t change tomorrow’s context in a complex way. That makes them ideal for ad allocation, email subject lines, pricing experiments with small state, and product recommendations where the primary concern is short-term click or revenue, not strategic long-horizon planning. You keep much of RL’s adaptivity without the fragility and tuning hell.

Compared with contextual bandits’ cousins like supervised learning models, plain supervised learning assumes you already have labeled data. Bandits don’t; they create their own data through interaction. If you can’t afford a big offline dataset or the environment shifts frequently (seasonality, trends, competitor changes), bandits adapt in real time instead of periodically retraining a static model.

Against heuristics and rules of thumb (“always pick the historically best option,” “rotate evenly”), bandits offer principled uncertainty handling. Algorithms like Thompson sampling or UCB quantify uncertainty and try out promising but under-tested options, which matters when the apparent winner might just be lucky noise. They also gracefully handle many arms where naive rotation would waste traffic.

Where bandits don’t shine is when you need strict hypothesis testing, clean interpretability, or long-term causal understanding (“does variant B fundamentally change user satisfaction over months?”). They optimize reward, not p-values or explanatory stories. For deep scientific inference or regulatory settings, classical experiments or causal designs are often better.

In short, multi-armed bandits are most valuable when you care about maximizing outcomes during learning, not merely identifying a winner after the fact, and when the world you’re acting in is too impatient for static experiments and too simple (or high-stakes) for full-blown reinforcement learning.
"""

result = client.pull(bandit_id, query)
if result.budget_warning:
    print(f"⚠️ {result.budget_warning}")
    continue_yn = ['y', 'n']
    answer = ''
    while choice not in continue_yn:
        print("Choose from the following:")
        print("'y' to continue")
        print("'n' to stop")
        answer = input("Enter your choice (y or n): ")
    if answer != 'y':
        print('Stopping before LLM call.')
        raise

print(f"Using {result.arm.model_name}, arm id {result.arm.id}.")
print(f"Event ID: {result.event_id}")

response = make_llm_request(query, result.arm.model_name, result.arm.system_prompt)

if len(response['response']) > len(query):
    score = 0
elif len(response['response']) < len(query) * 0.5:
    score = 1
elif len(response['response']) < len(query):
    score = 0.5
else:
    score = 0

# score = 0.6 if len(response['response']) < 1000 else 0.2

event = client.reward(
    bandit_id=bandit_id,
    event_id=result.event_id,
    score=score,
    llm_output={"response": response['llm_response']},
    cost=response['cost'],
    latency=response['latency']
)
# lb = client.leaderboard(bandit_id)

event = client.get_event(bandit_id=bandit_id, event_id=result.event_id)
print(f"Query: {event.user_query}")
print('-'*100)
print(f"Response: \n{event.llm_output['response']}")

available_options = ['0', '1']
choice = ""
while choice not in available_options:
    print("Choose from the following:")
    print("0 for bad")
    print("1 for good")
    choice = input("Enter your choice (0 or 1): ")
client.feedback(bandit_id=bandit_id, event_id=result.event_id, score=int(choice))

client.close()
