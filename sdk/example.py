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

# Initialize client
client = BanditoClient("http://localhost:8000")

# Authenticate
client.login("mike@mike.com", "Mike2123!")
print("Authenticated!")

# Pull an arm for a query
bandit_id = 1
query = "Explain the difference between supervised and unsupervised learning"

result = client.pull(bandit_id, query)
if result.budget_warning:
    print(f"⚠️ {result.budget_warning}")
    print('-'*50)

print(f"\nPulled arm {result.arm.id}:")
print(f"  Model: {result.arm.model_name}")
print(f"  Prompt: {result.arm.system_prompt[:50]}...")
print(f"  Context: {result.context}")
print(f"  Event ID: {result.event_id}")

start = time.perf_counter() # TODO: figure out how to measure this...without forcing user??

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    },
    data=json.dumps({
        "model": result.arm.model_name,
        "messages": [
            {"role": "user", "content": query},
            {"role": "system", "content": result.arm.system_prompt}
        ]
    })
)
# "PACKAGING" FOR UPDATE
output = response.json()
llm_response =  output['choices'][0]['message']['content']
cost = output['usage']['cost']
latency_ms = (time.perf_counter() - start) * 1000
score = 0.6 if len(llm_response) < 1000 else 0.2

event = client.reward(
    bandit_id=bandit_id,
    event_id=result.event_id,
    score=score,
    llm_output={"response": llm_response},
    cost=cost,
    latency=latency_ms
)
print(f"\nSubmitted reward: {score}")

# View leaderboard
lb = client.leaderboard(bandit_id)
print(json.dumps(lb.model_dump(), indent=4))

# View analysis
analysis = client.analysis(bandit_id)
print("\nModel Rankings:")
for entry in analysis["summary"]["model_ranking"]:
    print(f"  {entry['model']}: baseline_effect={entry['baseline_effect']:.4f}")

event = client.get_event(bandit_id=1, event_id=result.event_id)
print(f"Query: {event.user_query}")
print(f"Response: \n{event.llm_output['response']}")

available_options = ['0', '1']
choice = ""
while choice not in available_options:
    print("Choose from the following:")
    print("0 for bad")
    print("1 for good")
    choice = input("Enter your choice (0 or 1): ")
client.feedback(bandit_id=1, event_id=result.event_id, score=int(choice))

client.close()
