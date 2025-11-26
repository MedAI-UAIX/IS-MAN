import subprocess
import time
import os

agents = [
    {"name": "triage_agent", "yaml": "SonoMind/models/triage_agent.yaml", "port": 8000},
    {"name": "sonographer_agent", "yaml": "SonoMind/models/sonographer_agent.yaml", "port": 8001},
    {"name": "radiologist_agent", "yaml": "SonoMind/models/radiologist_agent.yaml", "port": 8002},
    {"name": "physician_agent", "yaml": "SonoMind/models/physician_agent.yaml", "port": 8003},
]

processes = []

for agent in agents:
    env = os.environ.copy()
    env["API_PORT"] = str(agent["port"])
    cmd = ["llamafactory-cli", "api", agent["yaml"]]
    print(f"Starting {agent['name']} on port {agent['port']}...")
    p = subprocess.Popen(cmd, env=env)
    processes.append(p)
    time.sleep(1)

print("All agents are running. Press Ctrl+C to stop.")

try:
    for p in processes:
        p.wait()
except KeyboardInterrupt:
    print("Stopping all agents...")
    for p in processes:
        p.terminate()
