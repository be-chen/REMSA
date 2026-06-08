from agent_orchestrator import FMSAgent

agent = FMSAgent()

print("Welcome to REMSA: Remote Sensing Foundation Model Selection Agent")
print("Describe what kind of remote sensing foundation model you need.")
# print("For better recommendations, please include at least:")
# print("  - Task: e.g., classification, segmentation, detection, change detection, retrieval")
# print("  - Data modality: e.g., optical imagery, SAR, hyperspectral, multispectral, LiDAR, time series")
# print("You can also mention details like specific application, sensor, resolution, bands, region, deployment device, or minimum performance requirements.")
print("Example: 'I need a model for crop mapping using Sentinel-2 multispectral imagery at 10m resolution in Europe.'")
print("Type 'exit' to quit.")

while True:
    user_input = input("\n[User] >> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    try:
        response, response_db = agent.run(user_input)
        if response == "exit":
            break
        print(f"\n{response}")
    except Exception as e:
        print(f"[Error] {e}")