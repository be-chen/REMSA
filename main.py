from agent_orchestrator import FMSAgent

agent = FMSAgent()

print("Welcome to RESMA: Remote Sensing Foundation Model Selection Agent")
print("Type your model requirement below. Type 'exit' to quit.")

while True:
    user_input = input("\n[User] >> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    try:
        response, response_db = agent.run(user_input)
        print(f"\n{response}")
    except Exception as e:
        print(f"[Error] {e}")