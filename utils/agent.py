def load_agent():
    from dqn_agent import DQNAgent
    agent = DQNAgent()
    agent.load("agent_weights.pth")  # Load trained weights
    return agent

def decide_action(agent, obs):
    return agent.select_action(obs)
