from env import MeatBuyingDiscreteEnv
from model import MeatBuyingQLAgent

MAX_EPISODE = 500

Env = MeatBuyingDiscreteEnv
Agent = MeatBuyingQLAgent


def train(env: Env, agent: Agent):
    for episode in range(MAX_EPISODE):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action()
            next_state, reward, done, _ = env.step()
            agent.update(action, reward)
            state = next_state
