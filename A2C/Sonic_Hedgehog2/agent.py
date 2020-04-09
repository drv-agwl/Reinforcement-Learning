from train import Agent
import sonic_env as envs
import numpy as np

if __name__ == '__main__':
    agent = Agent(alpha = 0.00001, beta = 0.00005)

    agent.actor.load_weights('actor-131.h5')
    agent.critic.load_weights('critic-131.h5')
    agent.policy.load_weights('policy-131.h5')

    print("Weights Loaded!")
    
    env = envs.make_train_0()
    score_history = []
    num_episodes = 2000
    
    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.custom_reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            env.render()
            
            agent.learn(observation, action, reward, observation_, done)
            observation = observation_
            
            score += reward
            
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        print("Episode ", i, " score %.2f average score %.2f" %(score, avg_score))

        if i%10 == 0:
          agent.actor.save_weights('actor-'+str(i+1)+'.h5')
          agent.critic.save_weights('critic-'+str(i+1)+'.h5')
          agent.policy.save_weights('policy-'+str(i+1)+'.h5')