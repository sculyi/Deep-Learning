from env import *
from agent import *
#from RL import *
from priorityDQN import *
import  time
import numpy as np
#Q-learning
'''
def update():

    for episode in range(1000000):
        # initial observation
        targets = []
        targets.append(MovingAgent(init_pos=[6, 0], dst_pos=[6, env.height - 10], init_head=4))
        targets.append(MovingAgent(init_pos=[0, 10], dst_pos=[env.width - 10, 10], init_head=2))
        RL.set_agents_num(len(targets))

        observation = env.reset(targets)
        #print("start a new training {}".format(episode+1))
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))
            action = [int(i) for i in action]
            # RL take action and get next observation and reward
            observation_, reward, done,_ = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                #print (RL.q_table.shape)
                break

    # end of game
    print('game over')
    env.destroy()
'''

#doouble DQN
MEMORY_SIZE = 300#0

def update():
    total_steps = []
    q_vals=[]
    win = False
    for episode in range(100000000):
        # initial observation
        steps = 0
        targets = []
        targets.append(MovingAgent(init_pos=[6, 0], dst_pos=[37, 37], init_head=4,width = env.width,height = env.height))
        targets.append(MovingAgent(init_pos=[0, 10], dst_pos=[50,25], init_head=2,width = env.width,height = env.height))
        #targets.append(MovingAgent(init_pos=[16, 40], dst_pos=[69, 40], init_head=2, width=env.width, height=env.height))
        RL.set_agents_num(len(targets))

        observation = env.reset(targets)
        #print("start a new training {}".format(episode+1))

        start_time = time.time()
        while True:
            # fresh env
            tt = np.sum(np.array(total_steps))
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)
            action = [int(i) for i in action]
            # RL take action and get next observation and reward
            observation_, reward, done,rewards = env.step(action)

            RL.store_transition(observation, action, rewards, observation_)
            if done:
                RL.learn()
                break
            if done and reward == 10.0:  # win
                win = True
                print("win at epoch:{},last time is {}".format(episode,time.time()-start_time))

            #if tt > MEMORY_SIZE :  # learning
            #   RL.learn()

            if tt - MEMORY_SIZE > 20000:  # stop game
                #print("timeout at epoch:{}, last time is {}".format(episode,time.time()-start_time))
                break
            observation = observation_
            steps += 1

        total_steps.append(steps)
        q_vals.append(RL.q)
        if episode % 300 ==0: print("epoch={}".format(episode))
        if win: break


    # end of game
    print('game over')
    env.destroy()
    return total_steps, q_vals

if __name__ == "__main__":
    env = testEnv()
    #RL = QLearningTable(actions=list(range(env.n_actions)))
    agent_num = 2
    RL = DoubleDQN(n_agents=agent_num, n_actions= env.n_actions, n_features= 2*agent_num)

    env.after(100, update)
    env.mainloop()