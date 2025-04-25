import json
import random
import asyncio
import websockets
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

tf.config.set_visible_devices([], 'GPU')
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="DQN Agent Parameters")
    parser.add_argument('--model_number', type=int, default=0, help="")
    parser.add_argument('--relearn', type=int, default=0, help="")    
    args = parser.parse_args()
    return args

def get_model(i_s, o_s, model_number):
    model = load_model(f'model_step_{model_number}.h5')
    print(f"loading models model_step_{model_number}.h5")
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
    return model

def create_dqn_model(input_size, output_size):
    model = Sequential([
        Dense(128, input_dim=input_size, activation='relu'),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(output_size, activation='linear')
    ])
    model.compile(loss=MeanSquaredError(), optimizer=Adam(learning_rate=0.001))
    return model

class DQNAgent:
    def __init__(self, input_size, output_size, relearn, 
                 model_number=0, 
                 epsilon=0.01, 
                 epsilon_decay=0.995, 
                 epsilon_min=0.01, 
                 gamma=0.99, 
                 target_update_frequency=100):
        if relearn == 1:
            print("loading models")
            self.model = get_model(input_size, output_size, model_number)
            self.target_model = get_model(input_size, output_size, model_number)
        else:
            self.model = create_dqn_model(input_size, output_size)
            self.target_model = create_dqn_model(input_size, output_size)
            self.update_target_model()
        
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.output_size = output_size
        self.target_update_frequency = target_update_frequency  # How often to update target model
        if relearn==0:
            self.step_counter = 0  # Keep track of the number of training steps
        else:
            self.step_counter = model_number+5
    def update_target_model(self):
        """Updates the target model to be the same as the online model."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state_vector, nor_mov_len):
        """Selects an action based on epsilon-greedy policy."""
        if np.random.rand() < self.epsilon or (nor_mov_len >= 75 and nor_mov_len%20 == 0):
            return random.randint(0, self.output_size - 1)  # Exploration
        q_values = self.model.predict(np.array([state_vector]), verbose=0)[0]  # Exploitation
        return int(np.argmax(q_values))  # Action with max Q-value

    def train(
            self, 
            state, 
            action, 
            reward, 
            next_state, 
            done):
        """Updates the model with the reward received from the environment."""
        target_qs = self.model.predict(np.array([state]), verbose=0)  # Predicted Q-values for the current state
        next_qs = self.target_model.predict(np.array([next_state]), verbose=0)  # Predicted Q-values for the next state

        # Apply the Bellman update to calculate the target Q-value
        if done:
            target_qs[0][action] = reward
        else:
            target_qs[0][action] = reward + self.gamma * np.max(next_qs)

        # Train the model to minimize the loss between the predicted and target Q-values
        self.model.fit(np.array([state]), target_qs, epochs=1, verbose=0)

        # Decay epsilon after each action to balance exploration and exploitation
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # Update the target model periodically
        self.step_counter += 1
        if (self.step_counter%3000 == 0):
            self.target_model.save(f'model_step_{self.step_counter}.h5')
        if (self.step_counter%3000 == 0):
            self.epsilon=0.1
        if (self.step_counter%100 == 0):
            # print(self.epsilon, self.step_counter)
            pass
        if self.step_counter % self.target_update_frequency == 0:
            self.update_target_model()

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, prev_state_vector, 
            prev_action, 
            reward, 
            state_vector, 
            done):
        """Store a transition (experience)"""
        self.buffer.append((
                        prev_state_vector, 
                        prev_action, 
                        reward, 
                        state_vector, 
                        done))

    def sample(self, batch_size):
        """Sample a batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        prev_state_vectors, prev_actions, rewards, state_vectors, dones= map(list, zip(*batch))
        return prev_state_vectors, prev_actions, rewards, state_vectors, dones

    def __len__(self):
        return len(self.buffer)


class ActionCycleDetector:
    def __init__(self, max_pattern_length=5, required_repeats=3):
        self.action_history = deque(maxlen=max_pattern_length * required_repeats)
        self.max_pattern_length = max_pattern_length
        self.required_repeats = required_repeats

    def add_action(self, action):
        self.action_history.append(action)
    def reset_q(self):
        self.action_history.clear()

    def detect_cycle(self):
        history = list(self.action_history)
        n = len(history)

        for pattern_len in range(1, self.max_pattern_length + 1):
            if n < pattern_len * self.required_repeats:
                continue

            pattern = history[-pattern_len:]
            match = True
            for i in range(2, self.required_repeats + 1):
                if history[-i * pattern_len:-(i - 1) * pattern_len] != pattern:
                    match = False
                    break

            if match and pattern != [1]:
                print(pattern)
                return True, pattern  # Cycle detected

        return False, None

def direction_to_one_hot(dir):
    x, y = dir['x'], dir['y']
    return [
        1 if y == -1 else 0,  # Up
        1 if y == 1 else 0,   # Down
        1 if x == -1 else 0,  # Left
        1 if x == 1 else 0    # Right
    ]

def vectorize_state(state):
    MAX_BODY_LENGTH = 20
    head = state.get("head")
    snake = state.get("snake")
    food = state.get("food")
    danger_state = state.get("dangerState")
    state_vector = [head['x'], head['y']]
    # for i in range(MAX_BODY_LENGTH):
    #     if i < len(snake) - 1:
    #         state_vector.append(snake[i+1]['x'])
    #         state_vector.append(snake[i+1]['y'])
    #     else:
    #         state_vector.append(0)
    #         state_vector.append(0)
    state_vector.extend([food['x'], food['y']])
    state_vector.extend(state.get('distance'))
    
    direction = state.get("direction")
    ldirection = state.get("ldirection")
    # state_vector.extend([direction['x'], direction['y']])
    # state_vector.extend([ldirection['x'], ldirection['y']])
    
    state_vector.extend(direction_to_one_hot(direction))
    state_vector.extend(direction_to_one_hot(ldirection))

    delta_x = food["x"] - head["x"]
    delta_y = food["y"] - head["y"]
    state_vector.extend([delta_x, delta_y])
    state_vector.append(state.get('tileCount'))
    state_vector.append(len(snake))
    state_vector.extend(danger_state)
    return np.array(state_vector)

def write_to_file(step, fruits):
    with open("logs.txt", "a") as f:
        f.write(f"{step}: {fruits}.\n")
# WebSocket handler with real-time action decision and sending to client
async def handle_connection(websocket):
    print("Client connected.")
    args = parse_args()
    agent = DQNAgent(input_size=26, 
                     output_size=3, 
                     model_number=args.model_number,
                     relearn=args.relearn
                     )
    print('1   --> ')
    replay_buffer = ReplayBuffer(capacity=10000)
    detector = ActionCycleDetector(max_pattern_length=10, required_repeats=8)
    prev_state_vector = None
    prev_action = None
    prev_state = None
    # move_reward = 1
    step=args.model_number
    step1=args.model_number
    replaysteps = 0
    episode=0
    episode_len = 0
    prev_episode_len = 0
    nor_mov_len = 0
    fruits=0
    try:
        print('2   --> ')
        async for message in websocket:
            state = json.loads(message)
            code = state.get("code")
            step=step+1
            if code == 2:  # When game state is ready for action
                reward = state.get("reward")
                krew = reward
                loopid = state.get("loopid")
                prinepisode=False
                if loopid != episode:
                    prinepisode=True
                    episode = loopid
                    prev_episode_len = episode_len
                    episode_len = 0
                episode_len = episode_len + 1
                state_vector = vectorize_state(state)
                state_vector = np.append(state_vector, 0)
                # Train the model with the previous state and action (if available)
                if prev_state_vector is not None:
                    gr=100
                    br=-10
                    mr=0

                    is_cycle, pattern = detector.detect_cycle()
                    if is_cycle and reward != gr:
                        # print(f"Cycle detected: {pattern}")
                        print(f"#C-{pattern}#", end=" ", flush=True)
                        detector.reset_q()
                        reward = -10
                    if reward == gr:
                        print("\n##########################")
                        print("###### FRUIT EATEN  ######")
                        print("##########################")
                        fruits+=1
                        
                    elif reward == br:
                        pass
                    elif reward == mr:
                        pass
                    
                    # Set to True if the game is over
                    done = False  
                    if reward == br:
                        done = True
                        detector.reset_q()

                    reward=round(reward,2)
                    # Print rewards
                    if reward == br: 
                        print(reward)
                    else:
                        print(reward, end=" ", flush=True)
                    
                    batch_size=32
                    if len(replay_buffer) >= 1000 and step%20 == 0:
                        prev_state_vectors, prev_actions, rewards, state_vectors, dones = replay_buffer.sample(batch_size)
                        print("Replay Buffer Training started")
                        step1=step + replaysteps*batch_size
                        replaysteps = replaysteps + 1
                        for i in range(batch_size):
                            step1+=1
                            if step1%1000 == 0:
                                write_to_file(step1, fruits)
                                fruits=0
                            # agent.train(
                            #     prev_state_vectors[i], 
                            #     prev_actions[i], 
                            #     rewards[i], 
                            #     state_vectors[i], 
                            #     dones[i]
                            # )
                        print("Replay Buffer Training ended")
                    state_vector[-1] = nor_mov_len
                    if reward == 100:
                        print('############# inserting good reward 30 times ############# ')
                        for i in range(7):
                            # replay_buffer.add(prev_state_vector, prev_action,reward,state_vector,done)
                            pass
                    else:
                        # replay_buffer.add(prev_state_vector, prev_action,reward,state_vector,done)
                        pass
                    
                    reward=round(reward,2)
                    # agent.train(
                    #     prev_state_vector, 
                    #     prev_action, 
                    #     reward, 
                    #     state_vector, 
                    #     done)
                    if (prinepisode) :
                        print(
                            '\n',
                            'model = ', args.model_number,
                            ' step = ', step + replaysteps*batch_size,
                            ' done = ', done,
                            ' reward = ', round(reward, 2),
                            ' episode = ', episode,
                            ' episode_len = ', prev_episode_len
                        )
                    # print(step, step%10==0)
                    if (step + replaysteps*batch_size)%1000==0:
                        write_to_file(step, fruits)
                        fruits=0
                action = agent.act(state_vector, nor_mov_len)
                detector.add_action(action)
                await websocket.send(json.dumps({"action": action + 1}))  # +1 for action index
                
                prev_state_vector = state_vector
                prev_action = action
                prev_state = state

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")

# Start WebSocket server
async def main():
    async with websockets.serve(handle_connection, "localhost", 8765):
        await asyncio.Future()

# Run the WebSocket server
asyncio.run(main())


'''
target_qs = self.model.predict(np.array([state]), verbose=0)  # Predicted Q-values for the current state
        next_qs = self.target_model.predict(np.array([next_state]), verbose=0)  # Predicted Q-values for the next state

        # Apply the Bellman update to calculate the target Q-value
        if done:
            target_qs[0][action] = reward
        else:
            target_qs[0][action] = reward + self.gamma * np.max(next_qs)

model doesn't know the action . action is our own interpretation.
model only knows Q-values
you are just providing the max of the Qvalues it provided.
so you are feeding the max Q value it gave you indicating the most significant part of it's - 
- output i.e; the action taken for future. (all this time we are assuming 
that model is good at predicting future Q values as well. so, if model predict bad future values 
the rewards are bad, and needs more exploration and more learning)

afterall why include a future reward (or future Q-values) ?
 i.e; to catch the transition from a normal state to going towards a good state -
  - based on the reward it is goingto get again
 i.e; to catch the transition from a normal state to going towards a bad state -
  - based on the reward it is goingto get again
 .
 . 
 .
 .
 .
 suppose the normal state is zero for so long but future predicted Q value is left all the time.
 now everytime you only feed the reward-punishment bellman equation you are just feeding the previous action
 so basically you are not changin anything except little.

 maybe the model starts giving [0.25, 0.25, 0.25, 0.25] --> indecision
 but suddenly we get a reward (here we also now the action taken). 
 the model will understand that by taking a certain action and previously it has a  0.25 future reward(indecision) 
 we got reward. it will update that action. previously it was a mild 0.25 yeild for too many normal rewards 
 and a sudden boost for a god reward is a story for the model to learn from. 


 too mnay small reward ---> a sudden big reward is a very good story
 too mnay small reward ---> a sudden negative reward is a very good story
 good bad good bad good bad ---. nothing good and insightful about this story.

 



 ----------------------------
 -----------------------------


 5. No Replay Memory / Too Small Memory
Without an experience buffer, you're training on a tiny slice of the data.

✅ Use a replay buffer like:

python
Copy
Edit
deque(maxlen=10000)
And train on random batches:

python
Copy
Edit
random.sample(memory, batch_size)

1 1   0.2     
2 4   0.4   
3 9   0.6   
4 16  0.8   






'''