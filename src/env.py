import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import subprocess as sp
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

# added left + jump
SPEEDRUN_MOVEMENT = [
    ['NOOP'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['right', 'A', 'B'],
    ['A'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['left', 'A', 'B'],
    ['down']
]

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),"-pix_fmt", "rgb24", "-r", "60", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())


def process_frame(frame):
    if frame is not None:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
        return frame
    else:
        return np.zeros((1, 84, 84))


class CustomReward(Wrapper):
    def __init__(self, env=None, world=None, stage=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.current_x = 40
        self.world = world
        self.stage = stage
        # delete later
        # self.previous_state = None
        # self.var=0
        # self.eps=0
        # self.reachTop=0
        # self.nearTop = 0
        self.jumpedHole = 0
        self.hitPipe=0
        self.ledge1=0
        self.ledge2=0        
        self.pipe3=0
        self.pastpipe4=0
        self.pipe5=0
        self.nearFlag=0
        self.pastsecpipe=0
        
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def step(self, action):
        # delete start
        # if (self.previous_state is None) and self.var==0:
        #     self.eps+=1
        #     self.previous_state = self.env.render(mode='rgb_array')
        #     self.previous_state = process_frame(self.previous_state)
        #     self.var+=1        
        # delete end
        
        state, reward, done, info = self.env.step(action)
        # if self.monitor:
        #     self.monitor.record(state)
        state = process_frame(state)
        
        # # delete start
        # if self.previous_state is not None and self.var==2:
        #     curr_state = state.reshape(84, 84)
        #     prev_state = self.previous_state.reshape(84, 84)
        #     total_difference = np.sum(np.abs(curr_state - prev_state))
        #     if total_difference>2500:
        #         print("WAHOO")
        #         reward+=3000
        # else:
        #     self.var+=1

        # self.previous_state= state.copy()
        # # delete end
        
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]
        if done:
            self.jumpedHole = 0
            self.hitPipe=0
            self.ledge1=0
            self.ledge2=0        
            self.pipe3=0
            self.pastpipe4=0
            self.pipe5=0
            self.nearFlag=0
            self.pastsecpipe=0
            # self.previous_state = None
            # self.var=0
            # self.eps=0
            if info["flag_get"]:
                reward += 100
            else:
                reward -= 100
                
                
                
                        
        # reward for 1-2 to make it work:
        x = info["x_pos"]
        y = info["y_pos"]
        if self.world == 1 and self.stage == 2:
            underbridge = (2320<=x<=2410 and y <= 115)
            goforpipe = (2560 <= x <=2700 and y <= 190)
            reachTop = (2574 <= x <= 2590 and y>=250)
            topTrap = (923 <= x <= 978 and y>=140)
            nearTop = (2540 <= x and y >= 269)
            
            if nearTop and self.nearTop == 0:
                self.nearTop+=1
                reward+=300
            if reachTop and self.reachTop == 0:
                self.reachTop+=1
                reward +=3000
                print("WAHOOOOO!")
                
            if underbridge or topTrap:
                print("what the fuck")
                reward -= 300
                done = True
            if goforpipe:
                reward -=300
                done = True



        if self.world == 8 and self.stage == 2:
            led1 = (2323<=x<=2327 and y==79)
            led2 = (2450<=x<=2454 and y==79)
            pipe3 = (2487 <=x<2490 and y==143)
            pastpipe = (2532<=x<2600)
            pipe5 = (2800<=x<=2850)
            nearFlag = (2861<=x<=2700)
            if(led1 and self.ledge1 < 1):
                self.ledge1+=1
                reward += 300
            if(led2 and self.ledge2 < 1):
                self.ledge2+=1
                reward += 350
            if(pipe3 and self.pipe3<1):
                self.pipe3+=1
                reward += 400
            if(pastpipe and self.pastpipe4 <1):
                self.pastpipe4+=1
                reward+=600
            if(pipe5 and self.pipe5 < 1):
                self.pipe5+=1
                reward += 800
            if(nearFlag and self.nearFlag < 1):
                self.nearFlag+=1
                reward+=1000
            
                
        if self.world == 8 and self.stage == 1:
            jumpedOverHole = (3636<=x<=3693)
            hitSecondPipe = (3862<=x<=3867 and y==143)
            pastSecondPipe = (x>=3900 and y>=180)
            if(jumpedOverHole and self.jumpedHole<1):
                self.jumpedHole +=1
                reward+=250
            if(hitSecondPipe and self.hitPipe<1):
                self.hitPipe +=1
                reward+=500
            if(pastSecondPipe and self.pastsecpipe<1):
                self.pastsecpipe += 1
                reward+=800
                
        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191) or info["x_pos"] < self.current_x - 500:
                reward -= 50
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward = -50
                done = True

        self.current_x = info["x_pos"]
        return state, reward / 10., done, info

    def reset(self):
        self.curr_score = 0
        self.current_x = 40
        # self.previous_state = None
        # self.var=0
        # self.eps=0
        # self.nearTop = 0
        # self.reachTop=0
        self.jumpedHole = 0
        self.hitPipe=0
        self.ledge1=0
        self.ledge2=0        
        self.pipe3=0
        self.pastpipe4=0
        self.pipe5=0
        self.nearFlag=0
        self.pastsecpipe=0
        return process_frame(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(skip, 84, 84))
        self.skip = skip
        self.states = np.zeros((skip, 84, 84), dtype=np.float32)

    def step(self, action):
        total_reward = 0
        last_states = []
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if i >= self.skip / 2:
                last_states.append(state)
            if done:
                self.reset()
                return self.states[None, :, :, :].astype(np.float32), total_reward, done, info
        max_state = np.max(np.concatenate(last_states, 0), 0)
        self.states[:-1] = self.states[1:]
        self.states[-1] = max_state
        return self.states[None, :, :, :].astype(np.float32), total_reward, done, info

    def reset(self):
        state = self.env.reset()
        self.states = np.concatenate([state for _ in range(self.skip)], 0)
        return self.states[None, :, :, :].astype(np.float32)


def create_train_env(world, stage, actions, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    env = JoypadSpace(env, actions)
    env = CustomReward(env, world, stage, monitor)
    env = CustomSkipFrame(env)
    return env


class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs, output_path=None):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe() for _ in range(num_envs)])
        if action_type == "right":
            actions = RIGHT_ONLY
        elif action_type == "simple":
            actions = SIMPLE_MOVEMENT
        elif action_type == "complex":
            actions = COMPLEX_MOVEMENT
        else:
            print("speeeeedrun checcck env.py")
            actions = SPEEDRUN_MOVEMENT
        self.envs = [create_train_env(world, stage, actions, output_path=output_path) for _ in range(num_envs)]
        self.num_states = self.envs[0].observation_space.shape[0]
        self.num_actions = len(actions)
        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                self.env_conns[index].send(self.envs[index].step(action.item()))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError
 