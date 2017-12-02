"""
author: lyi
date: 2017-09-27
description: a gridded env for RL
"""
#-*- coding:utf-8 -*-

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
from utils import *
import numpy as np
from agent import *
from maze_dis import bfs

class BaseEnv(tk.Tk, object):
    def __init__(self, unit=20, height=50, width=50, conflit_dis=5,
                 mindis_loadfile = None, mindis_savefile='mindis.txt'):
        super(BaseEnv, self).__init__()
        self.title('BaseEnv')
        self.unit = unit  # pixels per grid
        self.height = height  # grid height
        self.width = width  # grid width
        self.marginal = unit // 10 # marginal size between object and grid
        # if the dis between two object less equal than conflit_dis,
        #it means the situation is danger(the reward will be negative)
        self.conflit_dis = conflit_dis

        self.action_space = ['k','s', 'l', 'r'] #keep, straight forward, turn left and right
        self.n_actions = len(self.action_space)

        self.geometry('{0}x{1}'.format(self.width * self.unit, self.height * self.unit))
        #create canvas
        self.canvas = tk.Canvas(self, bg='grey',
                                height=self.height * self.unit,
                                width=self.width * self.unit)
        #event binding
        self.canvas.bind('<Button-1>', self.clicked)
        self.targets=[] # targets (instance of MovingAgent)
        self.build_basic_grids()#build the grids(width, height), black color
        # the orginal position of object
        self.origin = np.array([self.unit // 2, self.unit // 2])
        self.features = 0#the input dim of NN
        self.toDst = [] #save the objects which reach the destination
        self.min_dis ={} #calc the min dis between two grids
        self.paths = []  # save all the valid positions of env
        self.build_path() #build the path, white color
        #the remaining initial is not work
        self.mindis_loadfile = mindis_loadfile
        self.mindis_savefile = mindis_savefile
        self.cal_count = 0

        if mindis_loadfile is None:
            self.calc_min_dis()
        else:
            with open(mindis_loadfile,'r') as fr:
                for line in fr:
                    seg_val = line.replace("\n","").replace("\r").split("|")
                    if len(seg_val) != 2: continue
                    self.min_dis[seg_val[0]] = seg_val[1]
    #callback function of mouse click
    def clicked(self,event):
        print(event.x//self.unit, event.y//self.unit)

    def build_basic_grids(self):
        # create grids
        for c in range(0, self.width * self.unit, self.unit):
            x0, y0, x1, y1 = c, 0, c, self.height * self.unit
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, self.height * self.unit, self.unit):
            x0, y0, x1, y1 = 0, r, self.width * self.unit, r
            self.canvas.create_line(x0, y0, x1, y1)
    def add_target(self, acft):
        if not isinstance(acft, MovingAgent):
            raise Exception("Wrong parameter type!")
            return
        #use the count as tid of object
        tid = str(len(self.targets))
        #create the moving mark
        t = self.canvas.create_text(self.unit / 2, self.unit/2, fill='red', text=tid)
        #create static mark at destination position
        tt = self.canvas.create_text(self.unit / 2, self.unit / 2, fill='green', text=tid)
        #move target to (x,y) and static destination remark
        x,y = acft.cur_pos[0],acft.cur_pos[1]
        self.canvas.move(t, x * self.unit, y*self.unit)
        self.canvas.move(tt, acft.dst_pos[0]* self.unit, acft.dst_pos[1]* self.unit)

        acft.setTkId(t,tid) #set property
        self.canvas.pack()#add targets to canvas
        self.targets.append(acft) #save objects information
        #buffer size of the coords of all targets
        self.features = 2 * len(self.targets)
        #indicate the object if reach destination
        self.toDst = [1 for _ in range(len(self.targets))]

    def build_path(self):
        #add all path grid to self.paths
        pass
    def calc_min_dis(self):
        pass

    def reset(self, targets):
        #delete current object and re-init all targets
        self.render()
        for t in self.targets:
            self.canvas.delete(t.tk_id)
        self.targets = []
        for t in targets:
            self.add_target(t)
        # return observation
        return np.array([self.canvas.coords(t.tk_id) for t in self.targets]).reshape(self.features)
    #distance of two positions
    def dis(self,pos1,pos2):
        d = np.abs(np.array(pos1)-np.array(pos2))
        return d[0],d[1]
    #if the dis between two targets less than the thresold?
    def isConflict(self):
        for i in range(len(self.targets)):
            for j in range(i+1,len(self.targets),1):
                dx,dy = self.dis(self.targets[i].cur_pos, self.targets[j].cur_pos)
                if dx < self.conflit_dis or dy < self.conflit_dis:
                    return i,j
        return None, None
    '''calc the min-dis by bfs in maze_dis.py
    set a = dis between last position and dst position,
    b  = dis between current position and dst position,
    if a >b, reward = 3; if a < b, reward = -3, else reward = -1
    '''
    def MoveScore(self,s,s_,dst):
        '''s = (np.array(s) - self.origin) // self.unit
        s_ = (np.array(s_) - self.origin) // self.unit
        dst= (np.array(dst) - self.origin) // self.unit'''

        s = [s[1],s[0]]
        s_ = [s_[1], s_[0]]
        dst = [dst[1], dst[0]]
        #print(s,s_,dst)
        sp1, sp2 = str(s)+str(dst), str(s_)+str(dst)
        if not sp1 in self.min_dis:
            self.min_dis[sp1] = bfs(self.base_grid, s, dst)#calc min dis
            self.cal_count += 1
        if not sp2 in self.min_dis:
            self.min_dis[sp2] = bfs(self.base_grid, s_, dst)#calc min dis
            self.cal_count += 1
        if self.cal_count % 100 == 0 and (not self.mindis_savefile is None):
            print("save min dis to file")
            with open(self.mindis_savefile, 'w+') as fr:
                for k, v in self.min_dis.items():
                    fr.write("{}|{}\n".format(k, int(v)))

        d1, d2 = self.min_dis[sp1], self.min_dis[sp2]
        if d1 > d2:
            return 3.0
        elif d1 < d2:
            return -3.0
        else:
            return -1.0
        #obsoleted
        d1x,d1y = abs(dst[0]-s[0]),abs(dst[1]-s[1])
        d2x,d2y = abs(dst[0]-s_[0]),abs(dst[1]-s_[1])
        if d1x > d2x and d1y > d2y:
            return -1.0
        elif d1x < d2x and d1y < d2y:
            return 1.0
        else:
            return 0.0

    #next observation
    def step(self, actions):
        #coords of all targets
        ss = [self.canvas.coords(t.tk_id) for t in self.targets] #self.canvas.coords(self.rect)
        ss_, a_rewards=[],[] # the states for next time and the whole rewards, to be filled

        for s, t, a in zip(ss, self.targets, actions):
            idx = int(t.name)
            cur_grid = t.cur_pos
            dst = t.dst_pos
            a = a * (self.toDst[idx]) # if reach the dst, force to 0 (hold on)
            reward = 0.0
            #has arrived dst pos
            has_arravied = t.isDst(t.cur_pos)
           #step over for whole env
            x,y,arrived,dir_error = t.step(a)
            next_grid = [cur_grid[0] +x, cur_grid[1]+y]
            if arrived:
                reward = 10.0
                self.toDst[idx] = 0
                print("target {} reach the destination".format(t.name))
            #if dir_error:
            #    reward -= (ppa/3)

            if x != 0 or y != 0:
                self.canvas.move(t.tk_id, x * self.unit, y * self.unit)  # move agent
            s_ = self.canvas.coords(t.tk_id)#next coord of object
            # out of bounds, in the hell or conflicts
            id1,id2 = self.isConflict()
            if id1 is not None and id2 is not None:
                reward = -7.0
                #print('conflicts between agent {} and {}'.format(
                #    self.targets[id1].name, self.targets[id2].name))
            if not self.canvas.coords(t.tk_id) in self.paths:
                reward = -10.0
                #print("target {} out of paths".format(t.name))

            # leave destination(wrong action)
            if has_arravied and a != 0: reward = -10.0
            if reward == 0.0: # the reward depends on the distance to the dst pos
                reward = self.MoveScore(cur_grid, next_grid,dst)

            a_rewards.append(reward)
            ss_.append(np.array(s_)+ np.random.random()*np.random.random())

        #ss_ = [self.canvas.coords(t.tk_id) for t in self.targets] # next state
        #print("ss_",ss_)
        # reward function
        '''
        if arrived_t == len(self.targets):
            reward = 1
            done = True
        else:
            done = (reward == -1.0)
        '''
        reward, done = get_global_reward(a_rewards)
        #print('reward',reward)
        return np.array(ss_).reshape(self.features), reward, done, a_rewards

    def render(self,t=0.01):
        time.sleep(t)
        self.update()
#a test Env
class testEnv(BaseEnv, object):
    def __init__(self, unit=10, height=50, width=75, conflit_dis=2,
                 mindis_loadfile = None, mindis_savefile='mindis.txt'):
        super(testEnv, self).__init__(unit, height, width, conflit_dis, mindis_loadfile, mindis_savefile)

    def build_path(self):
        origin = self.origin
        pm = (self.unit - self.marginal) / 2
        nums=[10,self.height-10]
        for pos in nums:
            for i in range(self.width):
                hc = origin + np.array([self.unit * i, self.unit*pos])
                self.canvas.create_rectangle(
                    hc[0] - pm, hc[1] - pm,
                    hc[0] + pm, hc[1] + pm,
                    fill='white')
                self.paths.append(hc.tolist())
        nums = [6,self.width-6]
        for pos in nums:
            for i in range(self.height):
                hc = origin + np.array([self.unit * pos, self.unit * i])
                self.canvas.create_rectangle(
                    hc[0] - pm, hc[1] - pm,
                    hc[0] + pm, hc[1] + pm,
                    fill='white')
                self.paths.append(hc.tolist())
        init_pos = [( 3,3),(self.width-3, self.height-3),(3,self.height-3),(self.width-3,3)]
        delta_pos =[(1,1),(-1,-1),(1,-1),(-1,1)]
        for ip,dp in zip(init_pos,delta_pos):
            ix,iy,dx,dy = ip[0],ip[1],dp[0],dp[1]
            while ix <= self.width and iy <= self.height and ix >=0 and iy >= 0:
                hc = origin + np.array([self.unit *ix,self.unit *iy])
                self.canvas.create_rectangle(
                    hc[0] - pm, hc[1] - pm,
                    hc[0] + pm, hc[1] + pm,
                    fill='white')
                self.paths.append(hc.tolist())
                ix += dx
                iy += dy

        self.base_grid = np.ones((self.height+1, self.width+1 ))
        self.path_grid_index=[]
        for pos in self.paths:
            pos = (np.array(pos)-self.origin) // self.unit
            self.path_grid_index.append([pos[1],pos[0]])
            self.base_grid[pos[1]][pos[0]] = 0
        '''
        for i in range(self.base_grid.shape[0]):
            pstr = ''
            for j in range(self.base_grid.shape[1]):
                pstr += (str(self.base_grid[i][j])[0] +' ')
            print(pstr)
        '''
        self.canvas.pack()

    def calc_min_dis(self):
        return
        c = 0
        for i in range(len(self.path_grid_index)):
            for j in range(len(self.path_grid_index)):
                if i == j: #same grid
                    continue
                pos1, pos2 = self.path_grid_index[i], self.path_grid_index[j]
                if self.base_grid[pos1[0]][pos1[1]] == 1 or self.base_grid[pos2[0]][pos2[1]] == 1:
                    continue #canno moving
                mindis = bfs(self.base_grid, pos1, pos2)
                sp = str(pos1)+str(pos2)
                c+=1
                self.min_dis[sp]=mindis
        if not self.mindis_savefile is None:
            with open(self.mindis_savefile,'w+') as fr:
                for k,v in self.min_dis.items():
                    fr.write("{}|{}\n".format(k,int(v)))



def update():
    targets = []
    targets.append(MovingAgent(init_pos=[6, 0], dst_pos=[6, env.height - 10], init_head=4))
    targets.append(MovingAgent(init_pos=[0, 10], dst_pos=[env.width - 10, 10], init_head=2))
    for t in range(100):
        print("\ntest on iteration {}".format(t+1))
        targets = []
        targets.append(MovingAgent(init_pos=[6, 0], dst_pos=[6, env.height - 10], init_head=4))
        targets.append(MovingAgent(init_pos=[0, 10], dst_pos=[env.width - 10, 10], init_head=2))
        s = env.reset(targets)
        while True:
            env.render()
            a = [1 for _ in targets]#[np.random.randint(0,env.n_actions) for _ in targets]
            s, r, done = env.step(a)
            if done:
                break
        time.sleep(1)


if __name__ == '__main__':
    env = testEnv()
    env.after(100, update)
    env.mainloop()