import numpy as np
class MovingAgent():
    def __init__(self,init_pos=[0,0], dst_pos=[0,0],init_head=0,width=50, height=50):
        #print("start pos = {}, dst pos={}".format(init_pos,dst_pos))

        self.last_pos = []# save history positions of the agent
        self.cur_pos = init_pos # start position
        #init direction
        self.init_head = init_head
        self.dst_pos = dst_pos #destination position
        self.tk_id = -1 # object id in canvas
        self.all_pos=[{"l":self.last_pos,"c":self.cur_pos}] #save the positions at every second
        self.last_action = 0
        #canvas config
        self.width = width
        self.height = height

    def setTkId(self,tk_id,name):
        self.tk_id = tk_id
        self.name = name
    def isDst(self,pos_):
        return pos_ == self.dst_pos
    '''
    def head_type(self,dx,dy):
        #|/_\|(0,1,2,3,4) /_\(5,6,7)
        if dx == 0:
            if dy > 0:
                return 0
            elif dy  < 0:
                return 4
        elif dx  > 0:
            if dy > 0:
                return 1
            elif dy == 0:
                return 2
            elif dy < 0:
                return 3
        elif dx < 0:
            if dy == 0:
                return 6
            elif dy < 0:
                return 5
            elif dy > 0:
                return 7
    def move(self,head_type,action):
        if head_type == 0:
            if action == 0:
                return 0,1
            elif action == 1:
                return -1,1
            elif action == 2:
                return 1,1
        if head_type == 1:
            if action == 0:
                return 1,1
            elif action == 1:
                return 0,1
            elif action == 2:
                return 1,0
        if head_type == 2:
            if action == 0:
                return 1,0
            elif action == 1:
                return 1,1
            elif action == 2:
                return 1,-1
        if head_type == 3:
            if action == 0:
                return -1,-1
            elif action == 1:
                return 1,0
            elif action == 2:
                return 0,-1
        if head_type == 4:
            if action == 0:
                return 0,-1
            elif action == 1:
                return 1,-1
            elif action == 2:
                return -1,-1
        if head_type == 5:
            if action == 0:
                return -1,-1
            elif action == 1:
                return 0,-1
            elif action == 2:
                return -1,0
        if head_type == 6:
            if action == 0:
                return -1,0
            elif action == 1:
                return -1,-1
            elif action == 2:
                return -1,1
        if head_type == 7:
            if action == 0:
                return -1,1
            elif action == 1:
                return -1,0
            elif action == 2:
                return 0,1
     '''
    #calc the moving direction according to the dx and dy
    def head_type(self,dx,dy):
        #|/_\|(0,1,2,3,4) /_\(5,6,7)
        if dx == 0:
            if dy < 0: return 0
            elif dy  > 0: return 4
            elif dy == 0: return 8
        elif dx  > 0:
            if dy < 0: return 1
            elif dy == 0: return 2
            elif dy > 0:  return 3
        elif dx < 0:
            if dy == 0: return 6
            elif dy > 0: return 5
            elif dy < 0: return 7

    #got dx and dy of next step by the head and action
    def move(self,head_type,action):
        action -= 1 # temp process
        if head_type == 0:
            if action == 0: return 0,-1
            elif action == 1: return -1,-1
            elif action == 2: return 1,-1
        elif head_type == 1:
            if action == 0: return 1,-1
            elif action == 1: return 0,-1
            elif action == 2: return 1,0
        elif head_type == 2:
            if action == 0: return 1,0
            elif action == 1: return 1,-1
            elif action == 2: return 1,1
        elif head_type == 3:
            if action == 0: return -1,1
            elif action == 1: return 1,0
            elif action == 2: return 0,1
        elif head_type == 4:
            if action == 0: return 0,1
            elif action == 1: return 1,1
            elif action == 2: return -1,1
        elif head_type == 5:
            if action == 0: return -1,1
            elif action == 1: return 0,1
            elif action == 2: return -1,0
        elif head_type == 6:
            if action == 0: return -1,0
            elif action == 1: return -1,1
            elif action == 2: return -1,-1
        elif head_type == 7:
            if action == 0: return -1,-1
            elif action == 1: return -1,0
            elif action == 2: return 0,-1
        else: return 0,0
    #step
    def step(self,action):
        x,y=0,0
        if action != 0:
            if len(self.last_pos) == 0:
                ht = self.init_head
            else:
                for i in range(0,len(self.all_pos),1):
                    d_pos = self.all_pos[-(i+1)]
                    last,cur=d_pos['l'],d_pos['c']
                    if  last != cur:
                        if len(last)==0: ht = self.init_head
                        else:
                            #calc head_type by the adjacent position
                            dx = cur[0] - last[0]
                            dy = cur[1] - last[1]
                            # print(dx,dy)
                            ht = self.head_type(dx, dy)
                        break
            x,y = self.move(ht,action)
        #print(self.cur_pos)
        #avoid continuous zero action(keep standing)

        self.last_pos = [i for i in self.cur_pos]
        self.cur_pos[0] += x
        self.cur_pos[1] += y
        if self.cur_pos[0] <0 or self.cur_pos[0] > self.width or \
                        self.cur_pos[1] < 0 or self.cur_pos[1] > self.height:
            self.cur_pos = [i for i in self.last_pos]
            x = 0
            y = 0
        '''
        if self.cur_pos[0] <=0:
            self.cur_pos[0] = 0
        if self.cur_pos[0] >= self.width-1:
            self.cur_pos[0] = self.width-1
        if self.cur_pos[1] <=0:
            self.cur_pos[1]=0
        if self.cur_pos[1] >= self.height-1:
            self.cur_pos[1] = self.height-1
        '''

        #print(self.last_pos,self.cur_pos)
        #self.cur_pos[0] = int(self.cur_pos[0] )
        #self.cur_pos[1] = int(self.cur_pos[1] )
        self.all_pos.append({"l": self.last_pos, "c": self.cur_pos})

        dir_error = False#curently, discarded
        if (self.last_action == 2 and action == 3) or  (self.last_action == 3 and action == 2):
            dir_error = True

        self.last_action = action
        return x, y, self.isDst(self.cur_pos), dir_error