import tensorflow as tf
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

class SumTree(object):
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    def sample(self, n):
        b_idx, b_memory, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, self.tree.data[0].size)), np.empty((n, 1))
        pri_seg = self.tree.total_p / n       # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        min_prob += 1e-7
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob/min_prob, -self.beta)
            b_idx[i], b_memory[i, :] = idx, data
        return b_idx, b_memory, ISWeights

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


class DoubleDQN:
    def __init__(
            self,
            n_agents,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            sess=None,
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = memory_size // 5 #replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.learn_step_counter = 0
        # = np.zeros((self.memory_size, self.n_features * 2 + self.n_agents * 2))
        self.memory = Memory(capacity=memory_size)
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l, w_initializer, b_initializer, trainable):
            n_l1 = n_l[0]
            outs=[]

            n_layers = len(n_l)
            for i in range(self.n_agents):
                with tf.variable_scope('l1'):
                    w1 = tf.get_variable('w1_{}'.format(i), [self.n_features, n_l1],
                                         initializer=w_initializer, collections=c_names, trainable=trainable)
                    b1 = tf.get_variable('b1_{}'.format(i), [1, n_l1],
                                         initializer=b_initializer, collections=c_names, trainable=trainable)
                    l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
                with tf.variable_scope('l3'):
                    w3 = tf.get_variable('w3_{}'.format(i), [n_l1, n_l1],
                                         initializer=w_initializer, collections=c_names, trainable=trainable)
                    b3 = tf.get_variable('b3_{}'.format(i), [1, n_l1],
                                         initializer=b_initializer, collections=c_names, trainable=trainable)
                    l2 = tf.nn.relu(tf.matmul(l1, w3) + b3)
                #l2 = l1
                with tf.variable_scope('l2_{}'.format(i)):
                    w2 = tf.get_variable('w2_{}'.format(i), [n_l1,  self.n_actions],
                                         initializer=w_initializer, collections=c_names, trainable=trainable)
                    b2 = tf.get_variable('b2_{}'.format(i), [1,  self.n_actions],
                                     initializer=b_initializer, collections=c_names, trainable=trainable)
                    #out = tf.nn.softmax(tf.matmul(l1, w2) + b2)
                    out = tf.matmul(l2, w2) + b2

                outs.append( out )
            return tf.transpose(tf.stack(outs),[1,0,2])


        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        #flatten ouput for multi-agents
        self.q_target = tf.placeholder(tf.float32, [None, self.n_agents , self.n_actions], name='Q_target')  # for calculating loss
        self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')

        with tf.variable_scope('eval_net'):
            c_names, n_l, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], [20], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l, w_initializer, b_initializer,True)

        with tf.variable_scope('loss'):
            self.abs_errors = tf.reduce_sum(tf.abs(self.q_target -self.q_eval ), axis=[1,2])
            ts = tf.transpose(self.q_target,[1,0,2])
            es = tf.transpose(self.q_eval,[1,0,2])
            losses = []
            for i in range(self.n_agents):
                t,e = ts[i],es[i]
                t, e = tf.nn.softmax(t), tf.nn.softmax(e)
                loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(t, e))
                #loss = -tf.reduce_sum(self.ISWeights * t * tf.log(e))
                losses.append(loss)
            self.loss = tf.reduce_mean(tf.reduce_sum(losses))
            #self.abs_errors = tf.reduce_sum(tf.abs(ts[0]-es[0]+ts[1]-es[1]), axis=1)

        with tf.variable_scope('train'):
            #self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            self._train_op = [tf.train.RMSPropOptimizer(self.lr).minimize(_loss) for _loss in losses]

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l, w_initializer, b_initializer,False)
    def set_agents_num(self,n_agents):
        pass

    def store_transition(self, s, a, r, s_):
        def _softmax_(x):
            return np.exp(x) / np.sum(np.exp(x), axis=0)
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        self.memory_counter +=1
        #r = _softmax_(r)
        transition = np.hstack((s, r, a, s_))
        self.memory.store(transition)  # have high priority for newly arrived transition
        return
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s,  r, a, s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})[0]
        #print(actions_value.shape, actions_value)
        #actions_value = actions_value.reshape(self.n_agents,self.n_actions)
        action = np.argmax(actions_value,axis = 1)
        #print(action)
        action_value = np.max(actions_value,axis = 1)
        if hasattr(self, 'memory_counter') and self.memory_counter % self.memory_size == 0:
            print('PARAMS',actions_value)

        if not hasattr(self, 'q'):  # record action value it gets
            self.q = []
            self.running_q = np.zeros(self.n_agents)
        self.running_q = self.running_q*0.99 + 0.01 * action_value
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action
            action = np.array([np.random.randint(0, self.n_actions) for _ in range(self.n_agents)])
        #print('choosed actions:', action)
        #action = self.toDst * action
        return action

    def learn(self):
        #print("in learning...")
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #print('\ntarget_params_replaced\n')
        '''
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        '''
        tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)

        q_next, q_eval4next = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features+self.n_agents:self.n_features+2*self.n_agents ].astype(int)
        reward = np.array(batch_memory[:, self.n_features:self.n_features+self.n_agents ])

        max_act4next = np.argmax(q_eval4next, axis=2)        # the action that brings the highest value is evaluated by q_eval
        # print('max_act4next',q_eval4next, max_act4next)
        selected_q_next=[]
        for n_bs in batch_index:
            cur_sq = [q_next[n_bs][i][act] for i, act in enumerate(max_act4next[n_bs])]
            selected_q_next.append(cur_sq)
        selected_q_next = np.array(selected_q_next)
        #selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        #print('selected_q_next',selected_q_next.shape)

        cal_reward = reward + self.gamma * selected_q_next
        #print('cal_reward',cal_reward)
        for n_bs in batch_index:
            for i, act in enumerate(eval_act_index[n_bs]):
                q_target[n_bs][i][act]  = cal_reward[n_bs][i]

        #q_target = q_target.reshape(self.batch_size,self.n_agents*self.n_actions)
        #print('q_target',q_target)
        '''_, self.cost0,self.cost1 = self.sess.run([self._train_op, self.loss0,self.loss1],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})'''

        _, abs_errors,self.cost = self.sess.run(
            [self._train_op, self.abs_errors, self.loss],
            feed_dict={self.s: batch_memory[:, :self.n_features],
                       self.q_target: q_target,
                       self.ISWeights: ISWeights})
        #print('abs_errors',abs_errors)
        self.memory.batch_update(tree_idx, abs_errors)  # update priority

        self.cost_his.append(self.cost)
        #print('self.cost_his',self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1




