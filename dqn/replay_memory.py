"""Code from https://github.com/tambetm/simple_dqn/blob/master/src/replay_memory.py"""

import os
import sys
import random
import logging
import math
import numpy as np

from .utils import save_npy, load_npy, save_pkl, load_pkl
from prioritized_experience_replay.binary_heap import BinaryHeap


# class ReplayMemory(object):
#   """
#   Abstract super class for replay memory.
#   Defines required methods and configuration
#   All implementations need to implement this class
#   """

#   def __init__(self, config, model_dir):
#     raise NotImplementedError('Subclasses must override initialization!')

#   def add(self, previous, reward, action, next, terminal):
#     raise NotImplementedError('Subclasses must override add!')

#   def sample(self, step):
#     raise NotImplementedError('Subclasses must override sample!')

#   def save(self):
#     raise NotImplementedError('Subclasses must override save!')

#   def load(self):
#     raise NotImplementedError('Subclasses must override load!')




class ReplayUniform(object):

  def __init__(self, config, model_dir):
    self.model_dir = model_dir

    self.cnn_format = config.cnn_format
    self.memory_size = config.memory_size
    self.actions = np.empty(self.memory_size, dtype = np.uint8)
    self.rewards = np.empty(self.memory_size, dtype = np.integer)
    self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype = np.float16)
    self.terminals = np.empty(self.memory_size, dtype = np.bool)
    self.history_length = config.history_length
    self.dims = (config.screen_height, config.screen_width)
    self.batch_size = config.batch_size
    self.count = 0
    self.current = 0

    # pre-allocate prestates and poststates for minibatch
    self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)
    self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype = np.float16)

  def add(self, reward, action, next, terminal):
    assert next.shape == self.dims
    # NB! screen is post-state, after action and reward
    self.actions[self.current] = action
    self.rewards[self.current] = reward
    self.screens[self.current, ...] = next
    self.terminals[self.current] = terminal
    self.count = max(self.count, self.current + 1)
    self.current = (self.current + 1) % self.memory_size

  def getState(self, index):
    assert self.count > 0, "replay memory is empty, use at least --random_steps 1"
    # normalize index to expected range, allows negative indexes
    index = index % self.count
    # if is not in the beginning of matrix
    if index >= self.history_length - 1:
      # use faster slicing
      return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
    else:
      # otherwise normalize indexes and use slower list based access
      indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
      return self.screens[indexes, ...]

  def sample(self, step=0):  #Step argumnt required by subclass, but not here
    # memory must include poststate, prestate and history
    assert self.count > self.history_length
    # sample random indexes
    indexes = []
    while len(indexes) < self.batch_size:
      # find random index 
      while True:
        # sample one index (ignore states wraping over 
        index = random.randint(self.history_length, self.count - 1)
        # if wraps over current pointer, then get new one
        if index >= self.current and index - self.history_length < self.current:
          continue
        # if wraps over episode end, then get new one
        # NB! poststate (last screen) can be terminal state!
        if self.terminals[(index - self.history_length):index].any():
          continue
        # otherwise use this index
        break
      
      # NB! having index first is fastest in C-order matrices
      self.prestates[len(indexes), ...] = self.getState(index - 1)
      self.poststates[len(indexes), ...] = self.getState(index)
      indexes.append(index)

    actions = self.actions[indexes]
    rewards = self.rewards[indexes]
    terminals = self.terminals[indexes]

    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
    else:
      return self.prestates, actions, rewards, self.poststates, terminals

  def save(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      save_npy(array, os.path.join(self.model_dir, name))

  def load(self):
    for idx, (name, array) in enumerate(
        zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
            [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
      array = load_npy(os.path.join(self.model_dir, name))


class ReplayRanked(ReplayUniform):
  """
  Wrapper class for prioirity replay to match base class
  """
  

  def __init__(self, config, model_dir):

    #Covert configuration to appropriate one for prioritized replay memory
    # expConfig = {
    #   'size': config.memory_size,
    #   'batch_size': config.batch_size,
    #   'learn_start': config.learn_start,
    #   'total_steps': config.max_step #Is this the same? TODO: check this
    # }
    # self.exp = Experience(expConfig)

    #TODO: Initialize binary heap!

    #Initialize common data structures and config
    super(ReplayRanked, self).__init__(config, model_dir)

    #retrieve additional config 
    self.priority_size = config.priority_size
    self.partition_num = config.partition_num
    self.alpha = config.alpha
    self.beta_zero = config.beta_zero
    self.learn_start = config.learn_start
    self.total_steps = config.max_step

    self.record_size = 0

    #Initialize binary heap
    self.priority_queue = BinaryHeap(self.priority_size)

    #Preprocess priorities 
    self.distributions = self.build_distributions()

    self.beta_grad = (1 - self.beta_zero) / (self.total_steps - self.learn_start)


  def build_distributions(self):
    """
    preprocess pow of rank
    (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
    :return: distributions, dict
    """
    res = {}
    n_partitions = self.partition_num
    partition_num = 1
    # each part size
    partition_size = int(math.floor(self.memory_size / n_partitions))

    for n in range(partition_size, self.memory_size + 1, partition_size):
      if self.learn_start <= n <= self.priority_size:
        distribution = {}
        # P(i) = (rank i) ^ (-alpha) / sum ((rank i) ^ (-alpha))
        pdf = list(
            map(lambda x: math.pow(x, -self.alpha), range(1, n + 1))
        )
        pdf_sum = math.fsum(pdf)
        distribution['pdf'] = list(map(lambda x: x / pdf_sum, pdf))
        # split to k segment, and than uniform sample in each k
        # set k = batch_size, each segment has total probability is 1 / batch_size
        # strata_ends keep each segment start pos and end pos
        cdf = np.cumsum(distribution['pdf'])
        strata_ends = {1: 0, self.batch_size + 1: n}
        step = 1.0 / self.batch_size
        index = 1
        for s in range(2, self.batch_size + 1):
          while cdf[index] < step:
            index += 1
          strata_ends[s] = index
          step += 1.0 / self.batch_size

        distribution['strata_ends'] = strata_ends

        res[partition_num] = distribution

      partition_num += 1

    return res

  def add(self, reward, action, next, terminal):

    #TODO: call super class and store priorities too

    # assert next.shape == self.dims

    # #Transpose back since history get gives a transposed screen if cnn format is NHWC
    # print(previous.shape)
    # if self.cnn_format == 'NHWC':
    #   previous = np.transpose(previous, (2, 0, 1))
    # print(previous.shape)
    # assert previous.shape == self.dims
    # #Convert stored experience to tuple
    # experience = (previous, action, reward, next, terminal)
    # self.exp.store(experience)

    # self.count = max(self.count, self.current + 1)
    # self.current = (self.current + 1) % self.exp.size

    assert next.shape == self.dims #Asserted here too as to not add faulty data to PQ
    #Extra variable to sample from correct distribution
    if self.record_size <= self.memory_size:
      self.record_size += 1

    priority = self.priority_queue.get_max_priority()
    self.priority_queue.update(priority, self.current)
    super(ReplayRanked, self).add(reward, action, next, terminal)

  def retrieve(self, indices):
    """
    get experience from indices
    :param indices: list of experience id
    :return: experience replay samples
    """
    rewards = []
    actions = []
    terminals = []
    for v in indices:
      rewards.append(self.rewards[v])
      actions.append(self.actions[v])
      terminals.append(self.terminals[v])
    return rewards, actions, terminals

  def rebalance(self):
    """
    rebalance priority queue
    :return: None
    """
    self.priority_queue.balance_tree()

  def update_priority(self, indices, delta):
    """
    update priority according indices and deltas
    :param indices: list of experience id
    :param delta: list of delta, order correspond to indices
    :return: None
    """
    for i in range(0, len(indices)):
      self.priority_queue.update(math.fabs(delta[i]), indices[i])

  def sample(self, step):

    #TODO: sample, compute weights

  #   # memory must include poststate, prestate and history
  #   assert self.count > self.history_length

  #   self.actions = np.empty(self.exp.batch_size, dtype = np.uint8)
  #   self.rewards = np.empty(self.exp.batch_size, dtype = np.integer)
  #   self.terminals = np.empty(self.exp.batch_size, dtype = np.bool)
  #   prestates = np.empty((self.exp.batch_size, self.history_length) + self.dims, dtype = np.float16)
  #   poststates = np.empty((self.exp.batch_size, self.history_length) + self.dims, dtype = np.float16)
    
  #   experience, w, exp_indices = self.exp.sample(step)

  #   for i in range(self.exp.batch_size):
  #     sample = experience[i]
  #     prestates[i] = sample[0]
  #     actions[i] = sample[1]
  #     rewards[i] = sample[2]
  #     poststates[i] = sample[3]
  #     terminals[i] = sample[4]


  #   if self.cnn_format == 'NHWC':
  #     return np.transpose(prestates, (0, 2, 3, 1)), actions, \
  #       rewards, np.transpose(poststates, (0, 2, 3, 1)), terminals, w, exp_indices
  #   else:
  #     return prestates, actions, rewards, poststates, terminals, w, exp_indices

    if self.record_size < self.learn_start:
      sys.stderr.write('Record size less than learn start! Sample failed\n')
      return False, False, False, False, False, False, False
    # memory must include poststate, prestate and history
    assert self.count > self.history_length

    dist_index = int(math.floor((self.record_size * 1.0) / self.memory_size * self.partition_num))#Implicit cast to float
    partition_size = int(math.floor(self.memory_size / self.partition_num))
    partition_max = dist_index * partition_size
    distribution = self.distributions[dist_index] #Retrieve correct distribution according to number of transitions in memory
    rank_list = []
    # sample from k segments, results in list of ranks
    for n in range(1, self.batch_size + 1):
      index = random.randint(distribution['strata_ends'][n] + 1,
                             distribution['strata_ends'][n + 1])
      rank_list.append(index)

    # beta, increase by global_step, max 1
    beta = min(self.beta_zero + (step - self.learn_start - 1) * self.beta_grad, 1)
    # find all alpha pow, notice that pdf is a list, start from 0
    alpha_pow = [distribution['pdf'][v - 1] for v in rank_list]
    # w = (N * P(i)) ^ (-beta) / max w
    w = np.power(np.array(alpha_pow) * partition_max, -beta)
    w_max = max(w)
    w = np.divide(w, w_max)
    # rank list is priority queue id
    # convert to experience id
    rank_e_id = self.priority_queue.priority_to_experience(rank_list)
    # get experience id according rank_e_id
    rewards, actions, terminals = self.retrieve(rank_e_id)

    for i in range(self.batch_size):
      self.prestates[i, ...] = self.getState(rank_e_id[i] - 1)
      self.poststates[i, ...] = self.getState(rank_e_id[i])
    
    if self.cnn_format == 'NHWC':
      return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
        rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals, w, rank_e_id
    else:
      return self.prestates, actions, rewards, self.poststates, terminals, w, rank_e_id

  def save(self):

    super(ReplayRanked, self).save()
    save_pkl(self.priority_queue, os.path.join(self.model_dir, 'pq.pkl'))

    # if not os.path.exists(self.model_dir):
    #   os.makedirs(self.model_dir)
    # save_pkl(experience, os.path.join(self.model_dir, 'experience.pkl'))

  def load(self):

    super(ReplayRanked, self).load()
    self.priority_queue = load_pkl(os.path.join(self.model_dir, 'pq.pkl'))

    # self.exp = load_pkl(path)
    