from __future__ import (division, print_function)
import os
import time
import networkx as nx
import numpy as np
import copy
import pickle
from collections import defaultdict
from tqdm import tqdm
import concurrent.futures

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as distributed

from model import *
from dataset import *
from utils.logger import get_logger
from utils.train_helper import data_to_gpu, snapshot, load_model, EarlyStopper
from utils.data_helper import *
from utils.eval_helper import *
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from utils.vis_helper import draw_graph_list, draw_graph_list_separate
from utils.data_parallel import DataParallel


try:
  ###
  # workaround for solving the issue of multi-worker
  # https://github.com/pytorch/pytorch/issues/973
  import resource
  rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
  resource.setrlimit(resource.RLIMIT_NOFILE, (10000, rlimit[1]))
  ###
except:
  pass

logger = get_logger('exp_logger')
__all__ = ['GranRunner', 'compute_edge_ratio', 'get_graph', 'evaluate']

NPR = np.random.RandomState(seed=1234)

def compute_edge_ratio(G_list):
  num_edges_max, num_edges = .0, .0
  for gg in G_list:
    num_nodes = gg.number_of_nodes()
    num_edges += gg.number_of_edges()
    num_edges_max += num_nodes**2

  ratio = (num_edges_max - num_edges) / num_edges
  return ratio


def get_graph(adj):
  """ get a graph from zero-padded adj """
  # remove all zeros rows and columns
  adj = adj[~np.all(adj == 0, axis=1)]
  adj = adj[:, ~np.all(adj == 0, axis=0)]
  adj = np.asmatrix(adj)
  G = nx.from_numpy_matrix(adj)
  return G


def evaluate(graph_gt, graph_pred, degree_only=True):
  mmd_degree = degree_stats(graph_gt, graph_pred)

  if degree_only:
    mmd_4orbits = 0.0
    mmd_clustering = 0.0
    mmd_spectral = 0.0
  else:    
    mmd_4orbits = orbit_stats_all(graph_gt, graph_pred)
    mmd_clustering = clustering_stats(graph_gt, graph_pred)    
    mmd_spectral = spectral_stats(graph_gt, graph_pred)
    
  return mmd_degree, mmd_clustering, mmd_4orbits, mmd_spectral


class GranRunner(object):

  def __init__(self, config):
    self.config = config
    self.seed = config.seed
    self.dataset_conf = config.dataset
    self.model_conf = config.model
    self.train_conf = config.train
    self.test_conf = config.test
    self.use_gpu = config.use_gpu
    self.gpus = config.gpus
    self.device = config.device
    self.writer = SummaryWriter(config.save_dir)
    self.is_vis = config.test.is_vis
    self.better_vis = config.test.better_vis
    self.num_vis = config.test.num_vis
    self.vis_num_row = config.test.vis_num_row
    self.is_single_plot = config.test.is_single_plot
    self.num_gpus = len(self.gpus)
    self.is_shuffle = False

    assert self.use_gpu == True

    if self.train_conf.is_resume:
      self.config.save_dir = self.train_conf.resume_dir

    ### load graphs
    self.graphs = create_graphs(config.dataset.name, data_dir=config.dataset.data_path)
    
    self.train_ratio = config.dataset.train_ratio
    self.dev_ratio = config.dataset.dev_ratio
    self.block_size = config.model.block_size
    self.stride = config.model.sample_stride
    self.num_graphs = len(self.graphs)
    self.num_train = int(float(self.num_graphs) * self.train_ratio)
    self.num_dev = int(float(self.num_graphs) * self.dev_ratio)
    self.num_test_gt = self.num_graphs - self.num_train
    self.num_test_gen = config.test.num_test_gen

    logger.info('Train/val/test = {}/{}/{}'.format(self.num_train, self.num_dev,
                                                   self.num_test_gt))

    ### shuffle all graphs
    if self.is_shuffle:
      self.npr = np.random.RandomState(self.seed)
      self.npr.shuffle(self.graphs)

    self.graphs_train = self.graphs[:self.num_train]
    self.graphs_dev = self.graphs[:self.num_dev]
    self.graphs_test = self.graphs[self.num_train:]
    
    self.config.dataset.sparse_ratio = compute_edge_ratio(self.graphs_train)
    logger.info('No Edges vs. Edges in training set = {}'.format(
        self.config.dataset.sparse_ratio))

    self.num_nodes_pmf_train = np.bincount([len(gg.nodes) for gg in self.graphs_train])    
    self.max_num_nodes = len(self.num_nodes_pmf_train)
    self.num_nodes_pmf_train = self.num_nodes_pmf_train / self.num_nodes_pmf_train.sum()
    
    ### save split for benchmarking
    if config.dataset.is_save_split:      
      base_path = os.path.join(config.dataset.data_path, 'save_split')
      if not os.path.exists(base_path):
        os.makedirs(base_path)
      
      save_graph_list(
          self.graphs_train,
          os.path.join(base_path, '{}_train.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_dev,
          os.path.join(base_path, '{}_dev.p'.format(config.dataset.name)))
      save_graph_list(
          self.graphs_test,
          os.path.join(base_path, '{}_test.p'.format(config.dataset.name)))

  def train(self):
    ### create data loader
    train_dataset = eval(self.dataset_conf.loader_name)(self.config, self.graphs_train, tag='train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=self.train_conf.batch_size,
        shuffle=self.train_conf.shuffle,
        num_workers=self.train_conf.num_workers,
        collate_fn=train_dataset.collate_fn,
        drop_last=False)

    # create models
    model = eval(self.model_conf.name)(self.config)

    if self.use_gpu:
      model = DataParallel(model, device_ids=self.gpus).to(self.device)

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    if self.train_conf.optimizer == 'SGD':
      optimizer = optim.SGD(
          params,
          lr=self.train_conf.lr,
          momentum=self.train_conf.momentum,
          weight_decay=self.train_conf.wd)
    elif self.train_conf.optimizer == 'Adam':
      optimizer = optim.Adam(params, lr=self.train_conf.lr, weight_decay=self.train_conf.wd)
    else:
      raise ValueError("Non-supported optimizer!")

    early_stop = EarlyStopper([0.0], win_size=100, is_decrease=False)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=self.train_conf.lr_decay_epoch,
        gamma=self.train_conf.lr_decay)

    # reset gradient
    optimizer.zero_grad()

    # resume training
    resume_epoch = 0
    if self.train_conf.is_resume:
      model_file = os.path.join(self.train_conf.resume_dir,
                                self.train_conf.resume_model)
      load_model(
          model.module if self.use_gpu else model,
          model_file,
          self.device,
          optimizer=optimizer,
          scheduler=lr_scheduler)
      resume_epoch = self.train_conf.resume_epoch

    # Training Loop
    iter_count = 0    
    results = defaultdict(list)
    for epoch in range(resume_epoch, self.train_conf.max_epoch):
      model.train()
      lr_scheduler.step()
      train_iterator = train_loader.__iter__()

      for inner_iter in range(len(train_loader) // self.num_gpus):
        optimizer.zero_grad()

        batch_data = []
        if self.use_gpu:
          for _ in self.gpus:
            data = train_iterator.next()
            batch_data.append(data)
            iter_count += 1
        
        
        avg_train_loss = .0        
        for ff in range(self.dataset_conf.num_fwd_pass):
          batch_fwd = []
          
          if self.use_gpu:
            for dd, gpu_id in enumerate(self.gpus):
              data = {}
              data['adj'] = batch_data[dd][ff]['adj'].pin_memory().to(gpu_id, non_blocking=True)          
              data['edges'] = batch_data[dd][ff]['edges'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_gnn'] = batch_data[dd][ff]['node_idx_gnn'].pin_memory().to(gpu_id, non_blocking=True)
              data['node_idx_feat'] = batch_data[dd][ff]['node_idx_feat'].pin_memory().to(gpu_id, non_blocking=True)
              data['label'] = batch_data[dd][ff]['label'].pin_memory().to(gpu_id, non_blocking=True)
              data['att_idx'] = batch_data[dd][ff]['att_idx'].pin_memory().to(gpu_id, non_blocking=True)
              data['subgraph_idx'] = batch_data[dd][ff]['subgraph_idx'].pin_memory().to(gpu_id, non_blocking=True)
              batch_fwd.append((data,))

          if batch_fwd:
            train_loss = model(*batch_fwd).mean()              
            avg_train_loss += train_loss              

            # assign gradient
            train_loss.backward()
        
        # clip_grad_norm_(model.parameters(), 5.0e-0)
        optimizer.step()
        avg_train_loss /= float(self.dataset_conf.num_fwd_pass)
        
        # reduce
        train_loss = float(avg_train_loss.data.cpu().numpy())
        
        self.writer.add_scalar('train_loss', train_loss, iter_count)
        results['train_loss'] += [train_loss]
        results['train_step'] += [iter_count]

        if iter_count % self.train_conf.display_iter == 0 or iter_count == 1:
          logger.info("NLL Loss @ epoch {:04d} iteration {:08d} = {}".format(epoch + 1, iter_count, train_loss))

      # snapshot model
      if (epoch + 1) % self.train_conf.snapshot_epoch == 0:
        logger.info("Saving Snapshot @ epoch {:04d}".format(epoch + 1))
        snapshot(model.module if self.use_gpu else model, optimizer, self.config, epoch + 1, scheduler=lr_scheduler)
    
    pickle.dump(results, open(os.path.join(self.config.save_dir, 'train_stats.p'), 'wb'))
    self.writer.close()
    
    return 1

  def test(self):
    self.config.save_dir = self.test_conf.test_model_dir

    ### Compute Erdos-Renyi baseline    
    if self.config.test.is_test_ER:
      p_ER = sum([aa.number_of_edges() for aa in self.graphs_train]) / sum([aa.number_of_nodes() ** 2 for aa in self.graphs_train])      
      graphs_gen = [nx.fast_gnp_random_graph(self.max_num_nodes, p_ER, seed=ii) for ii in range(self.num_test_gen)]
    else:
      ### load model
      model = eval(self.model_conf.name)(self.config)
      model_file = os.path.join(self.config.save_dir, self.test_conf.test_model_name)
      load_model(model, model_file, self.device)

      if self.use_gpu:
        model = nn.DataParallel(model, device_ids=self.gpus).to(self.device)

      model.eval()

      ### Generate Graphs
      A_pred = []
      num_nodes_pred = []
      num_test_batch = int(np.ceil(self.num_test_gen / self.test_conf.batch_size))

      gen_run_time = []
      for ii in tqdm(range(num_test_batch)):
        with torch.no_grad():        
          start_time = time.time()
          input_dict = {}
          input_dict['is_sampling']=True
          input_dict['batch_size']=self.test_conf.batch_size
          input_dict['num_nodes_pmf']=self.num_nodes_pmf_train
          A_tmp = model(input_dict)
          gen_run_time += [time.time() - start_time]
          A_pred += [aa.data.cpu().numpy() for aa in A_tmp]
          num_nodes_pred += [aa.shape[0] for aa in A_tmp]

      logger.info('Average test time per mini-batch = {}'.format(
          np.mean(gen_run_time)))
          
      graphs_gen = [get_graph(aa) for aa in A_pred]

    ### Visualize Generated Graphs
    if self.is_vis:
      num_col = self.vis_num_row
      num_row = int(np.ceil(self.num_vis / num_col))
      test_epoch = self.test_conf.test_model_name
      test_epoch = test_epoch[test_epoch.rfind('_') + 1:test_epoch.find('.pth')]
      save_name = os.path.join(self.config.save_dir, '{}_gen_graphs_epoch_{}_block_{}_stride_{}.png'.format(self.config.test.test_model_name[:-4], test_epoch, self.block_size, self.stride))

      # remove isolated nodes for better visulization
      graphs_pred_vis = [copy.deepcopy(gg) for gg in graphs_gen[:self.num_vis]]

      if self.better_vis:
        for gg in graphs_pred_vis:
          gg.remove_nodes_from(list(nx.isolates(gg)))

      # display the largest connected component for better visualization
      vis_graphs = []
      for gg in graphs_pred_vis:        
        CGs = [gg.subgraph(c) for c in nx.connected_components(gg)]
        CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
        vis_graphs += [CGs[0]]

      if self.is_single_plot:
        draw_graph_list(vis_graphs, num_row, num_col, fname=save_name, layout='spring')
      else:
        draw_graph_list_separate(vis_graphs, fname=save_name[:-4], is_single=True, layout='spring')

      save_name = os.path.join(self.config.save_dir, 'train_graphs.png')

      if self.is_single_plot:
        draw_graph_list(
            self.graphs_train[:self.num_vis],
            num_row,
            num_col,
            fname=save_name,
            layout='spring')
      else:      
        draw_graph_list_separate(
            self.graphs_train[:self.num_vis],
            fname=save_name[:-4],
            is_single=True,
            layout='spring')

    ### Evaluation
    if self.config.dataset.name in ['lobster']:
      acc = eval_acc_lobster_graph(graphs_gen)
      logger.info('Validity accuracy of generated graphs = {}'.format(acc))

    num_nodes_gen = [len(aa) for aa in graphs_gen]
    
    # Compared with Validation Set    
    num_nodes_dev = [len(gg.nodes) for gg in self.graphs_dev]  # shape B X 1
    mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev = evaluate(self.graphs_dev, graphs_gen, degree_only=False)
    mmd_num_nodes_dev = compute_mmd([np.bincount(num_nodes_dev)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

    # Compared with Test Set    
    num_nodes_test = [len(gg.nodes) for gg in self.graphs_test]  # shape B X 1
    mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test = evaluate(self.graphs_test, graphs_gen, degree_only=False)
    mmd_num_nodes_test = compute_mmd([np.bincount(num_nodes_test)], [np.bincount(num_nodes_gen)], kernel=gaussian_emd)

    logger.info("Validation MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(mmd_num_nodes_dev, mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev))
    logger.info("Test MMD scores of #nodes/degree/clustering/4orbits/spectral are = {}/{}/{}/{}/{}".format(mmd_num_nodes_test, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test))

    if self.config.dataset.name in ['lobster']:
      return mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test, acc
    else:
      return mmd_degree_dev, mmd_clustering_dev, mmd_4orbits_dev, mmd_spectral_dev, mmd_degree_test, mmd_clustering_test, mmd_4orbits_test, mmd_spectral_test
