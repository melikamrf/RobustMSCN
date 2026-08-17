import time
import numpy as np
import pdb
import math
import pandas as pd
import json
import sys
import torch
import os
from collections import defaultdict
import random
import copy

from query_representation.utils import *

from evaluation.eval_fns import *
from .dataset import QueryDataset, pad_sets, to_variable,\
        mscn_collate_fn,mscn_collate_fn_together

from .nets import *
from .decoder import Decoder
from evaluation.flow_loss import FlowLoss, \
        get_optimization_variables, get_subsetg_vectors
from .discriminator import LatentDiscriminator, LatentGenerator
from .DANN import train_one_epoch_dann as _train_one_epoch_dann
from .seeding import DEFAULT_TRAIN_SEED, derive_seed, make_generator, \
        make_numpy_generator, make_worker_init_fn, seed_everything

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.optim.swa_utils import AveragedModel, SWALR

import wandb
import random
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

QERR_MIN_EPS=0.0000001
DEBUG_TIMES=False

def qloss_torch(yhat, ytrue):
    assert yhat.shape == ytrue.shape
    # yhat = yhat + 1
    # ytrue = ytrue + 1

    epsilons = to_variable([QERR_MIN_EPS]*len(yhat)).float()

    ytrue = torch.max(ytrue, epsilons)
    yhat = torch.max(yhat, epsilons)

    errors = torch.max( (ytrue / yhat), (yhat / ytrue))

    return errors

def mse_pos(yhat, ytrue):
    assert yhat.shape == ytrue.shape
    errors = torch.nn.MSELoss(reduction="none")(yhat, ytrue)

    for i,err in enumerate(errors):
        if yhat[i] < ytrue[i]:
            errors[i] *= 10

    return errors

def mse_ranknet(yhat, ytrue):
    mseloss = torch.nn.MSELoss(reduction="mean")(yhat, ytrue)
    rloss = ranknet_loss(yhat, ytrue)
    return mseloss + 0.1*rloss

def ranknet_loss(batch_pred, batch_label):
    '''
    :param batch_preds: [batch, ranking_size] each row represents the relevance predictions for documents within a ltr_adhoc
    :param batch_label:  [batch, ranking_size] each row represents the standard relevance grades for documents within a ltr_adhoc
    :return:
    '''
    batch_pred = batch_pred.unsqueeze(0)
    batch_label = batch_label.unsqueeze(0)
    # batch_pred = batch_pred.T
    # batch_label = batch_label.T
    sigma = 1.0

    batch_s_ij = torch.unsqueeze(batch_pred, dim=2) - torch.unsqueeze(batch_pred, dim=1)  # computing pairwise differences w.r.t. predictions, i.e., s_i - s_j

    batch_p_ij = 1.0 / (torch.exp(-sigma * batch_s_ij) + 1.0)

    batch_std_diffs = torch.unsqueeze(batch_label, dim=2) - torch.unsqueeze(batch_label, dim=1)  # computing pairwise differences w.r.t. standard labels, i.e., S_{ij}
    batch_Sij = torch.clamp(batch_std_diffs, min=-1.0, max=1.0)  # ensuring S_{ij} \in {-1, 0, 1}
    batch_std_p_ij = 0.5 * (1.0 + batch_Sij)

    # about reduction, both mean & sum would work, mean seems straightforward due to the fact that the number of pairs differs from query to query
    batch_loss = F.binary_cross_entropy(input=torch.triu(batch_p_ij, diagonal=1), target=torch.triu(batch_std_p_ij, diagonal=1), reduction='mean')

    return batch_loss

class CardinalityEstimationAlg():

    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        pass

    def train(self, training_samples, **kwargs):
        pass

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subplan). Each key should be ' ' separated
        list of aliases / table names
        '''
        pass

    def get_exp_name(self):
        name = self.__str__()
        if not hasattr(self, "rand_id"):
            self.rand_id = str(random.getrandbits(32))
            print("Experiment name will be: ", name + self.rand_id)

        name += self.rand_id
        return name

    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        return self.__class__.__name__

    def save_model(self, save_dir="./", suffix_name=""):
        pass

def get_true_ests(samples, featurizer):
    all_ests = []
    query_idx = 0
    for sample in samples:
        ests = {}
        node_keys = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in node_keys:
            node_keys.remove(SOURCE_NODE)
        node_keys.sort()
        for subq_idx, node in enumerate(node_keys):
            cards = sample["subset_graph"].nodes()[node]["cardinality"]
            alias_key = node
            est_card = cards["actual"]
            # idx = query_idx + subq_idx
            # est_card = featurizer.unnormalize(pred[idx], cards["total"])
            # assert est_card > 0
            ests[alias_key] = est_card

        all_ests.append(ests)
        query_idx += len(node_keys)
    return all_ests

def format_model_test_output_joinkey(pred, samples, featurizer):
    all_ests = []
    query_idx = 0

    for si, sample in enumerate(samples):
        ests = {}

        edge_keys = list(sample["subset_graph"].edges())
        edge_keys.sort(key = lambda x: str(x))

        subq_idx = 0
        for _, edge in enumerate(edge_keys):
            # cards = sample["subset_graph"].nodes()[node]["cardinality"]
            edgek = edge
            idx = query_idx + subq_idx
            est_card = featurizer.unnormalize(pred[idx], None)
            assert est_card >= 0
            ests[edgek] = est_card
            subq_idx += 1

        all_ests.append(ests)
        query_idx += subq_idx

    return all_ests

def format_model_test_output(pred, samples, featurizer, subplan_mask=None):
    '''
    @subplan_mask: optional, same format as QueryDataset's subplan_mask
    ([[list(node), ...], ...] per sample). Must be passed whenever `pred`
    came from a dataset that was itself built with this subplan_mask --
    otherwise the per-sample node count here won't match how many
    predictions `pred` actually has for that sample, and indexing into
    `pred` will drift/overrun.
    '''
    all_ests = []
    query_idx = 0
    # print("len pred: ", len(pred))

    for si, sample in enumerate(samples):
        ests = {}
        if subplan_mask is not None and si < len(subplan_mask):
            # same set of nodes QueryDataset kept for this sample; re-sort
            # since that's the order the dataset laid out its rows in
            node_keys = [tuple(node) for node in subplan_mask[si]]
        else:
            node_keys = list(sample["subset_graph"].nodes())
        if SOURCE_NODE in node_keys:
            node_keys.remove(SOURCE_NODE)
        node_keys.sort()

        subq_idx = 0
        for _, node in enumerate(node_keys):
            # if featurizer.max_num_tables != -1 and \
                # featurizer.max_num_tables < len(node):
                # # dummy estimate
                # ests[node] = 1.0
                # continue

            cards = sample["subset_graph"].nodes()[node]["cardinality"]
            alias_key = node
            idx = query_idx + subq_idx
            if "total" in cards:
                est_card = featurizer.unnormalize(pred[idx], cards["total"])
            else:
                est_card = featurizer.unnormalize(pred[idx], None)

            assert est_card > 0
            ests[alias_key] = est_card
            subq_idx += 1

        all_ests.append(ests)
        # query_idx += len(node_keys)
        query_idx += subq_idx

    return all_ests

class NN(CardinalityEstimationAlg):

    def __init__(self, model_params, *args, **kwargs):
        self.kwargs = kwargs
        for k, val in kwargs.items():
            self.__setattr__(k, val)

        for k, val in model_params.items():
            self.__setattr__(k, val)

        # Seed for training-time randomness (weight init, shuffling, dropout).
        # None when an alg is constructed outside main() -- notebooks, tests --
        # so fall back to the default rather than crashing.
        if getattr(self, "train_seed", None) is None:
            self.train_seed = DEFAULT_TRAIN_SEED
        self.train_seed = int(self.train_seed)

        # Dedicated RNG streams for the per-batch input transforms: random
        # onehot/feature masking (onehot_dropout) and the bitmap index
        # permutation (random_bitmap_idx). Both fire on every training batch.
        # Drawing them from the global numpy/torch state would make them
        # depend on every other draw in the process -- the periodic eval, a
        # visualization, an extra logging call -- so the masks would change
        # when eval_epoch changed, even at the same --train_seed. Private
        # generators make them a function of the seed and the batch index
        # only.
        self._mask_rng = make_numpy_generator(self.train_seed, "onehot_mask")
        self._batch_transform_gen = make_generator(
                self.train_seed, "batch_transform")
        self._test_bitmap_gen = make_generator(
                self.train_seed, "test_random_bitmap")

        # when estimates are log-normalized, then optimizing for mse is
        # basically equivalent to optimizing for q-error
        self.num_workers = 8
        if self.loss_func_name == "qloss":
            self.loss_func = qloss_torch
            self.load_query_together = False
        elif self.loss_func_name == "mse":
            self.loss_func = torch.nn.MSELoss(reduction="none")
            self.load_query_together = False
        elif self.loss_func_name == "mse_pos":
            self.loss_func = mse_pos
            self.load_query_together = False
        elif self.loss_func_name == "flowloss":
            self.loss_func = FlowLoss.apply
            self.load_query_together = True
            if self.mb_size > 16:
                self.mb_size = 1
            self.num_workers = 1
        elif self.loss_func_name == "mse+ranknet":
            self.loss_func = mse_ranknet
            self.load_query_together = True
            if self.mb_size > 16:
                self.mb_size = 1
        else:
            assert False

        if self.load_query_together:
            self.collate_fn = mscn_collate_fn_together
        elif ("mscn" not in self.__str__().lower()
                and "mstn" not in self.__str__().lower()):
            self.collate_fn = None
        else:
            if hasattr(self, "load_padded_mscn_feats"):
                if self.load_padded_mscn_feats:
                    self.collate_fn = None
                else:
                    self.collate_fn = mscn_collate_fn
            else:
                self.collate_fn = None

        self.eval_fn_handles = []
        for efn in self.eval_fns.split(","):
            if efn in ["planloss"]:
                print("skipping eval fn: ", efn)
                continue
            self.eval_fn_handles.append(get_eval_fn(efn))

    def init_net(self, sample):
        # Re-seed here rather than relying on the seeding done in main(): all
        # the featurization/split work in between consumes a variable amount
        # of RNG, so without this the initial weights would depend on e.g. how
        # many queries got loaded. Seeding at the point of construction makes
        # init a pure function of --train_seed, which is what lets two
        # architectures be compared at "the same" seed.
        seed_everything(derive_seed(self.train_seed, "net_init"))
        net = self._init_net(sample)
        print(net)

        if self.optimizer_name == "ams":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=True, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr,
                    amsgrad=False, weight_decay=self.weight_decay)
        elif self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(net.parameters(),
                    lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        else:
            assert False

        # if self.use_wandb:
            # wandb.watch(net)

        return net, optimizer

    def periodic_eval(self):
        if not self.use_wandb:
            return

        start = time.time()
        curerrs = {}

        for st, ds in self.eval_ds.items():
            # if st == "train":
                # continue
            samples = self.samples[st]

            preds, _ = self._eval_ds(ds, samples)

            if self.featurizer.card_type == "joinkey":
                preds1 = format_model_test_output_joinkey(preds,
                        samples, self.featurizer)
                preds = joinkey_cards_to_subplan_cards(samples, preds1,
                        "actual", 2)
                # def joinkey_cards_to_subplan_cards(samples, joinkey_cards,
                        # basecard_type, basecard_tables):

            else:
                preds = format_model_test_output(preds,
                        samples, self.featurizer)
                assert len(preds) == len(samples)

            # do evaluations
            for efunc in self.eval_fn_handles:
                if "Constraint" in str(efunc):
                    continue
                if "PostgresPlanCost-C" == str(efunc):
                    if self.true_costs[st] == 0:
                        truepreds = get_true_ests(samples, self.featurizer)
                        truecosts = efunc.eval(samples, truepreds,
                                args=None, samples_type=st,
                                result_dir=None,
                                query_dir = None,
                                user = self.featurizer.user,
                                db_name = self.featurizer.db_name,
                                db_host = self.featurizer.db_host,
                                port = self.featurizer.port,
                                pwd = self.featurizer.pwd,
                                num_processes = 16,
                                alg_name = self.__str__(),
                                save_pdf_plans=False,
                                use_wandb=False)
                        self.true_costs[st] = np.sum(truecosts)
                        truecost = np.sum(truecosts)
                    else:
                        truecost = self.true_costs[st]

                errors = efunc.eval(samples, preds,
                        args=None, samples_type=st,
                        result_dir=None,
                        user = self.featurizer.user,
                        query_dir = None,
                        db_name = self.featurizer.db_name,
                        db_host = self.featurizer.db_host,
                        pwd = self.featurizer.pwd,
                        port = self.featurizer.port,
                        num_processes = 16,
                        alg_name = self.__str__(),
                        save_pdf_plans=False,
                        use_wandb=False)

                if "PostgresPlanCost-C" == str(efunc):
                    assert truecost != 0.0
                    totcost = np.sum(errors)
                    relcost = totcost / truecost
                    key = str(efunc)+"-Relative-"+st
                    wandb.log({key: relcost, "epoch":self.epoch})
                    curerrs[key] = round(relcost,4)
                else:
                    err = np.mean(errors)
                    wandb.log({str(efunc)+"-"+st: err, "epoch":self.epoch})
                    curerrs[str(efunc)+"-"+st] = round(err,4)

                    median_err = np.median(errors)
                    p90 = np.percentile(errors, 90)
                    p99 = np.percentile(errors, 99)

                    wandb.log({str(efunc)+"-"+st+"-median": median_err,
                        "epoch":self.epoch})
                    # wandb.log({str(efunc)+"-"+st+"-90p": p90,
                        # "epoch":self.epoch})
                    # wandb.log({str(efunc)+"-"+st+"-99p": p99,
                        # "epoch":self.epoch})

                    curerrs[str(efunc)+"-"+st+"-median"] = round(median_err,4)
                    curerrs[str(efunc)+"-"+st+"-90p"] = round(p90,4)
                    curerrs[str(efunc)+"-"+st+"-99p"] = round(p99,4)

        if self.early_stopping == 2:
            self.all_errs.append(curerrs)

        print("Epoch ", self.epoch, curerrs)

    def update_flow_training_info(self):
        fstart = time.time()
        # precompute a whole bunch of training things
        self.flow_training_info = []
        # farchive = klepto.archives.dir_archive("./flow_info_archive",
                # cached=True, serialized=True)
        # farchive.load()
        new_seen = False
        for sample in self.training_samples:
            qkey = deterministic_hash(sample["sql"])
            # if qkey in farchive:
            if False:
                subsetg_vectors = farchive[qkey]
                assert len(subsetg_vectors) == 10
            else:
                new_seen = True
                subsetg_vectors = list(get_subsetg_vectors(sample,
                    self.cost_model))

            true_cards = np.zeros(len(subsetg_vectors[0]),
                    dtype=np.float32)
            nodes = list(sample["subset_graph"].nodes())

            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)

            nodes.sort()
            for i, node in enumerate(nodes):
                true_cards[i] = \
                    sample["subset_graph"].nodes()[node]["cardinality"]["actual"]

            trueC_vec, dgdxT, G, Q = \
                get_optimization_variables(true_cards,
                    subsetg_vectors[0], self.featurizer.min_val,
                        self.featurizer.max_val,
                        self.featurizer.ynormalization,
                        subsetg_vectors[4],
                        subsetg_vectors[5],
                        subsetg_vectors[3],
                        subsetg_vectors[1],
                        subsetg_vectors[2],
                        subsetg_vectors[6],
                        subsetg_vectors[7],
                        self.cost_model, subsetg_vectors[-1])

            Gv = to_variable(np.zeros(len(subsetg_vectors[0]))).float()
            Gv[subsetg_vectors[-2]] = 1.0
            trueC_vec = to_variable(trueC_vec).float()
            dgdxT = to_variable(dgdxT).float()
            G = to_variable(G).float()
            Q = to_variable(Q).float()

            trueC = torch.eye(len(trueC_vec)).float().detach()
            for i, curC in enumerate(trueC_vec):
                trueC[i,i] = curC

            invG = torch.inverse(G)
            v = invG @ Gv
            left = (Gv @ torch.transpose(invG,0,1)) @ torch.transpose(Q, 0, 1)
            right = Q @ (v)
            left = left.detach().cpu()
            right = right.detach().cpu()
            opt_flow_loss = left @ trueC @ right
            del trueC

            # print(opt_flow_loss)
            # pdb.set_trace()

            self.flow_training_info.append((subsetg_vectors, trueC_vec,
                    opt_flow_loss))

        print("precomputing flow info took: ", time.time()-fstart)

    def _collect_target_queries(self, kwargs):
        target_samples = []
        if "evalqs" not in kwargs or kwargs["evalqs"] is None:
            return target_samples

        evalqs = kwargs["evalqs"]
        if len(evalqs) > 0 and isinstance(evalqs[0], list):
            for cur_evalqs in evalqs:
                target_samples.extend(cur_evalqs)
        else:
            target_samples = list(evalqs)

        return target_samples

    def _latent_visualization_enabled(self):
        return bool(getattr(self, "visualize_latents", False))

    def _latent_visualization_epoch_frequency(self):
        return int(getattr(self, "latent_viz_every_epochs", 0) or 0)

    def _latent_visualization_sample_limit(self):
        return max(1, int(getattr(self, "latent_viz_max_points", 2000) or 2000))

    def _latent_visualization_method(self):
        """Get the dimensionality reduction method for latent visualization.
        
        Options: 'pca' or 'tsne'. Defaults to 'pca'.
        """
        method = str(getattr(self, "latent_viz_method", "pca") or "pca").lower()
        if method not in ["pca", "tsne"]:
            method = "pca"
        return method

    def _prepare_run_plot_dir(self, kwargs):
        run_result_dir = kwargs.get("result_dir", "./results")
        return os.path.join(run_result_dir, self.get_exp_name())

    def _ensure_latent_visualization_ready(self):
        if not self._latent_visualization_enabled():
            return False

        if hasattr(self.net, "configure_auxiliary_components"):
            self.net.configure_auxiliary_components(enable_latent_interface=True)

        if not hasattr(self.net, "get_latent_visualization_tensors"):
            raise RuntimeError(
                "Latent visualization requires an MSCN-style network with "
                "get_latent_visualization_tensors()."
            )
        return True

    def _build_latent_visualization_loader(self, samples):
        if samples is None or len(samples) == 0:
            return None

        ds = self.init_dataset(
            samples,
            self.load_query_together,
            max_num_tables=self.max_num_tables,
            load_padded_mscn_feats=self.load_padded_mscn_feats,
        )
        batch_size = 1 if self.load_query_together else self.mb_size
        return data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def _setup_latent_visualization_loaders(self, target_samples=None):
        self.latent_viz_source_loader = getattr(self, "trainloader", None)

        self.latent_viz_eval_loaders = {}
        for eval_name, eval_ds in getattr(self, "eval_ds", {}).items():
            eval_loader = self._build_eval_dataset_loader(eval_ds)
            if eval_loader is not None:
                self.latent_viz_eval_loaders[eval_name] = eval_loader

        target_loader = getattr(self, "target_loader", None)
        if target_loader is None and target_samples is not None and len(target_samples) > 0:
            target_loader = self._build_latent_visualization_loader(target_samples)
        self.latent_viz_target_loader = target_loader

    def _build_eval_dataset_loader(self, ds):
        if ds is None:
            return None

        batch_size = 1 if self.load_query_together else self.mb_size
        return data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def _latent_mmd_enabled(self):
        return bool(getattr(self, "track_latent_mmd", True))

    def _latent_mmd_sample_limit(self):
        default_limit = getattr(self, "latent_viz_max_points", 2000)
        configured_limit = getattr(self, "latent_mmd_max_points", default_limit)
        return max(1, int(configured_limit or default_limit))

    def _ensure_latent_mmd_ready(self):
        if not self._latent_mmd_enabled():
            return False
        if not hasattr(self, "net"):
            return False

        if hasattr(self.net, "configure_auxiliary_components"):
            self.net.configure_auxiliary_components(enable_latent_interface=True)

        return hasattr(self.net, "forward_with_latent") and \
            hasattr(self.net, "compute_mmd")

    def _collect_latent_embeddings(self, loader, max_points):
        if loader is None:
            return None
        if not self._ensure_latent_mmd_ready():
            return None

        net = self.net
        was_training = net.training
        net.eval()
        collected = []
        num_rows = 0

        # Set fixed seed for deterministic sampling across epochs
        if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(42)  # Fixed seed for reproducibility

        with torch.no_grad():
            for xbatch, _, _ in loader:
                _, z = net.forward_with_latent(xbatch)
                if z.ndim == 1:
                    z = z.reshape(1, -1)

                remaining = max_points - num_rows
                if remaining <= 0:
                    break

                take = min(remaining, z.shape[0])
                collected.append(z[:take].detach().cpu())
                num_rows += take

                if num_rows >= max_points:
                    break

        if was_training:
            net.train()

        if len(collected) == 0:
            return None

        return torch.cat(collected, dim=0)

    def compute_latent_mmd(self, source_loader=None, target_loader=None):
        if not self._ensure_latent_mmd_ready():
            return None

        if source_loader is None:
            source_loader = getattr(self, "trainloader", None)
        if target_loader is None:
            target_loader = getattr(self, "target_loader", None)
        if target_loader is None:
            target_loader = getattr(self, "latent_viz_target_loader", None)

        if source_loader is None or target_loader is None:
            return None

        max_points = self._latent_mmd_sample_limit()
        z_source = self._collect_latent_embeddings(source_loader, max_points)
        z_target = self._collect_latent_embeddings(target_loader, max_points)
        if z_source is None or z_target is None:
            return None
        if z_source.shape[0] == 0 or z_target.shape[0] == 0:
            return None

        return self.net.compute_mmd(
            z_source,
            z_target,
            detach=True,
            as_float=True,
        )

    def compute_eval_latent_mmds(self, source_loader=None):
        if source_loader is None:
            source_loader = getattr(self, "trainloader", None)
        if source_loader is None:
            return {}

        eval_mmds = {}
        for eval_name, eval_ds in self.eval_ds.items():
            target_loader = self._build_eval_dataset_loader(eval_ds)
            if target_loader is None:
                continue

            latent_mmd = self.compute_latent_mmd(
                source_loader=source_loader,
                target_loader=target_loader,
            )
            if latent_mmd is not None:
                eval_mmds[eval_name] = latent_mmd

        return eval_mmds

    def _format_eval_latent_mmds(self, eval_latent_mmds):
        return {
            key: round(val, 6)
            for key, val in eval_latent_mmds.items()
        }

    def _collect_latent_visualization_views(self, loader, max_points):
        if loader is None:
            return {}

        net = self.net
        # Visualization should not change the model's training mode permanently.
        # We switch to eval() only for deterministic feature extraction, then
        # restore train() if the caller was in training mode.
        was_training = net.training
        net.eval()
        collected = defaultdict(list)
        num_rows = 0

        # Set fixed seed for deterministic sampling across epochs
        if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(42)  # Fixed seed for reproducibility

        with torch.no_grad():
            for xbatch, _, _ in loader:
                views = net.get_latent_visualization_tensors(xbatch)
                batch_rows = None
                converted = {}
                for key, tensor in views.items():
                    arr = tensor.detach().cpu().numpy()
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    converted[key] = arr
                    if batch_rows is None:
                        batch_rows = arr.shape[0]

                if batch_rows is None or batch_rows == 0:
                    continue

                remaining = max_points - num_rows
                if remaining <= 0:
                    break

                take = min(remaining, batch_rows)
                for key, arr in converted.items():
                    collected[key].append(arr[:take])
                num_rows += take

                if num_rows >= max_points:
                    break

        if was_training:
            net.train()

        return {
            key: np.concatenate(val, axis=0)
            for key, val in collected.items() if len(val) > 0
        }

    def _project_latent_view(self, source_view, target_view):
        arrays = []
        source_count = 0
        target_count = 0

        if source_view is not None and len(source_view) > 0:
            arrays.append(source_view)
            source_count = len(source_view)
        if target_view is not None and len(target_view) > 0:
            arrays.append(target_view)
            target_count = len(target_view)

        if len(arrays) == 0:
            return None, None

        combined = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
        combined = combined - combined.mean(axis=0, keepdims=True)

        method = self._latent_visualization_method()
        
        if method == "tsne":
            projected = self._project_with_tsne(combined)
        else:  # default to pca
            projected = self._project_with_pca(combined)

        source_proj = projected[:source_count] if source_count > 0 else None
        target_proj = projected[source_count:source_count + target_count] \
            if target_count > 0 else None
        return source_proj, target_proj
    
    def _project_with_pca(self, combined):
        """Project data to 2D using PCA (SVD)."""
        if combined.shape[1] == 1:
            projected = np.concatenate(
                [combined, np.zeros((combined.shape[0], 1), dtype=combined.dtype)],
                axis=1,
            )
        else:
            _, _, vt = np.linalg.svd(combined, full_matrices=False)
            basis = vt[:2].T
            projected = combined @ basis
            if projected.shape[1] == 1:
                projected = np.concatenate(
                    [projected, np.zeros((projected.shape[0], 1), dtype=projected.dtype)],
                    axis=1,
                )
        return projected
    
    def _project_with_tsne(self, combined):
        """Project data to 2D using t-SNE."""
        try:
            tsne = TSNE(n_components=2, perplexity=30.0, random_state=42)
            projected = tsne.fit_transform(combined)
            return projected.astype(np.float32)
        except Exception as e:
            print(f"Warning: t-SNE projection failed ({e}). Falling back to PCA.")
            return self._project_with_pca(combined)

    def _save_latent_visualization(self, save_dir, tag="final"):
        if not self._latent_visualization_enabled():
            return None
        if not self._ensure_latent_visualization_ready():
            return None

        max_points = self._latent_visualization_sample_limit()
        source_views = self._collect_latent_visualization_views(
            getattr(self, "latent_viz_source_loader", None), max_points)

        if len(source_views) == 0:
            return None

        target_loaders = getattr(self, "latent_viz_eval_loaders", {})
        if len(target_loaders) == 0:
            fallback_loader = getattr(self, "latent_viz_target_loader", None)
            if fallback_loader is None:
                return None
            target_loaders = {"target": fallback_loader}

        split_mmds = getattr(self, "latest_eval_latent_mmds", {}) or {}

        source_view = source_views.get("out_mlp1_input")
        if source_view is None or len(source_view) == 0:
            return None

        os.makedirs(save_dir, exist_ok=True)
        plot_paths = []
        for split_name, target_loader in target_loaders.items():
            target_views = self._collect_latent_visualization_views(
                target_loader, max_points)
            target_view = target_views.get("out_mlp1_input")
            if target_view is None or len(target_view) == 0:
                continue

            source_proj, target_proj = self._project_latent_view(
                source_view,
                target_view,
            )
            if source_proj is None and target_proj is None:
                continue

            fig, ax = plt.subplots(1, 1, figsize=(7, 6))
            if source_proj is not None and len(source_proj) > 0:
                ax.scatter(
                    source_proj[:, 0], source_proj[:, 1],
                    s=10, alpha=0.55, label="source", color="#1f77b4",
                )
            if target_proj is not None and len(target_proj) > 0:
                ax.scatter(
                    target_proj[:, 0], target_proj[:, 1],
                    s=10, alpha=0.55, label=split_name, color="#ff7f0e",
                )

            mmd_value = split_mmds.get(split_name)
            if mmd_value is None:
                caption = f"MMD(source, {split_name}) = n/a"
            else:
                caption = f"MMD(source, {split_name}) = {mmd_value:.6f}"

            ax.set_title(f"Latent space: source vs {split_name}")
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            ax.grid(True, alpha=0.2)
            ax.legend()
            fig.text(0.5, 0.01, caption, ha="center", fontsize=10)
            fig.tight_layout(rect=(0, 0.04, 1, 1))

            plot_path = os.path.join(save_dir, f"latent_views_{split_name}_{tag}.png")
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            plot_paths.append(plot_path)

        if len(plot_paths) == 0:
            return None
        return plot_paths[-1]

    def _maybe_save_latent_visualization(self, save_dir, force=False, tag=None):
        if not self._latent_visualization_enabled():
            return None

        if not force:
            freq = self._latent_visualization_epoch_frequency()
            if freq <= 0 or self.epoch % freq != 0:
                return None
            if tag is None:
                tag = "epoch{:04d}".format(self.epoch)
        elif tag is None:
            tag = "final_epoch{:04d}".format(self.epoch)

        plot_path = self._save_latent_visualization(
            save_dir,
            tag=tag,
        )
        if plot_path is not None:
            print("Saved latent visualization to:", plot_path)
        return plot_path

    def _slice_batch_inputs(self, xbatch, batch_size):
        if torch.is_tensor(xbatch):
            return xbatch[:batch_size]

        sliced = {}
        for k, v in xbatch.items():
            if torch.is_tensor(v):
                sliced[k] = v[:batch_size]
            else:
                sliced[k] = v
        return sliced

    def _slice_batch_info(self, info, batch_size):
        if isinstance(info, dict):
            sliced = {}
            for k, v in info.items():
                if torch.is_tensor(v):
                    sliced[k] = v[:batch_size]
                elif isinstance(v, np.ndarray):
                    sliced[k] = v[:batch_size]
                elif hasattr(v, "__len__"):
                    sliced[k] = v[:batch_size]
                else:
                    sliced[k] = v
            return sliced

        if isinstance(info, list):
            return info[:batch_size]

        return info

    def _apply_training_batch_transforms(self, xbatch):
        if self.random_bitmap_idx and "join" in xbatch:
            idxs = torch.randperm(xbatch["join"].shape[-1],
                    generator=self._batch_transform_gen)
            xbatch["join"] = xbatch["join"][:, :, idxs]

        if self.onehot_dropout == 0:
            return xbatch

        if not self.onehot_dropout:
            return xbatch

        if self.featurizer.featurization_type == "combined":
            mask = np.zeros(xbatch.shape[1])
            mask[-1] = 1
            mask[-3] = 1
            mask[-4] = 1
            mask = self._get_onehot_mask(mask)
            xbatch = xbatch * mask
            return xbatch

        tf_mask = self._get_onehot_mask(self.featurizer.table_onehot_mask)
        jf_mask = self._get_onehot_mask(self.featurizer.join_onehot_mask)
        pf_mask = self._get_onehot_mask(self.featurizer.pred_onehot_mask)

        if self.featurizer.pred_features:
            xbatch["pred"] = xbatch["pred"] * pf_mask
        if self.featurizer.join_features:
            xbatch["join"] = xbatch["join"] * jf_mask
        if self.featurizer.table_features:
            xbatch["table"] = xbatch["table"] * tf_mask
        return xbatch

    def _ensure_latent_discriminator_ready(self):
        enable_decoder = bool(getattr(self, "enable_decoder", False))
        if hasattr(self.net, "configure_auxiliary_components"):
            self.net.configure_auxiliary_components(
                enable_latent_interface=True,
                enable_discriminator=True,
                enable_decoder=enable_decoder,
            )

        if getattr(self.net, "discriminator", None) is None:
            hidden_dims = getattr(self.net, "discriminator_hidden_dims", None)
            dropout = getattr(self.net, "discriminator_dropout", 0.1)
            self.net.register_auxiliary_modules(
                discriminator=LatentDiscriminator(
                    self.net.latent_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                ).to(device)
            )

        self._ensure_decoder_ready()

    def _ensure_latent_generator_ready(self):
        enable_decoder = bool(getattr(self, "enable_decoder", False))
        if hasattr(self.net, "configure_auxiliary_components"):
            self.net.configure_auxiliary_components(
                enable_latent_interface=True,
                enable_discriminator=True,
                enable_generator=True,
                enable_decoder=enable_decoder,
            )

        if getattr(self.net, "generator", None) is None:
            noise_dim = getattr(self.net, "generator_noise_dim", None)
            if noise_dim is None:
                noise_dim = self.net.latent_dim
            hidden_dims = getattr(self.net, "generator_hidden_dims", None)
            dropout = getattr(self.net, "generator_dropout", 0.1)
            self.net.register_auxiliary_modules(
                generator=LatentGenerator(
                    noise_dim,
                    self.net.latent_dim,
                    hidden_dims=hidden_dims,
                    dropout=dropout,
                ).to(device)
            )

        self._ensure_decoder_ready()

    def _ensure_decoder_ready(self):
        if not bool(getattr(self, "enable_decoder", False)):
            return

        if hasattr(self.net, "configure_auxiliary_components"):
            self.net.configure_auxiliary_components(
                enable_latent_interface=True,
                enable_decoder=True,
            )

        if getattr(self.net, "decoder", None) is not None:
            return

        decoder_output_dim = getattr(self.net, "decoder_output_dim", None)
        if decoder_output_dim is None:
            raise RuntimeError(
                "Decoder output dimension is undefined. Set decoder_output_dim "
                "or enable load_padded_mscn_feats so it can be inferred."
            )

        hidden_dims = getattr(self.net, "decoder_hidden_dims", None)
        dropout = getattr(self.net, "decoder_dropout", 0.0)
        self.net.register_auxiliary_modules(
            decoder=Decoder(
                self.net.latent_dim,
                decoder_output_dim,
                hidden_dims=hidden_dims,
                dropout=dropout,
            ).to(device)
        )

    def _decoder_loss_enabled(self):
        return bool(getattr(self, "enable_decoder", False)) and \
            hasattr(self.net, "decode") and getattr(self.net, "decoder", None) is not None

    def _compute_decoder_reconstruction_loss(self, z_source, xbatch_source,
            z_target, xbatch_target):
        target_source = self.net.get_decoder_target(xbatch_source).detach()
        target_target = self.net.get_decoder_target(xbatch_target).detach()

        recon_source = self.net.decode(z_source)
        recon_target = self.net.decode(z_target)

        loss_recon_source = torch.nn.functional.mse_loss(
            recon_source, target_source)
        loss_recon_target = torch.nn.functional.mse_loss(
            recon_target, target_target)
        loss_recon = loss_recon_source + loss_recon_target

        if not hasattr(self, "_decoder_debug_printed"):
            print(
                "[DECODER DEBUG] enabled={}, decoder_is_none={}, "
                "z_source_shape={}, recon_source_shape={}, target_source_shape={}, "
                "loss_recon_source={:.12e}, loss_recon_target={:.12e}, "
                "loss_recon_total={:.12e}".format(
                    bool(getattr(self, "enable_decoder", False)),
                    getattr(self.net, "decoder", None) is None,
                    tuple(z_source.shape),
                    tuple(recon_source.shape),
                    tuple(target_source.shape),
                    loss_recon_source.item(),
                    loss_recon_target.item(),
                    loss_recon.item(),
                )
            )
            self._decoder_debug_printed = True

        return loss_recon, loss_recon_source, loss_recon_target

    def _build_optimizer_for_params(self, params, lr):
        if len(params) == 0:
            raise ValueError("Cannot build optimizer for an empty parameter list.")

        if self.optimizer_name == "ams":
            return torch.optim.Adam(
                params, lr=lr, amsgrad=True, weight_decay=self.weight_decay
            )
        if self.optimizer_name in ["adam", "adamw"]:
            return torch.optim.Adam(
                params, lr=lr, amsgrad=False, weight_decay=self.weight_decay
            )
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=self.weight_decay
            )
        raise ValueError(f"Unsupported optimizer_name: {self.optimizer_name}")

    def _init_new_discriminator_optimizers(self):
        named_params = [(name, param) for name, param in self.net.named_parameters()
                if param.requires_grad]

        discriminator_params = [p for n, p in named_params if n.startswith("discriminator.")]
        decoder_params = [p for n, p in named_params if n.startswith("decoder.")]
        regressor_params = [p for n, p in named_params
                if n.startswith("out_mlp2.") and not n.startswith("discriminator.")]
        encoder_params = [p for n, p in named_params
                if not n.startswith("discriminator.")
                and not n.startswith("out_mlp2.")
                and not n.startswith("decoder.")]

        if len(discriminator_params) == 0:
            raise RuntimeError(
                "Latent discriminator parameters were not found. "
                "Make sure enable_discriminator is active."
            )
        if len(regressor_params) == 0:
            raise RuntimeError("Regressor parameters (out_mlp2.*) were not found.")
        if len(encoder_params) == 0:
            raise RuntimeError("Encoder parameters were not found.")

        reg_lr = getattr(self, "adv_regression_lr", self.lr)
        disc_lr = getattr(self, "adv_discriminator_lr", self.lr)
        gen_lr = getattr(self, "adv_generator_lr", self.lr)

        self.opt_regression = self._build_optimizer_for_params(
            encoder_params + regressor_params + decoder_params, reg_lr)
        self.opt_discriminator = self._build_optimizer_for_params(
            discriminator_params, disc_lr)
        self.opt_generator = self._build_optimizer_for_params(
            encoder_params, gen_lr)

        self.optimizer = self.opt_regression
        self.bce_loss = torch.nn.BCELoss()

        if bool(getattr(self, "enable_decoder", False)):
            print(
                "[DECODER DEBUG] optimizer setup: decoder_param_tensors={}, "
                "decoder_output_dim={}".format(
                    len(decoder_params),
                    getattr(self.net, "decoder_output_dim", None),
                )
            )

    def _init_latent_generator_optimizers(self):
        named_params = [
            (name, param) for name, param in self.net.named_parameters()
            if param.requires_grad
        ]

        discriminator_params = [p for n, p in named_params if n.startswith("discriminator.")]
        generator_params = [p for n, p in named_params if n.startswith("generator.")]
        decoder_params = [p for n, p in named_params if n.startswith("decoder.")]
        regressor_params = [
            p for n, p in named_params
            if n.startswith("out_mlp2.") and not n.startswith("discriminator.")
        ]
        encoder_params = [
            p for n, p in named_params
            if not n.startswith("discriminator.")
            and not n.startswith("generator.")
            and not n.startswith("out_mlp2.")
            and not n.startswith("decoder.")
        ]

        if len(discriminator_params) == 0:
            raise RuntimeError(
                "Latent discriminator parameters were not found. "
                "Make sure enable_discriminator is active."
            )
        if len(generator_params) == 0:
            raise RuntimeError(
                "Latent generator parameters were not found. "
                "Make sure enable_generator is active."
            )
        if len(regressor_params) == 0:
            raise RuntimeError("Regressor parameters (out_mlp2.*) were not found.")
        if len(encoder_params) == 0:
            raise RuntimeError("Encoder parameters were not found.")

        reg_lr = getattr(self, "adv_regression_lr", self.lr)
        disc_lr = getattr(self, "adv_discriminator_lr", self.lr)
        # Per request: generator learning rate matches discriminator learning rate.
        gen_lr = disc_lr

        self.opt_regression = self._build_optimizer_for_params(
            encoder_params + regressor_params + decoder_params, reg_lr
        )
        self.opt_discriminator = self._build_optimizer_for_params(
            discriminator_params, disc_lr
        )
        self.opt_latent_generator = self._build_optimizer_for_params(
            generator_params, gen_lr
        )

        self.optimizer = self.opt_regression
        self.bce_loss = torch.nn.BCELoss()

    def _compute_regression_loss(self, pred, ybatch, info):
        if self.loss_func_name == "flowloss":
            raise RuntimeError(
                "train_with_new_discriminator does not support flowloss. "
                "Use mse/qloss variants for this training mode."
            )

        if self.loss_func_name == "qloss" and self.featurizer.ynormalization == "log":
            pred_unnorm = self.featurizer.unnormalize_torch(pred, None)
            ybatch_unnorm = self.featurizer.unnormalize_torch(ybatch, None)
            losses = self.loss_func(pred_unnorm, ybatch_unnorm)
            return self._reduce_losses(losses, info)

        losses = self.loss_func(pred, ybatch)
        return self._reduce_losses(losses, info)

    def _compute_mean_loss_from_numpy(self, preds, ys):
        if self.loss_func_name == "flowloss":
            return None

        pred_t = torch.from_numpy(preds).float()
        y_t = torch.from_numpy(ys).float()

        if self.loss_func_name == "qloss" and self.featurizer.ynormalization == "log":
            pred_t = self.featurizer.unnormalize_torch(pred_t, None)
            y_t = self.featurizer.unnormalize_torch(y_t, None)

        losses = self.loss_func(pred_t, y_t)
        if torch.is_tensor(losses):
            if losses.ndim == 0:
                return float(losses.item())
            return float(torch.mean(losses).item())
        return float(np.mean(losses))

    def _compute_loss_metrics_from_numpy(self, preds, ys):
        """Compute mean, median, and 99th percentile of loss/Q-Error."""
        if self.loss_func_name == "flowloss":
            return {"mean": None, "median": None, "p99": None}

        pred_t = torch.from_numpy(preds).float()
        y_t = torch.from_numpy(ys).float()

        if self.loss_func_name == "qloss" and self.featurizer.ynormalization == "log":
            pred_t = self.featurizer.unnormalize_torch(pred_t, None)
            y_t = self.featurizer.unnormalize_torch(y_t, None)

        losses = self.loss_func(pred_t, y_t)
        if torch.is_tensor(losses):
            if losses.ndim == 0:
                losses = torch.tensor([losses.item()])
            losses_np = losses.detach().cpu().numpy()
        else:
            losses_np = np.asarray(losses)
        
        return {
            "mean": float(np.mean(losses_np)) if len(losses_np) > 0 else None,
            "median": float(np.median(losses_np)) if len(losses_np) > 0 else None,
            "p99": float(np.percentile(losses_np, 99)) if len(losses_np) > 0 else None,
        }

    def _compute_qerror_metrics_from_numpy(self, preds, samples, subplan_mask=None):
        if samples is None or len(samples) == 0:
            return {"mean": None, "median": None, "p99": None}

        if getattr(self.featurizer, "card_type", None) == "joinkey":
            # NOTE: format_model_test_output_joinkey doesn't support
            # subplan_mask (it indexes by subset_graph edges, not nodes,
            # so the node-based mask format doesn't directly apply here).
            # preds must have come from an unmasked dataset in this mode.
            formatted_preds = format_model_test_output_joinkey(
                preds,
                samples,
                self.featurizer,
            )
            errors = get_eval_fn("qerr_joinkey").eval(samples, formatted_preds)
        else:
            formatted_preds = format_model_test_output(
                preds,
                samples,
                self.featurizer,
                subplan_mask=subplan_mask,
            )
            errors = get_eval_fn("qerr").eval(
                samples,
                formatted_preds,
                result_dir=None,
            )

        if len(errors) == 0:
            return {"mean": None, "median": None, "p99": None}

        return {
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p99": float(np.percentile(errors, 99)),
        }

    def _compute_train_qerror_metrics(self, subplan_mask=None):
        '''
        Q-error over the training queries, so the val q-error curve can be read
        against it (fit vs. generalization).

        Reuses self.trainds instead of featurizing a second copy of the
        training set -- QueryDataset keeps all feature vectors in memory, so a
        dedicated "train" eval dataset would double that. The flip side is that
        trainds' rows only line up with format_model_test_output's per-sample
        node ordering when it was built one-row-per-subplan and without the
        max_num_tables filter (which silently drops nodes that
        format_model_test_output still walks over), so bail out otherwise
        rather than report drifted numbers.
        '''
        skipped = {"mean": None, "median": None, "p99": None}

        if getattr(self, "trainds", None) is None:
            return skipped
        if getattr(self, "training_samples", None) is None:
            return skipped
        if self.load_query_together or self.max_num_tables != -1:
            return skipped

        train_preds, _ = self._eval_ds(self.trainds, self.training_samples)
        return self._compute_qerror_metrics_from_numpy(
            train_preds,
            self.training_samples,
            subplan_mask=subplan_mask,
        )

    def _compute_eval_qerror_metrics(self):
        '''
        Q-error on every registered eval dataset -- the held-out test split
        plus each eval workload (JOB, CEB-IMDb, ...) -- keyed by the same name
        self.eval_ds uses. "val" is skipped: it gets its own curve from the
        caller, and re-doing it here would just be a second forward pass.

        Note this does its own forward passes rather than piggybacking on
        periodic_eval(): that one returns early unless wandb is on, and it
        formats predictions without each dataset's subplan_mask, so its
        numbers are only right for unmasked eval sets.
        '''
        eval_qerrs = {}

        for name, ds in getattr(self, "eval_ds", {}).items():
            if name == "val":
                continue

            samples = self.samples.get(name)
            if samples is None or len(samples) == 0:
                continue

            preds, _ = self._eval_ds(ds, samples)
            metrics = self._compute_qerror_metrics_from_numpy(
                preds,
                samples,
                subplan_mask=getattr(self, "eval_subplan_masks", {}).get(name),
            )
            if metrics["mean"] is None:
                continue

            eval_qerrs[name] = metrics

        return eval_qerrs

    def _format_eval_qerrs(self, eval_qerrs):
        return {
            name: "{:.4f}(median={:.4f}, 99p={:.4f})".format(
                metrics["mean"], metrics["median"], metrics["p99"])
            for name, metrics in eval_qerrs.items()
        }

    def _save_adversarial_training_plot(self, save_dir):
        if not hasattr(self, "adversarial_train_history"):
            return None
        if len(self.adversarial_train_history) == 0:
            return None

        os.makedirs(save_dir, exist_ok=True)

        epochs = np.arange(len(self.adversarial_train_history))
        reg_loss = [m["loss_reg"] for m in self.adversarial_train_history]
        recon_loss = [m.get("loss_recon", np.nan) for m in self.adversarial_train_history]
        phase1_loss = [m.get("loss_phase1", np.nan) for m in self.adversarial_train_history]
        disc_loss = [m["loss_d"] for m in self.adversarial_train_history]
        gen_loss = [m["loss_g"] for m in self.adversarial_train_history]
        val_loss = [m.get("val_loss", np.nan) for m in self.adversarial_train_history]
        disc_acc = [m["disc_acc"] for m in self.adversarial_train_history]
        source_acc = [m["disc_acc_source"] for m in self.adversarial_train_history]
        target_acc = [m["disc_acc_target"] for m in self.adversarial_train_history]
        fool_acc = [m["gen_fool_acc"] for m in self.adversarial_train_history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))

        axes[0].plot(epochs, reg_loss, label="Regression Loss")
        if np.isfinite(np.asarray(recon_loss)).any():
            axes[0].plot(epochs, recon_loss, label="Reconstruction Loss")
        if np.isfinite(np.asarray(phase1_loss)).any():
            axes[0].plot(epochs, phase1_loss, label="Phase 1 Total Loss")
        if np.isfinite(np.asarray(val_loss)).any():
            axes[0].plot(
                epochs,
                val_loss,
                label="Validation Loss",
                linestyle="--",
                marker="o",
                markersize=3,
            )
        axes[0].set_title("Regression And Reconstruction")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epochs, disc_loss, label="Discriminator Loss")
        axes[1].plot(epochs, gen_loss, label="Generator Loss")
        axes[1].set_title("Adversarial Losses")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(epochs, disc_acc, label="Disc Acc")
        axes[2].plot(epochs, source_acc, label="Source Acc")
        axes[2].plot(epochs, target_acc, label="Target Acc")
        axes[2].plot(epochs, fool_acc, label="Fool Acc")
        axes[2].set_title("Adversarial Accuracy")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Accuracy")
        axes[2].set_ylim(0.0, 1.0)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "adversarial_training_curves.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def _save_standard_training_plot(self, save_dir):
        if not hasattr(self, "standard_train_history"):
            return None
        if len(self.standard_train_history) == 0:
            return None

        os.makedirs(save_dir, exist_ok=True)

        epochs = np.arange(len(self.standard_train_history))
        train_loss = [m.get("train_loss", np.nan) for m in self.standard_train_history]
        val_loss = [m.get("val_loss", np.nan) for m in self.standard_train_history]

        train_mask = np.isfinite(np.asarray(train_loss))
        val_mask = np.isfinite(np.asarray(val_loss))

        fig, ax = plt.subplots(1, 1, figsize=(9, 4))

        if train_mask.any():
            ax.plot(
                epochs[train_mask],
                np.asarray(train_loss)[train_mask],
                label="Train Loss",
                color="#1f77b4",
            )
        if val_mask.any():
            ax.plot(
                epochs[val_mask],
                np.asarray(val_loss)[val_mask],
                label="Validation Loss",
                color="#ff7f0e",
                linestyle="--",
                marker="o",
                markersize=3,
            )

        ax.set_title("Training Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(save_dir, "standard_training_curves.png")
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def _save_latent_mmd_plot(self, history, save_dir, plot_name, title):
        if history is None or len(history) == 0:
            return None

        series_names = []
        for metrics in history:
            eval_mmds = metrics.get("eval_latent_mmds", {})
            for name in eval_mmds.keys():
                if name not in series_names:
                    series_names.append(name)

        if len(series_names) == 0:
            return None

        os.makedirs(save_dir, exist_ok=True)
        epochs = np.arange(len(history))
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))

        for series_name in series_names:
            values = []
            for metrics in history:
                eval_mmds = metrics.get("eval_latent_mmds", {})
                values.append(eval_mmds.get(series_name, np.nan))

            values = np.asarray(values, dtype=np.float64)
            mask = np.isfinite(values)
            if not mask.any():
                continue

            ax.plot(
                epochs[mask],
                values[mask],
                marker="o",
                linestyle="--",
                linewidth=2,
                markersize=3,
                label=series_name,
            )

        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MMD")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plot_path = os.path.join(save_dir, plot_name)
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    # Colors for the q-error splits. Train/val are pinned so they keep the same
    # color across runs; eval datasets take the rest in registration order.
    QERR_SPLIT_COLORS = [
        "#ff7f0e", "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    ]

    def _collect_qerror_series(self, history):
        '''
        [(label, {stat_key: values_array})] for train, val and every eval
        dataset that recorded a q-error, in a stable order.
        '''
        series = []

        for split_key, label in (("train", "Train"), ("val", "Val")):
            series.append((label, {
                "mean": [m.get(split_key + "_qerr", np.nan) for m in history],
                "median": [m.get(split_key + "_qerr_median", np.nan)
                    for m in history],
                "p99": [m.get(split_key + "_qerr_p99", np.nan)
                    for m in history],
            }))

        # eval datasets come and go by name; take them in first-seen order so
        # colors stay put from epoch to epoch
        eval_names = []
        for metrics in history:
            for name in metrics.get("eval_qerrs", {}):
                if name not in eval_names:
                    eval_names.append(name)

        for name in eval_names:
            per_stat = {"mean": [], "median": [], "p99": []}
            for metrics in history:
                cur = metrics.get("eval_qerrs", {}).get(name, {})
                for stat in per_stat:
                    per_stat[stat].append(cur.get(stat, np.nan))
            # "test" -> "Test", but leave names like JOB / CEB-IMDb-Complex be
            label = name.capitalize() if name.islower() else name
            series.append((label, per_stat))

        out = []
        for label, per_stat in series:
            arrays = {stat: np.asarray(vals, dtype=np.float32)
                    for stat, vals in per_stat.items()}
            if not any(np.isfinite(a).any() for a in arrays.values()):
                continue
            out.append((label, arrays))

        return out

    def _save_qerror_plot(self, history, save_dir, plot_name, title):
        if history is None or len(history) == 0:
            return None

        series = self._collect_qerror_series(history)
        if len(series) == 0:
            return None

        # One panel per statistic, one line per split: with train, val, test and
        # a few eval workloads, all of them on a single axes is unreadable, and
        # the comparison that matters is between splits *within* a statistic.
        stats = [("mean", "Mean"), ("median", "Median"), ("p99", "99p")]
        stats = [(key, label) for key, label in stats
                if any(np.isfinite(arrays[key]).any() for _, arrays in series)]
        if len(stats) == 0:
            return None

        os.makedirs(save_dir, exist_ok=True)
        epochs = np.arange(len(history))
        fig, axes = plt.subplots(1, len(stats), figsize=(6*len(stats), 4),
                squeeze=False)

        # one shared legend for the figure; a split can be missing from a given
        # panel, so collect handles across all of them
        legend_handles = {}

        for pi, (stat_key, stat_label) in enumerate(stats):
            ax = axes[0][pi]
            for si, (label, arrays) in enumerate(series):
                values = arrays[stat_key]
                mask = np.isfinite(values)
                if not mask.any():
                    continue

                lines = ax.plot(
                    epochs[mask],
                    values[mask],
                    label=label,
                    color=self.QERR_SPLIT_COLORS[
                        si % len(self.QERR_SPLIT_COLORS)],
                    marker="o",
                    markersize=3,
                )
                legend_handles.setdefault(label, lines[0])

            ax.set_title("{} Q-Error".format(stat_label))
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Q-Error")
            # q-error is a ratio >= 1 and the splits routinely sit orders of
            # magnitude apart (train ~2, a shifted eval workload ~100), so a
            # linear axis would flatten the lower curves into the floor
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3, which="both")

        ncol = max(1, min(len(legend_handles), 5))
        fig.legend(list(legend_handles.values()), list(legend_handles.keys()),
                loc="lower center", ncol=ncol, frameon=False)
        fig.suptitle(title)

        # leave room for the suptitle and for however many rows the shared
        # legend wraps onto, so it never lands on top of the x-axis labels
        legend_rows = int(np.ceil(len(legend_handles) / ncol))
        bottom = min(0.06 + 0.05*legend_rows, 0.4)
        fig.tight_layout(rect=[0, bottom, 1, 0.94])
        plot_path = os.path.join(save_dir, plot_name)
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return plot_path

    def train_one_epoch_with_new_discriminator(self, target_loader):
        start = time.time()
        reg_losses = []
        recon_losses = []
        phase1_losses = []
        disc_losses = []
        gen_losses = []
        disc_accs = []
        disc_acc_source = []
        disc_acc_target = []
        gen_fool_accs = []

        target_iter = iter(target_loader)

        lambda_adv = getattr(self, "lambda_adv", 0.01)
        
        for batch_idx, (xbatch_source, ybatch_source, info_source) in enumerate(self.trainloader):
            ybatch_source = ybatch_source.to(device, non_blocking=True)

            try:
                xbatch_target, _, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                xbatch_target, _, _ = next(target_iter)

            source_batch_size = ybatch_source.shape[0]
            if isinstance(xbatch_target, dict):
                target_batch_size = xbatch_target["flow"].shape[0]
            else:
                target_batch_size = xbatch_target.shape[0]

            current_batch_size = min(source_batch_size, target_batch_size)
            if current_batch_size <= 0:
                continue

            xbatch_source = self._slice_batch_inputs(xbatch_source, current_batch_size)
            ybatch_source = ybatch_source[:current_batch_size]
            info_source = self._slice_batch_info(info_source, current_batch_size)
            xbatch_target = self._slice_batch_inputs(xbatch_target, current_batch_size)

            xbatch_source = self._apply_training_batch_transforms(xbatch_source)
            xbatch_target = self._apply_training_batch_transforms(xbatch_target)

            # Phase 1: regression on source only.
            self.opt_regression.zero_grad()
            if self._decoder_loss_enabled():
                pred_source, z_source = self.net.forward_with_latent(xbatch_source)
                _, z_target = self.net.forward_with_latent(xbatch_target)
                pred = pred_source.squeeze(1)
            else:
                pred = self.net(xbatch_source).squeeze(1)

            if self.subplan_level_outputs:
                idxs = torch.zeros(pred.shape, dtype=torch.bool)
                for i, nt in enumerate(info_source["num_tables"]):
                    if nt >= 10:
                        nt = 10
                    nt -= 1
                    idxs[i, nt] = True
                pred = pred[idxs]

            loss_reg = self._compute_regression_loss(pred, ybatch_source, info_source)
            loss_recon = torch.tensor(0.0, device=device)
            if self._decoder_loss_enabled():
                loss_recon, _, _ = self._compute_decoder_reconstruction_loss(
                    z_source, xbatch_source, z_target, xbatch_target)

            decoder_loss_weight = getattr(self, "decoder_loss_weight", 1.0)
            phase1_loss = loss_reg + (decoder_loss_weight * loss_recon)
            phase1_loss.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(
                    self.opt_regression.param_groups[0]["params"],
                    self.clip_gradient,
                )
            self.opt_regression.step()

            # Phase 2: train discriminator (source vs target).
            if self.epoch >= getattr(self, "discriminator_warmup_epochs", 5):
                disc_update_interval = max(
                    1, int(getattr(self, "discriminator_update_interval", 2))
                )
                generator_steps_per_batch = max(
                    1, int(getattr(self, "generator_steps_per_batch", 1))
                )
                source_label_value = float(
                    getattr(self, "discriminator_source_label", 1.0)
                )
                target_label_value = float(
                    getattr(self, "discriminator_target_label", 0.0)
                )

                train_disc = (batch_idx % disc_update_interval == 0)
                labels_source = torch.full(
                    (current_batch_size, 1),
                    source_label_value,
                    device=device,
                )
                labels_target = torch.full(
                    (current_batch_size, 1),
                    target_label_value,
                    device=device,
                )
                hard_labels_source = torch.ones_like(labels_source)
                hard_labels_target = torch.zeros_like(labels_target)
                loss_d = torch.tensor(0.0, device=device)
                pred_source_disc = torch.full(
                    (current_batch_size, 1), 0.5, device=device
                )
                pred_target_disc = torch.full(
                    (current_batch_size, 1), 0.5, device=device
                )
                if train_disc:
                    self.opt_discriminator.zero_grad()
                    _, z_source_det = self.net.forward_with_latent(xbatch_source)
                    _, z_target_det = self.net.forward_with_latent(xbatch_target)
                    z_source_det = z_source_det.detach()
                    z_target_det = z_target_det.detach()

                    pred_source_disc = self.net.discriminate(z_source_det)
                    pred_target_disc = self.net.discriminate(z_target_det)

                    loss_d_source = self.bce_loss(pred_source_disc, labels_source)
                    loss_d_target = self.bce_loss(pred_target_disc, labels_target)
                    loss_d = 0.5 * (loss_d_source + loss_d_target)
                    loss_d.backward()
                    self.opt_discriminator.step()


                # Phase 3: train encoder to fool discriminator on target.
                # Run this multiple times to let the encoder catch up.
                for _ in range(generator_steps_per_batch):
                    self.opt_generator.zero_grad()
                    
                    # Forward pass to get the latent vector
                    _, z_target_gen = self.net.forward_with_latent(xbatch_target)
                    
                    # Trick labels (1.0)
                    trick_labels = torch.ones(current_batch_size, 1, device=device)
                    pred_target_gen = self.net.discriminate(z_target_gen)
                    
                    loss_g = self.bce_loss(pred_target_gen, trick_labels)
                    #loss_g = lambda_adv * self.bce_loss(pred_target_gen, trick_labels)
                    loss_g.backward()
                    
                    if self.clip_gradient is not None:
                        clip_grad_norm_(
                            self.opt_generator.param_groups[0]["params"],
                            self.clip_gradient,
                        )
                    self.opt_generator.step()

                with torch.no_grad():
                    disc_pred_source_cls = (pred_source_disc >= 0.5).float()
                    disc_pred_target_cls = (pred_target_disc >= 0.5).float()
                    
                    source_acc = (disc_pred_source_cls == hard_labels_source).float().mean().item()
                    target_acc = (disc_pred_target_cls == hard_labels_target).float().mean().item()
                    

                    combined_preds = torch.cat([disc_pred_source_cls, disc_pred_target_cls], dim=0)
                    #combined_labels = torch.cat([labels_source, labels_target], dim=0)
                    combined_labels = torch.cat([hard_labels_source, hard_labels_target], dim=0)

                    disc_acc = (combined_preds == combined_labels).float().mean().item()
                    fool_acc = (pred_target_gen >= 0.5).float().mean().item()

            
                disc_losses.append(loss_d.item())
                gen_losses.append(loss_g.item())
                disc_accs.append(disc_acc)
                disc_acc_source.append(source_acc)
                disc_acc_target.append(target_acc)
                gen_fool_accs.append(fool_acc)
            reg_losses.append(loss_reg.item())
            recon_losses.append(loss_recon.item())
            phase1_losses.append(phase1_loss.item())

        metrics = {
            "loss_reg": float(np.mean(reg_losses)) if len(reg_losses) > 0 else 0.0,
            "loss_recon": float(np.mean(recon_losses)) if len(recon_losses) > 0 else 0.0,
            "loss_phase1": float(np.mean(phase1_losses)) if len(phase1_losses) > 0 else 0.0,
            "loss_d": float(np.mean(disc_losses)) if len(disc_losses) > 0 else 0.0,
            "loss_g": float(np.mean(gen_losses)) if len(gen_losses) > 0 else 0.0,
            "disc_acc": float(np.mean(disc_accs)) if len(disc_accs) > 0 else 0.0,
            "disc_acc_source": float(np.mean(disc_acc_source)) if len(disc_acc_source) > 0 else 0.0,
            "disc_acc_target": float(np.mean(disc_acc_target)) if len(disc_acc_target) > 0 else 0.0,
            "gen_fool_acc": float(np.mean(gen_fool_accs)) if len(gen_fool_accs) > 0 else 0.0,
            "epoch_seconds": round(time.time() - start, 2),
        }
        return metrics

    def train_one_epoch_dann(self, target_loader):
        return _train_one_epoch_dann(self, target_loader)

    def train_one_epoch_with_latent_generator(self):
        start = time.time()
        reg_losses = []
        recon_losses = []
        phase1_losses = []
        disc_losses = []
        gen_losses = []
        disc_accs = []
        disc_acc_source = []
        disc_acc_fake = []
        gen_fool_accs = []

        for _, (xbatch_source, ybatch_source, info_source) in enumerate(self.trainloader):
            ybatch_source = ybatch_source.to(device, non_blocking=True)
            current_batch_size = ybatch_source.shape[0]
            if current_batch_size <= 0:
                continue

            xbatch_source = self._slice_batch_inputs(xbatch_source, current_batch_size)
            ybatch_source = ybatch_source[:current_batch_size]
            info_source = self._slice_batch_info(info_source, current_batch_size)

            xbatch_source = self._apply_training_batch_transforms(xbatch_source)

            # Phase 1: regression on source only.
            self.opt_regression.zero_grad()
            if self.subplan_level_outputs:
                pred = self.net(xbatch_source).squeeze(1)
                idxs = torch.zeros(pred.shape, dtype=torch.bool)
                for i, nt in enumerate(info_source["num_tables"]):
                    if nt >= 10:
                        nt = 10
                    nt -= 1
                    idxs[i, nt] = True
                pred = pred[idxs]
            else:
                pred = self.net(xbatch_source).squeeze(1)

            loss_reg = self._compute_regression_loss(pred, ybatch_source, info_source)
            loss_reg.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(
                    self.opt_regression.param_groups[0]["params"],
                    self.clip_gradient,
                )
            self.opt_regression.step()

            # Phase 2: train discriminator on source vs generated fake target.
            self.opt_discriminator.zero_grad()
            _, z_source_det = self.net.forward_with_latent(xbatch_source)
            z_source_det = z_source_det.detach()

            noise_det = self.net.sample_noise(current_batch_size, device_override=z_source_det.device)
            z_fake_det = self.net.generate(noise_det).detach()

            labels_source = torch.ones(current_batch_size, 1, device=device)
            labels_fake = torch.zeros(current_batch_size, 1, device=device)

            pred_source_disc = self.net.discriminate(z_source_det)
            pred_fake_disc = self.net.discriminate(z_fake_det)

            loss_d_source = self.bce_loss(pred_source_disc, labels_source)
            loss_d_fake = self.bce_loss(pred_fake_disc, labels_fake)
            loss_d = 0.5 * (loss_d_source + loss_d_fake)
            loss_d.backward()
            self.opt_discriminator.step()

            # Phase 3: train generator only to fool discriminator.
            self.opt_latent_generator.zero_grad()
            noise_gen = self.net.sample_noise(current_batch_size, device_override=z_source_det.device)
            z_fake_gen = self.net.generate(noise_gen)
            trick_labels = torch.ones(current_batch_size, 1, device=device)
            pred_fake_gen = self.net.discriminate(z_fake_gen)
            loss_g = self.bce_loss(pred_fake_gen, trick_labels)
            loss_g.backward()
            if self.clip_gradient is not None:
                clip_grad_norm_(
                    self.opt_latent_generator.param_groups[0]["params"],
                    self.clip_gradient,
                )
            self.opt_latent_generator.step()

            with torch.no_grad():
                disc_pred_source_cls = (pred_source_disc >= 0.5).float()
                disc_pred_fake_cls = (pred_fake_disc >= 0.5).float()
                source_acc = (disc_pred_source_cls == labels_source).float().mean().item()
                fake_acc = (disc_pred_fake_cls == labels_fake).float().mean().item()
                combined_preds = torch.cat([disc_pred_source_cls, disc_pred_fake_cls], dim=0)
                combined_labels = torch.cat([labels_source, labels_fake], dim=0)
                disc_acc = (combined_preds == combined_labels).float().mean().item()
                fool_acc = (pred_fake_gen >= 0.5).float().mean().item()

            reg_losses.append(loss_reg.item())
            disc_losses.append(loss_d.item())
            gen_losses.append(loss_g.item())
            disc_accs.append(disc_acc)
            disc_acc_source.append(source_acc)
            disc_acc_fake.append(fake_acc)
            gen_fool_accs.append(fool_acc)

        metrics = {
            "loss_reg": float(np.mean(reg_losses)) if len(reg_losses) > 0 else 0.0,
            "loss_recon": float(np.mean(recon_losses)) if len(recon_losses) > 0 else 0.0,
            "loss_phase1": float(np.mean(phase1_losses)) if len(phase1_losses) > 0 else 0.0,
            "loss_d": float(np.mean(disc_losses)) if len(disc_losses) > 0 else 0.0,
            "loss_g": float(np.mean(gen_losses)) if len(gen_losses) > 0 else 0.0,
            "disc_acc": float(np.mean(disc_accs)) if len(disc_accs) > 0 else 0.0,
            "disc_acc_source": float(np.mean(disc_acc_source)) if len(disc_acc_source) > 0 else 0.0,
            "disc_acc_fake": float(np.mean(disc_acc_fake)) if len(disc_acc_fake) > 0 else 0.0,
            # Keep existing key so plotting/reporting stays compatible.
            "disc_acc_target": float(np.mean(disc_acc_fake)) if len(disc_acc_fake) > 0 else 0.0,
            "gen_fool_acc": float(np.mean(gen_fool_accs)) if len(gen_fool_accs) > 0 else 0.0,
            "epoch_seconds": round(time.time() - start, 2),
        }
        return metrics

    def train(self, training_samples, **kwargs):

        self.all_errs = []
        self.best_model_epoch = -1
        self.model_weights = []
        self.standard_train_history = []
        self.adv_weights = None
        self.adv_weight_level = kwargs.get("adv_weight_level", "dataset")

        self.true_costs = {}
        self.true_costs["val"] = 0.0
        self.true_costs["test"] = 0.0
        # self.true_costs["job"] = 0.0
        # self.true_costs["jobm"] = 0.0

        assert isinstance(training_samples[0], dict)
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples
        target_samples = self._collect_target_queries(kwargs)

        self.seen_subplans = set()
        for sample in training_samples:
            for node in sample["subset_graph"].nodes():
                self.seen_subplans.add(str(node))

        if "subplan_mask" in kwargs:
            subplan_mask = kwargs["subplan_mask"]
        else:
            subplan_mask = None
        val_subplan_mask = kwargs.get("val_subplan_mask", None)
        test_subplan_mask = kwargs.get("test_subplan_mask", None)
        # optional list, aligned with kwargs["evalqs"], of per-eval-group
        # subplan masks; restricts the during-training eval featurization to
        # the selected subplans (subquery-level --eval_csv), same reason as
        # train/val/test.
        eval_subplan_masks = kwargs.get("eval_subplan_masks", None)

        self.trainds = self.init_dataset(training_samples,
                self.load_query_together,
                max_num_tables = self.max_num_tables,
                load_padded_mscn_feats=self.load_padded_mscn_feats,
                subplan_mask = subplan_mask
                )

        if "adv_weights" in kwargs and kwargs["adv_weights"] is not None:
            self.adv_weights = np.asarray(kwargs["adv_weights"], dtype=np.float32)
            if self.adv_weight_level == "dataset":
                expected = len(self.trainds)
            elif self.adv_weight_level == "query":
                expected = len(training_samples)
            else:
                raise ValueError(f"Unsupported adv_weight_level: {self.adv_weight_level}")

            if len(self.adv_weights) != expected:
                raise ValueError(
                    f"Expected {expected} adversarial weights for level "
                    f"{self.adv_weight_level}, got {len(self.adv_weights)}"
                )

        self.trainloader = data.DataLoader(self.trainds,
                batch_size=self.mb_size, shuffle=True,
                collate_fn=self.collate_fn,
                # Without an explicit generator the shuffle order comes from
                # the global torch RNG, i.e. from whatever ran before this
                # point. Tie it to --train_seed instead.
                generator=make_generator(self.train_seed, "trainloader"),
                worker_init_fn=make_worker_init_fn(self.train_seed),
                # num_workers=self.num_workers
                )

        # if self.eval_epoch >= self.max_epochs and \
            # "flowloss" not in self.loss_func_name:
            # del training_samples[1:]

        self.eval_ds = {}
        self.samples = {}
        # subplan_mask each eval dataset was built with, keyed the same way as
        # self.eval_ds. Needed to compute q-error for it later:
        # format_model_test_output has to walk exactly the nodes the dataset
        # kept, or its estimates land on the wrong subplans.
        self.eval_subplan_masks = {}

        # if self.eval_epoch < self.max_epochs:
            # self.samples["train"] = training_samples
            # self.eval_ds["train"] = self.init_dataset(training_samples,
                    # self.load_query_together,
                    # max_num_tables = -1,
                    # load_padded_mscn_feats=self.load_padded_mscn_feats)


        if self.eval_epoch < self.max_epochs:
            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(kwargs["valqs"], False,
                        load_padded_mscn_feats=self.load_padded_mscn_feats,
                        subplan_mask=val_subplan_mask)
                self.samples["val"] = kwargs["valqs"]
                self.eval_subplan_masks["val"] = val_subplan_mask

            # if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                # pass
            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                if len(kwargs["testqs"]) > 400:
                    ns = int(len(kwargs["testqs"]) / 10)
                    # Private RNG at a fixed seed: this is a DATA decision
                    # (which test queries get evaluated), so it must not move
                    # with --train_seed. It also used to call random.seed(42)
                    # on the global RNG, resetting the stream for everything
                    # constructed afterwards.
                    test_rng = random.Random(42)
                    if test_subplan_mask is not None:
                        # keep testqs and its mask aligned by sampling the
                        # same indices out of both
                        idxs = test_rng.sample(range(len(kwargs["testqs"])), ns)
                        testqs = [kwargs["testqs"][i] for i in idxs]
                        cur_test_subplan_mask = [test_subplan_mask[i] for i in idxs]
                    else:
                        testqs = test_rng.sample(kwargs["testqs"], ns)
                        cur_test_subplan_mask = None
                else:
                    testqs = kwargs["testqs"]
                    cur_test_subplan_mask = test_subplan_mask

                self.eval_ds["test"] = self.init_dataset(testqs,
                        False,
                        load_padded_mscn_feats=self.load_padded_mscn_feats,
                        subplan_mask=cur_test_subplan_mask)
                self.samples["test"] = testqs
                self.eval_subplan_masks["test"] = cur_test_subplan_mask

            if "evalqs" in kwargs and len(kwargs["eval_qdirs"]) > 0:
                eval_qdirs = kwargs["eval_qdirs"]

                for ei, cur_evalqs in enumerate(kwargs["evalqs"]):
                    evalqname = eval_qdirs[ei]
                    # mask aligned with this eval group's qreps; any branch
                    # that filters cur_evalqs must filter this in lockstep so
                    # subplan_mask[i] keeps pointing at cur_evalqs[i].
                    cur_eval_mask = None
                    if eval_subplan_masks is not None and ei < len(eval_subplan_masks):
                        cur_eval_mask = eval_subplan_masks[ei]

                    if "job" in evalqname:
                        evalqname = "JOB"
                        print("Going to remove JOB Q29 from evaluation because it takes too long for computing PPC")
                        if cur_eval_mask is not None:
                            kept = [(q, m) for q, m in zip(cur_evalqs, cur_eval_mask)
                                    if "29" not in q["name"]]
                            cur_evalqs = [q for q, _ in kept]
                            cur_eval_mask = [m for _, m in kept]
                        else:
                            cur_evalqs = [q for q in cur_evalqs if "29" not in
                                    q["name"]]

                    elif "imdb" in evalqname:
                        # evalqname = "CEB-IMDb"
                        if cur_eval_mask is not None:
                            group_pairs = [(q, m) for q, m in zip(cur_evalqs, cur_eval_mask)
                                    if "group" in q["sql"].lower()]
                            not_group_pairs = [(q, m) for q, m in zip(cur_evalqs, cur_eval_mask)
                                    if "group" not in q["sql"].lower()]
                            group_evalqs = [q for q, _ in group_pairs]
                            group_mask = [m for _, m in group_pairs]
                            not_group_evalqs = [q for q, _ in not_group_pairs]
                            not_group_mask = [m for _, m in not_group_pairs]
                        else:
                            group_evalqs = [q for q in cur_evalqs if "group" in
                                    q["sql"].lower()]
                            not_group_evalqs = [q for q in cur_evalqs if "group" not in
                                    q["sql"].lower()]
                            group_mask = None
                            not_group_mask = None
                        gqname = "CEB-IMDb-Complex"
                        not_gqname = "CEB-IMDb-NoGroupNoLike"
                        self.eval_ds[gqname] = \
                                self.init_dataset(group_evalqs, False,
                                load_padded_mscn_feats=self.load_padded_mscn_feats,
                                subplan_mask=group_mask)
                        self.true_costs[gqname] = 0.0
                        self.samples[gqname] = group_evalqs
                        self.eval_subplan_masks[gqname] = group_mask

                        self.eval_ds[not_gqname] = \
                                self.init_dataset(not_group_evalqs, False,
                                load_padded_mscn_feats=self.load_padded_mscn_feats,
                                subplan_mask=not_group_mask)
                        self.true_costs[not_gqname] = 0.0
                        self.samples[not_gqname] = not_group_evalqs
                        self.eval_subplan_masks[not_gqname] = not_group_mask

                        continue

                    elif "stats" in evalqname:
                        evalqname = "Stats-CEB"

                    print("{}, num eval queries: {}".format(evalqname,
                        len(cur_evalqs)))

                    if len(cur_evalqs) == 0:
                        continue

                    self.eval_ds[evalqname] = self.init_dataset(cur_evalqs,
                            False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats,
                            subplan_mask=cur_eval_mask)
                    self.true_costs[evalqname] = 0.0
                    self.samples[evalqname] = cur_evalqs
                    self.eval_subplan_masks[evalqname] = cur_eval_mask

        # TODO: initialize self.num_features
        self.net, self.optimizer = self.init_net(self.trainds[0])
        run_plot_dir = self._prepare_run_plot_dir(kwargs)
        self._ensure_latent_visualization_ready()
        self._setup_latent_visualization_loaders(target_samples=target_samples)

        model_size = self.num_parameters()
        print("""Training samples: {}, Model size: {}""".
                format(len(self.trainds), model_size))

        if "flow" in self.loss_func_name:
            self.update_flow_training_info()

        if self.training_opt == "swa":
            self.swa_net = AveragedModel(self.net)
            # self.swa_start = self.swa_start
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.opt_lr)

        if self.max_epochs == -1:
            total_epochs = 1000
        else:
            total_epochs = self.max_epochs

        if self.early_stopping:
            eplosses = []
            pct_chngs = []

        ## debug code
        # fkeys = list(dir(self.featurizer))
        # fkeys.sort()
        # attrs = ""
        # for k in fkeys:
            # attrvals = getattr(self.featurizer, k)
            # if not hasattr(attrvals, "__len__") and \
                # "method" not in str(attrvals):
                # attr = str(k) + str(attrvals) + ";"
                # print(attr)

        # fkeys = list(dir(self))
        # fkeys.sort()
        # attrs = ""
        # for k in fkeys:
            # attrvals = getattr(self, k)
            # if not hasattr(attrvals, "__len__") and \
                # "method" not in str(attrvals):
                # attr = str(k) + str(attrvals) + ";"
                # print(attr)

        # pdb.set_trace()

        for self.epoch in range(0,total_epochs):
            should_eval = (self.epoch % self.eval_epoch == 0)
            if should_eval:
                self.periodic_eval()

            train_loss = self.train_one_epoch()

            epoch_metrics = {"train_loss": train_loss}

            if should_eval:
                train_qerr_metrics = self._compute_train_qerror_metrics(
                        subplan_mask)
                if train_qerr_metrics["mean"] is not None:
                    epoch_metrics["train_qerr"] = train_qerr_metrics["mean"]
                    epoch_metrics["train_qerr_median"] = train_qerr_metrics["median"]
                    epoch_metrics["train_qerr_p99"] = train_qerr_metrics["p99"]
                    print(
                        "Epoch {} train_qerr={:.6f}(median={:.6f}, 99p={:.6f})".format(
                            self.epoch,
                            epoch_metrics["train_qerr"],
                            epoch_metrics["train_qerr_median"],
                            epoch_metrics["train_qerr_p99"],
                        )
                    )

            if should_eval and "val" in self.eval_ds:
                val_preds, val_ys = self._eval_ds(self.eval_ds["val"], self.samples["val"])
                val_loss_metrics = self._compute_loss_metrics_from_numpy(val_preds, val_ys)
                if val_loss_metrics["mean"] is not None:
                    epoch_metrics["val_loss"] = val_loss_metrics["mean"]
                    epoch_metrics["val_loss_median"] = val_loss_metrics["median"]
                    epoch_metrics["val_loss_p99"] = val_loss_metrics["p99"]

                val_qerr_metrics = self._compute_qerror_metrics_from_numpy(
                    val_preds,
                    self.samples.get("val"),
                    subplan_mask=val_subplan_mask,
                )
                if val_qerr_metrics["mean"] is not None:
                    epoch_metrics["val_qerr"] = val_qerr_metrics["mean"]
                    epoch_metrics["val_qerr_median"] = val_qerr_metrics["median"]
                    epoch_metrics["val_qerr_p99"] = val_qerr_metrics["p99"]
                    print(
                        "Epoch {} val_qerr={:.6f}(median={:.6f}, 99p={:.6f})".format(
                            self.epoch,
                            epoch_metrics["val_qerr"],
                            epoch_metrics["val_qerr_median"],
                            epoch_metrics["val_qerr_p99"],
                        )
                    )

            if should_eval:
                eval_qerrs = self._compute_eval_qerror_metrics()
                if len(eval_qerrs) > 0:
                    epoch_metrics["eval_qerrs"] = eval_qerrs
                    print(
                        "Epoch {} eval_qerrs={}".format(
                            self.epoch,
                            self._format_eval_qerrs(eval_qerrs),
                        )
                    )
                    if self.use_wandb:
                        for name, metrics in eval_qerrs.items():
                            wandb.log({
                                "EvalQError-" + name: metrics["mean"],
                                "EvalQError-" + name + "-Median": metrics["median"],
                                "EvalQError-" + name + "-99p": metrics["p99"],
                                "epoch": self.epoch,
                            })

            if should_eval:
                eval_latent_mmds = self.compute_eval_latent_mmds()
                if len(eval_latent_mmds) > 0:
                    epoch_metrics["eval_latent_mmds"] = eval_latent_mmds
                    epoch_metrics["latent_mmd"] = float(np.mean(list(eval_latent_mmds.values())))
                    self.latest_eval_latent_mmds = eval_latent_mmds
                    print(
                        "Epoch {} latent_mmds={}".format(
                            self.epoch,
                            self._format_eval_latent_mmds(eval_latent_mmds),
                        )
                    )
                    if self.use_wandb:
                        wandb.log({
                            "LatentMMD": epoch_metrics["latent_mmd"],
                            "epoch": self.epoch,
                        })

                        if "train_qerr" in epoch_metrics:
                            wandb.log({
                                "TrainQError": epoch_metrics["train_qerr"],
                                "TrainQError-Median": epoch_metrics["train_qerr_median"],
                                "TrainQError-99p": epoch_metrics["train_qerr_p99"],
                                "epoch": self.epoch,
                            })

                        if "val_qerr" in epoch_metrics:
                            wandb.log({
                                "ValQError": epoch_metrics["val_qerr"],
                                "ValQError-Median": epoch_metrics["val_qerr_median"],
                                "ValQError-99p": epoch_metrics["val_qerr_p99"],
                                "epoch": self.epoch,
                            })

            self._maybe_save_latent_visualization(run_plot_dir)

            self.standard_train_history.append(epoch_metrics)

            self.model_weights.append(copy.deepcopy(self.net.state_dict()))

            # TODO: needs to decide if we should stop training
            if self.early_stopping == 1:
                if "val" in self.eval_ds:
                    ds = self.eval_ds["val"]
                else:
                    ds = self.eval_ds["train"]

                preds, ys = self._eval_ds(ds)
                losses = self.loss_func(torch.from_numpy(preds), torch.from_numpy(ys))
                eploss = torch.mean(losses).item()
                if len(eplosses) >= 1:
                    pct = 100* ((eploss-eplosses[-1])/eplosses[-1])
                    pct_chngs.append(pct)

                eplosses.append(eploss)
                if len(pct_chngs) > 5:
                    trailing_chng = np.mean(pct_chngs[-5:-1])
                    if trailing_chng > -0.1:
                        print("Going to exit training at epoch: ", self.epoch)
                        break

            elif self.early_stopping == 2:
                self.periodic_eval()
                ppc_rel = self.all_errs[-1]['PostgresPlanCost-C-Relative-val']

                if len(eplosses) >= 1:
                    pct = 100* ((ppc_rel-eplosses[-1])/eplosses[-1])
                    pct_chngs.append(pct)

                eplosses.append(ppc_rel)

                if self.epoch > 2 and pct_chngs[-1] > 1:
                    print(eplosses)
                    print(pct_chngs)
                    # print(eplosses[-5:-1])
                    # print(pct_chngs[-5:-1])
                    # revert to model before this epoch's training
                    print("Going to exit training at epoch: ", self.epoch)
                    self.best_model_epoch = self.epoch-1
                    break

        # self.periodic_eval()

        if self.training_opt == "swa":
            torch.optim.swa_utils.update_bn(self.trainloader, self.swa_net)

        if self.best_model_epoch != -1:
            print("""training done, will update our model based on validation set""")
            assert len(self.model_weights) > 0
            self.net.load_state_dict(self.model_weights[self.best_model_epoch])

            # self.nets[0].load_state_dict(self.best_model_dict)
            # self.nets[0].eval()

        std_plot_path = self._save_standard_training_plot(run_plot_dir)
        if std_plot_path is not None:
            print("Saved standard training plot to:", std_plot_path)

        std_qerr_plot_path = self._save_qerror_plot(
            self.standard_train_history,
            run_plot_dir,
            "standard_val_qerror.png",
            "Q-Error by Epoch",
        )
        if std_qerr_plot_path is not None:
            print("Saved standard q-error plot to:", std_qerr_plot_path)

        std_mmd_plot_path = self._save_latent_mmd_plot(
            self.standard_train_history,
            run_plot_dir,
            "standard_eval_latent_mmd.png",
            "Latent MMD by Eval Dataset",
        )
        if std_mmd_plot_path is not None:
            print("Saved standard latent MMD plot to:", std_mmd_plot_path)

        self._maybe_save_latent_visualization(run_plot_dir, force=True)
        self.save_model(save_dir='./saved_models', suffix_name="_epoch"+str(self.epoch))

    def train_with_new_discriminator(self, training_samples, **kwargs):
        self.all_errs = []
        self.best_model_epoch = -1
        self.model_weights = []
        self.adv_weights = None
        self.adv_weight_level = kwargs.get("adv_weight_level", "dataset")
        self.adversarial_train_history = []

        if self.loss_func_name == "flowloss" or self.load_query_together:
            raise RuntimeError(
                "train_with_new_discriminator currently supports non-flowloss training only."
            )

        self.true_costs = {}
        self.true_costs["val"] = 0.0
        self.true_costs["test"] = 0.0

        assert isinstance(training_samples[0], dict)
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples
        target_samples = self._collect_target_queries(kwargs)

        self.seen_subplans = set()
        for sample in training_samples:
            for node in sample["subset_graph"].nodes():
                self.seen_subplans.add(str(node))

        if "subplan_mask" in kwargs:
            subplan_mask = kwargs["subplan_mask"]
        else:
            subplan_mask = None
        val_subplan_mask = kwargs.get("val_subplan_mask", None)
        test_subplan_mask = kwargs.get("test_subplan_mask", None)

        self.trainds = self.init_dataset(
            training_samples,
            self.load_query_together,
            max_num_tables=self.max_num_tables,
            load_padded_mscn_feats=self.load_padded_mscn_feats,
            subplan_mask=subplan_mask,
        )

        if "adv_weights" in kwargs and kwargs["adv_weights"] is not None:
            self.adv_weights = np.asarray(kwargs["adv_weights"], dtype=np.float32)
            if self.adv_weight_level == "dataset":
                expected = len(self.trainds)
            elif self.adv_weight_level == "query":
                expected = len(training_samples)
            else:
                raise ValueError(f"Unsupported adv_weight_level: {self.adv_weight_level}")

            if len(self.adv_weights) != expected:
                raise ValueError(
                    f"Expected {expected} adversarial weights for level "
                    f"{self.adv_weight_level}, got {len(self.adv_weights)}"
                )

        self.trainloader = data.DataLoader(
            self.trainds,
            batch_size=self.mb_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            generator=make_generator(self.train_seed, "trainloader"),
            worker_init_fn=make_worker_init_fn(self.train_seed),
        )

        self.eval_ds = {}
        self.samples = {}
        # see the note in train(): the mask each eval dataset was built with,
        # needed to line its predictions back up with its subplans
        self.eval_subplan_masks = {}

        if self.eval_epoch < self.max_epochs:
            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(
                    kwargs["valqs"], False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                    subplan_mask=val_subplan_mask,
                )
                self.samples["val"] = kwargs["valqs"]
                self.eval_subplan_masks["val"] = val_subplan_mask

            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                if len(kwargs["testqs"]) > 400:
                    ns = int(len(kwargs["testqs"]) / 10)
                    # Private RNG at a fixed seed: this is a DATA decision
                    # (which test queries get evaluated), so it must not move
                    # with --train_seed. It also used to call random.seed(42)
                    # on the global RNG, resetting the stream for everything
                    # constructed afterwards.
                    test_rng = random.Random(42)
                    if test_subplan_mask is not None:
                        # keep testqs and its mask aligned by sampling the
                        # same indices out of both
                        idxs = test_rng.sample(range(len(kwargs["testqs"])), ns)
                        testqs = [kwargs["testqs"][i] for i in idxs]
                        cur_test_subplan_mask = [test_subplan_mask[i] for i in idxs]
                    else:
                        testqs = test_rng.sample(kwargs["testqs"], ns)
                        cur_test_subplan_mask = None
                else:
                    testqs = kwargs["testqs"]
                    cur_test_subplan_mask = test_subplan_mask

                self.eval_ds["test"] = self.init_dataset(
                    testqs, False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                    subplan_mask=cur_test_subplan_mask,
                )
                self.samples["test"] = testqs
                self.eval_subplan_masks["test"] = cur_test_subplan_mask

            if "evalqs" in kwargs and len(kwargs["eval_qdirs"]) > 0:
                eval_qdirs = kwargs["eval_qdirs"]
                for ei, cur_evalqs in enumerate(kwargs["evalqs"]):
                    evalqname = eval_qdirs[ei]
                    if "job" in evalqname:
                        evalqname = "JOB"
                        print("Going to remove JOB Q29 from evaluation because it takes too long for computing PPC")
                        cur_evalqs = [q for q in cur_evalqs if "29" not in q["name"]]
                    elif "imdb" in evalqname:
                        group_evalqs = [q for q in cur_evalqs if "group" in q["sql"].lower()]
                        not_group_evalqs = [q for q in cur_evalqs if "group" not in q["sql"].lower()]
                        gqname = "CEB-IMDb-Complex"
                        not_gqname = "CEB-IMDb-NoGroupNoLike"
                        self.eval_ds[gqname] = self.init_dataset(
                            group_evalqs, False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats,
                        )
                        self.true_costs[gqname] = 0.0
                        self.samples[gqname] = group_evalqs
                        self.eval_subplan_masks[gqname] = None

                        self.eval_ds[not_gqname] = self.init_dataset(
                            not_group_evalqs, False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats,
                        )
                        self.true_costs[not_gqname] = 0.0
                        self.samples[not_gqname] = not_group_evalqs
                        self.eval_subplan_masks[not_gqname] = None
                        continue
                    elif "stats" in evalqname:
                        evalqname = "Stats-CEB"

                    print("{}, num eval queries: {}".format(evalqname, len(cur_evalqs)))
                    if len(cur_evalqs) == 0:
                        continue

                    self.eval_ds[evalqname] = self.init_dataset(
                        cur_evalqs, False,
                        load_padded_mscn_feats=self.load_padded_mscn_feats,
                    )
                    self.true_costs[evalqname] = 0.0
                    self.samples[evalqname] = cur_evalqs
                    self.eval_subplan_masks[evalqname] = None

        self.net, self.optimizer = self.init_net(self.trainds[0])
        if not hasattr(self.net, "forward_with_latent") or \
                not hasattr(self.net, "discriminate"):
            raise RuntimeError(
                "train_with_new_discriminator requires an MSCN-style network "
                "with forward_with_latent() and discriminate()."
            )
        self._ensure_latent_discriminator_ready()
        self._init_new_discriminator_optimizers()
        if bool(getattr(self, "enable_decoder", False)) and \
                not self._decoder_loss_enabled():
            raise RuntimeError(
                "Decoder is enabled in config but decoder loss path is not active. "
                "The run would otherwise log recon_loss=0.0 silently."
            )

        if self.training_opt == "swa":
            raise RuntimeError(
                "SWA training is not supported in train_with_new_discriminator."
            )

        target_samples = self._collect_target_queries(kwargs)
        if len(target_samples) == 0:
            raise ValueError(
                "No target queries available for train_with_new_discriminator. "
                "Pass evalqs."
            )

        run_plot_dir = self._prepare_run_plot_dir(kwargs)

        self.targetds = self.init_dataset(
            target_samples,
            self.load_query_together,
            max_num_tables=self.max_num_tables,
            load_padded_mscn_feats=self.load_padded_mscn_feats,
        )
        self.target_loader = data.DataLoader(
            self.targetds,
            batch_size=self.mb_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            generator=make_generator(self.train_seed, "target_loader"),
            worker_init_fn=make_worker_init_fn(self.train_seed, "target_loader"),
        )
        self._setup_latent_visualization_loaders(target_samples=target_samples)

        model_size = self.num_parameters()
        print(
            "Training samples: {}, Target samples: {}, Model size: {}".format(
                len(self.trainds), len(self.targetds), model_size
            )
        )

        if self.max_epochs == -1:
            total_epochs = 1000
        else:
            total_epochs = self.max_epochs

        if self.early_stopping:
            eplosses = []
            pct_chngs = []

        train_epoch_fn = getattr(
            self, "_adversarial_epoch_train_fn", self.train_one_epoch_with_new_discriminator
        )

        for self.epoch in range(0, total_epochs):
            should_eval = (self.epoch % self.eval_epoch == 0)
            if self.epoch % self.eval_epoch == 0:
                self.periodic_eval()

            epoch_metrics = train_epoch_fn(self.target_loader)

            if should_eval:
                train_qerr_metrics = self._compute_train_qerror_metrics(
                        subplan_mask)
                if train_qerr_metrics["mean"] is not None:
                    epoch_metrics["train_qerr"] = train_qerr_metrics["mean"]
                    epoch_metrics["train_qerr_median"] = train_qerr_metrics["median"]
                    epoch_metrics["train_qerr_p99"] = train_qerr_metrics["p99"]

            if should_eval and "val" in self.eval_ds:
                val_preds, val_ys = self._eval_ds(self.eval_ds["val"], self.samples["val"])
                val_loss_metrics = self._compute_loss_metrics_from_numpy(val_preds, val_ys)
                if val_loss_metrics["mean"] is not None:
                    epoch_metrics["val_loss"] = val_loss_metrics["mean"]
                    epoch_metrics["val_loss_median"] = val_loss_metrics["median"]
                    epoch_metrics["val_loss_p99"] = val_loss_metrics["p99"]
                val_qerr_metrics = self._compute_qerror_metrics_from_numpy(
                    val_preds,
                    self.samples.get("val"),
                    subplan_mask=val_subplan_mask,
                )
                if val_qerr_metrics["mean"] is not None:
                    epoch_metrics["val_qerr"] = val_qerr_metrics["mean"]
                    epoch_metrics["val_qerr_median"] = val_qerr_metrics["median"]
                    epoch_metrics["val_qerr_p99"] = val_qerr_metrics["p99"]

            if should_eval:
                eval_qerrs = self._compute_eval_qerror_metrics()
                if len(eval_qerrs) > 0:
                    epoch_metrics["eval_qerrs"] = eval_qerrs
                    print(
                        "Epoch {} eval_qerrs={}".format(
                            self.epoch,
                            self._format_eval_qerrs(eval_qerrs),
                        )
                    )

            if should_eval:
                eval_latent_mmds = self.compute_eval_latent_mmds()
                if len(eval_latent_mmds) > 0:
                    epoch_metrics["eval_latent_mmds"] = eval_latent_mmds
                    epoch_metrics["latent_mmd"] = float(np.mean(list(eval_latent_mmds.values())))
                    self.latest_eval_latent_mmds = eval_latent_mmds
                    print(
                        "Epoch {} latent_mmds={}".format(
                            self.epoch,
                            self._format_eval_latent_mmds(eval_latent_mmds),
                        )
                    )

            self._maybe_save_latent_visualization(run_plot_dir)

            self.adversarial_train_history.append(epoch_metrics)
            self.model_weights.append(copy.deepcopy(self.net.state_dict()))

            if self.epoch % 2 == 0:
                print(
                    "Epoch {} took {}s, reg_loss={:.6f}, recon_loss={:.6f}, phase1_loss={:.6f}, disc_loss={:.6f}, disc_grad_norm={:.6f}, gen_loss={:.6f}, disc_acc={:.6f}, source_acc={:.6f}, target_acc={:.6f}, fool_acc={:.6f}, latent_mmd={:.6f}, train_qerr={:.6f}(median={:.6f}, 99p={:.6f}), val_qerr={:.6f}(median={:.6f}, 99p={:.6f})".format(
                        self.epoch,
                        epoch_metrics["epoch_seconds"],
                        epoch_metrics["loss_reg"],
                        epoch_metrics.get("loss_recon", 0.0),
                        epoch_metrics.get("loss_phase1", epoch_metrics["loss_reg"]),
                        epoch_metrics["loss_d"],
                        epoch_metrics.get("disc_grad_norm", float("nan")),
                        epoch_metrics["loss_g"],
                        epoch_metrics["disc_acc"],
                        epoch_metrics["disc_acc_source"],
                        epoch_metrics["disc_acc_target"],
                        epoch_metrics["gen_fool_acc"],
                        epoch_metrics.get("latent_mmd", float("nan")),
                        epoch_metrics.get("train_qerr", float("nan")),
                        epoch_metrics.get("train_qerr_median", float("nan")),
                        epoch_metrics.get("train_qerr_p99", float("nan")),
                        epoch_metrics.get("val_qerr", float("nan")),
                        epoch_metrics.get("val_qerr_median", float("nan")),
                        epoch_metrics.get("val_qerr_p99", float("nan")),
                    )
                )

            if self.use_wandb:
                wandb_payload = {
                    "TrainLoss": epoch_metrics["loss_reg"],
                    "Adv-Loss-D": epoch_metrics["loss_d"],
                    "Adv-Loss-G": epoch_metrics["loss_g"],
                    "Adv-Disc-Acc": epoch_metrics["disc_acc"],
                    "Adv-Disc-Source-Acc": epoch_metrics["disc_acc_source"],
                    "Adv-Disc-Target-Acc": epoch_metrics["disc_acc_target"],
                    "Adv-Gen-Fool-Acc": epoch_metrics["gen_fool_acc"],
                    "epoch": self.epoch,
                }
                if "disc_grad_norm" in epoch_metrics:
                    wandb_payload["Adv-Disc-Grad-Norm"] = epoch_metrics["disc_grad_norm"]
                if "latent_mmd" in epoch_metrics:
                    wandb_payload["LatentMMD"] = epoch_metrics["latent_mmd"]
                if "train_qerr" in epoch_metrics:
                    wandb_payload["TrainQError"] = epoch_metrics["train_qerr"]
                if "train_qerr_median" in epoch_metrics:
                    wandb_payload["TrainQError-Median"] = epoch_metrics["train_qerr_median"]
                if "train_qerr_p99" in epoch_metrics:
                    wandb_payload["TrainQError-99p"] = epoch_metrics["train_qerr_p99"]
                if "val_qerr" in epoch_metrics:
                    wandb_payload["ValQError"] = epoch_metrics["val_qerr"]
                if "val_qerr_median" in epoch_metrics:
                    wandb_payload["ValQError-Median"] = epoch_metrics["val_qerr_median"]
                if "val_qerr_p99" in epoch_metrics:
                    wandb_payload["ValQError-99p"] = epoch_metrics["val_qerr_p99"]
                for name, metrics in epoch_metrics.get("eval_qerrs", {}).items():
                    wandb_payload["EvalQError-" + name] = metrics["mean"]
                    wandb_payload["EvalQError-" + name + "-Median"] = metrics["median"]
                    wandb_payload["EvalQError-" + name + "-99p"] = metrics["p99"]
                wandb.log(wandb_payload)

            if self.early_stopping == 1:
                if "val" in self.eval_ds:
                    ds = self.eval_ds["val"]
                else:
                    ds = self.eval_ds["train"]

                preds, ys = self._eval_ds(ds)
                losses = self.loss_func(torch.from_numpy(preds), torch.from_numpy(ys))
                eploss = torch.mean(losses).item()
                if len(eplosses) >= 1:
                    pct = 100 * ((eploss - eplosses[-1]) / eplosses[-1])
                    pct_chngs.append(pct)

                eplosses.append(eploss)
                if len(pct_chngs) > 5:
                    trailing_chng = np.mean(pct_chngs[-5:-1])
                    if trailing_chng > -0.1:
                        print("Going to exit training at epoch: ", self.epoch)
                        break

            elif self.early_stopping == 2:
                self.periodic_eval()
                ppc_rel = self.all_errs[-1]["PostgresPlanCost-C-Relative-val"]

                if len(eplosses) >= 1:
                    pct = 100 * ((ppc_rel - eplosses[-1]) / eplosses[-1])
                    pct_chngs.append(pct)

                eplosses.append(ppc_rel)

                if self.epoch > 2 and pct_chngs[-1] > 1:
                    print(eplosses)
                    print(pct_chngs)
                    print("Going to exit training at epoch: ", self.epoch)
                    self.best_model_epoch = self.epoch - 1
                    break

        if self.best_model_epoch != -1:
            print("training done, will update our model based on validation set")
            assert len(self.model_weights) > 0
            self.net.load_state_dict(self.model_weights[self.best_model_epoch])

        adv_plot_path = self._save_adversarial_training_plot(run_plot_dir)
        if adv_plot_path is not None:
            print("Saved adversarial training plot to:", adv_plot_path)

        adv_qerr_plot_path = self._save_qerror_plot(
            self.adversarial_train_history,
            run_plot_dir,
            "adversarial_val_qerror.png",
            "Q-Error by Epoch",
        )
        if adv_qerr_plot_path is not None:
            print("Saved adversarial q-error plot to:", adv_qerr_plot_path)

        adv_mmd_plot_path = self._save_latent_mmd_plot(
            self.adversarial_train_history,
            run_plot_dir,
            "adversarial_eval_latent_mmd.png",
            "Latent MMD by Eval Dataset",
        )
        if adv_mmd_plot_path is not None:
            print("Saved adversarial latent MMD plot to:", adv_mmd_plot_path)
        self._maybe_save_latent_visualization(run_plot_dir, force=True)

        self.save_model(save_dir="./saved_models", suffix_name="_epoch" + str(self.epoch))

    def train_with_dann(self, training_samples, **kwargs):
        self._adversarial_epoch_train_fn = self.train_one_epoch_dann
        try:
            self.train_with_new_discriminator(training_samples, **kwargs)
        finally:
            if hasattr(self, "_adversarial_epoch_train_fn"):
                del self._adversarial_epoch_train_fn

    def train_with_latent_generator(self, training_samples, **kwargs):
        self.all_errs = []
        self.best_model_epoch = -1
        self.model_weights = []
        self.adv_weights = None
        self.adv_weight_level = kwargs.get("adv_weight_level", "dataset")
        self.adversarial_train_history = []

        if self.loss_func_name == "flowloss" or self.load_query_together:
            raise RuntimeError(
                "train_with_latent_generator currently supports non-flowloss training only."
            )

        self.true_costs = {}
        self.true_costs["val"] = 0.0
        self.true_costs["test"] = 0.0

        assert isinstance(training_samples[0], dict)
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples
        target_samples = self._collect_target_queries(kwargs)

        self.seen_subplans = set()
        for sample in training_samples:
            for node in sample["subset_graph"].nodes():
                self.seen_subplans.add(str(node))

        if "subplan_mask" in kwargs:
            subplan_mask = kwargs["subplan_mask"]
        else:
            subplan_mask = None
        val_subplan_mask = kwargs.get("val_subplan_mask", None)
        test_subplan_mask = kwargs.get("test_subplan_mask", None)

        self.trainds = self.init_dataset(
            training_samples,
            self.load_query_together,
            max_num_tables=self.max_num_tables,
            load_padded_mscn_feats=self.load_padded_mscn_feats,
            subplan_mask=subplan_mask,
        )

        if "adv_weights" in kwargs and kwargs["adv_weights"] is not None:
            self.adv_weights = np.asarray(kwargs["adv_weights"], dtype=np.float32)
            if self.adv_weight_level == "dataset":
                expected = len(self.trainds)
            elif self.adv_weight_level == "query":
                expected = len(training_samples)
            else:
                raise ValueError(f"Unsupported adv_weight_level: {self.adv_weight_level}")

            if len(self.adv_weights) != expected:
                raise ValueError(
                    f"Expected {expected} adversarial weights for level "
                    f"{self.adv_weight_level}, got {len(self.adv_weights)}"
                )

        self.trainloader = data.DataLoader(
            self.trainds,
            batch_size=self.mb_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            generator=make_generator(self.train_seed, "trainloader"),
            worker_init_fn=make_worker_init_fn(self.train_seed),
        )

        self.eval_ds = {}
        self.samples = {}
        # see the note in train(): the mask each eval dataset was built with,
        # needed to line its predictions back up with its subplans
        self.eval_subplan_masks = {}

        if self.eval_epoch < self.max_epochs:
            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(
                    kwargs["valqs"], False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                    subplan_mask=val_subplan_mask,
                )
                self.samples["val"] = kwargs["valqs"]
                self.eval_subplan_masks["val"] = val_subplan_mask

            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                if len(kwargs["testqs"]) > 400:
                    ns = int(len(kwargs["testqs"]) / 10)
                    # Private RNG at a fixed seed: this is a DATA decision
                    # (which test queries get evaluated), so it must not move
                    # with --train_seed. It also used to call random.seed(42)
                    # on the global RNG, resetting the stream for everything
                    # constructed afterwards.
                    test_rng = random.Random(42)
                    if test_subplan_mask is not None:
                        # keep testqs and its mask aligned by sampling the
                        # same indices out of both
                        idxs = test_rng.sample(range(len(kwargs["testqs"])), ns)
                        testqs = [kwargs["testqs"][i] for i in idxs]
                        cur_test_subplan_mask = [test_subplan_mask[i] for i in idxs]
                    else:
                        testqs = test_rng.sample(kwargs["testqs"], ns)
                        cur_test_subplan_mask = None
                else:
                    testqs = kwargs["testqs"]
                    cur_test_subplan_mask = test_subplan_mask

                self.eval_ds["test"] = self.init_dataset(
                    testqs, False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                    subplan_mask=cur_test_subplan_mask,
                )
                self.samples["test"] = testqs
                self.eval_subplan_masks["test"] = cur_test_subplan_mask

            if "evalqs" in kwargs and len(kwargs["eval_qdirs"]) > 0:
                eval_qdirs = kwargs["eval_qdirs"]
                for ei, cur_evalqs in enumerate(kwargs["evalqs"]):
                    evalqname = eval_qdirs[ei]
                    if "job" in evalqname:
                        evalqname = "JOB"
                        print("Going to remove JOB Q29 from evaluation because it takes too long for computing PPC")
                        cur_evalqs = [q for q in cur_evalqs if "29" not in q["name"]]
                    elif "imdb" in evalqname:
                        group_evalqs = [q for q in cur_evalqs if "group" in q["sql"].lower()]
                        not_group_evalqs = [q for q in cur_evalqs if "group" not in q["sql"].lower()]
                        gqname = "CEB-IMDb-Complex"
                        not_gqname = "CEB-IMDb-NoGroupNoLike"
                        self.eval_ds[gqname] = self.init_dataset(
                            group_evalqs, False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats,
                        )
                        self.true_costs[gqname] = 0.0
                        self.samples[gqname] = group_evalqs
                        self.eval_subplan_masks[gqname] = None

                        self.eval_ds[not_gqname] = self.init_dataset(
                            not_group_evalqs, False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats,
                        )
                        self.true_costs[not_gqname] = 0.0
                        self.samples[not_gqname] = not_group_evalqs
                        self.eval_subplan_masks[not_gqname] = None
                        continue
                    elif "stats" in evalqname:
                        evalqname = "Stats-CEB"

                    print("{}, num eval queries: {}".format(evalqname, len(cur_evalqs)))
                    if len(cur_evalqs) == 0:
                        continue

                    self.eval_ds[evalqname] = self.init_dataset(
                        cur_evalqs, False,
                        load_padded_mscn_feats=self.load_padded_mscn_feats,
                    )
                    self.true_costs[evalqname] = 0.0
                    self.samples[evalqname] = cur_evalqs
                    self.eval_subplan_masks[evalqname] = None

        self.net, self.optimizer = self.init_net(self.trainds[0])
        if not hasattr(self.net, "forward_with_latent") or \
                not hasattr(self.net, "discriminate") or \
                not hasattr(self.net, "generate"):
            raise RuntimeError(
                "train_with_latent_generator requires an MSCN-style network "
                "with forward_with_latent(), discriminate(), and generate()."
            )

        self._ensure_latent_discriminator_ready()
        self._ensure_latent_generator_ready()
        self._init_latent_generator_optimizers()

        if self.training_opt == "swa":
            raise RuntimeError(
                "SWA training is not supported in train_with_latent_generator."
            )

        run_plot_dir = self._prepare_run_plot_dir(kwargs)
        self._setup_latent_visualization_loaders(target_samples=target_samples)

        model_size = self.num_parameters()
        print(
            "Training samples: {}, Model size: {}".format(
                len(self.trainds), model_size
            )
        )

        if self.max_epochs == -1:
            total_epochs = 1000
        else:
            total_epochs = self.max_epochs

        if self.early_stopping:
            eplosses = []
            pct_chngs = []

        for self.epoch in range(0, total_epochs):
            should_eval = (self.epoch % self.eval_epoch == 0)
            if should_eval:
                self.periodic_eval()

            epoch_metrics = self.train_one_epoch_with_latent_generator()

            if should_eval:
                train_qerr_metrics = self._compute_train_qerror_metrics(
                        subplan_mask)
                if train_qerr_metrics["mean"] is not None:
                    epoch_metrics["train_qerr"] = train_qerr_metrics["mean"]
                    epoch_metrics["train_qerr_median"] = train_qerr_metrics["median"]
                    epoch_metrics["train_qerr_p99"] = train_qerr_metrics["p99"]

            if should_eval and "val" in self.eval_ds:
                val_preds, val_ys = self._eval_ds(self.eval_ds["val"], self.samples["val"])
                val_loss_metrics = self._compute_loss_metrics_from_numpy(val_preds, val_ys)
                if val_loss_metrics["mean"] is not None:
                    epoch_metrics["val_loss"] = val_loss_metrics["mean"]
                    epoch_metrics["val_loss_median"] = val_loss_metrics["median"]
                    epoch_metrics["val_loss_p99"] = val_loss_metrics["p99"]
                val_qerr_metrics = self._compute_qerror_metrics_from_numpy(
                    val_preds,
                    self.samples.get("val"),
                    subplan_mask=val_subplan_mask,
                )
                if val_qerr_metrics["mean"] is not None:
                    epoch_metrics["val_qerr"] = val_qerr_metrics["mean"]
                    epoch_metrics["val_qerr_median"] = val_qerr_metrics["median"]
                    epoch_metrics["val_qerr_p99"] = val_qerr_metrics["p99"]

            if should_eval:
                eval_qerrs = self._compute_eval_qerror_metrics()
                if len(eval_qerrs) > 0:
                    epoch_metrics["eval_qerrs"] = eval_qerrs
                    print(
                        "Epoch {} eval_qerrs={}".format(
                            self.epoch,
                            self._format_eval_qerrs(eval_qerrs),
                        )
                    )

            if should_eval:
                eval_latent_mmds = self.compute_eval_latent_mmds()
                if len(eval_latent_mmds) > 0:
                    epoch_metrics["eval_latent_mmds"] = eval_latent_mmds
                    epoch_metrics["latent_mmd"] = float(np.mean(list(eval_latent_mmds.values())))
                    self.latest_eval_latent_mmds = eval_latent_mmds
                    print(
                        "Epoch {} latent_mmds={}".format(
                            self.epoch,
                            self._format_eval_latent_mmds(eval_latent_mmds),
                        )
                    )

            self._maybe_save_latent_visualization(run_plot_dir)

            self.adversarial_train_history.append(epoch_metrics)
            self.model_weights.append(copy.deepcopy(self.net.state_dict()))

            if self.epoch % 2 == 0:
                print(
                    "Epoch {} took {}s, reg_loss={}, disc_loss={}, gen_loss={}, disc_acc={}, source_acc={}, fake_acc={}, fool_acc={}, latent_mmd={}, train_qerr={}(median={}, 99p={}), val_qerr={}(median={}, 99p={})".format(
                        self.epoch,
                        epoch_metrics["epoch_seconds"],
                        round(epoch_metrics["loss_reg"], 6),
                        round(epoch_metrics["loss_d"], 6),
                        round(epoch_metrics["loss_g"], 6),
                        round(epoch_metrics["disc_acc"], 6),
                        round(epoch_metrics["disc_acc_source"], 6),
                        round(epoch_metrics["disc_acc_fake"], 6),
                        round(epoch_metrics["gen_fool_acc"], 6),
                        round(epoch_metrics.get("latent_mmd", float("nan")), 6),
                        round(epoch_metrics.get("train_qerr", float("nan")), 6),
                        round(epoch_metrics.get("train_qerr_median", float("nan")), 6),
                        round(epoch_metrics.get("train_qerr_p99", float("nan")), 6),
                        round(epoch_metrics.get("val_qerr", float("nan")), 6),
                        round(epoch_metrics.get("val_qerr_median", float("nan")), 6),
                        round(epoch_metrics.get("val_qerr_p99", float("nan")), 6),
                    )
                )

            if self.use_wandb:
                wandb_payload = {
                    "TrainLoss": epoch_metrics["loss_reg"],
                    "ReconLoss": epoch_metrics.get("loss_recon", 0.0),
                    "Phase1Loss": epoch_metrics.get("loss_phase1", epoch_metrics["loss_reg"]),
                    "Adv-Loss-D": epoch_metrics["loss_d"],
                    "Adv-Loss-G": epoch_metrics["loss_g"],
                    "Adv-Disc-Acc": epoch_metrics["disc_acc"],
                    "Adv-Disc-Source-Acc": epoch_metrics["disc_acc_source"],
                    "Adv-Disc-Fake-Acc": epoch_metrics["disc_acc_fake"],
                    "Adv-Gen-Fool-Acc": epoch_metrics["gen_fool_acc"],
                    "epoch": self.epoch,
                }
                if "latent_mmd" in epoch_metrics:
                    wandb_payload["LatentMMD"] = epoch_metrics["latent_mmd"]
                if "train_qerr" in epoch_metrics:
                    wandb_payload["TrainQError"] = epoch_metrics["train_qerr"]
                if "train_qerr_median" in epoch_metrics:
                    wandb_payload["TrainQError-Median"] = epoch_metrics["train_qerr_median"]
                if "train_qerr_p99" in epoch_metrics:
                    wandb_payload["TrainQError-99p"] = epoch_metrics["train_qerr_p99"]
                if "val_qerr" in epoch_metrics:
                    wandb_payload["ValQError"] = epoch_metrics["val_qerr"]
                if "val_qerr_median" in epoch_metrics:
                    wandb_payload["ValQError-Median"] = epoch_metrics["val_qerr_median"]
                if "val_qerr_p99" in epoch_metrics:
                    wandb_payload["ValQError-99p"] = epoch_metrics["val_qerr_p99"]
                for name, metrics in epoch_metrics.get("eval_qerrs", {}).items():
                    wandb_payload["EvalQError-" + name] = metrics["mean"]
                    wandb_payload["EvalQError-" + name + "-Median"] = metrics["median"]
                    wandb_payload["EvalQError-" + name + "-99p"] = metrics["p99"]
                wandb.log(wandb_payload)

            if self.early_stopping == 1:
                if "val" in self.eval_ds:
                    ds = self.eval_ds["val"]
                else:
                    ds = self.eval_ds["train"]

                preds, ys = self._eval_ds(ds)
                losses = self.loss_func(torch.from_numpy(preds), torch.from_numpy(ys))
                eploss = torch.mean(losses).item()
                if len(eplosses) >= 1:
                    pct = 100 * ((eploss - eplosses[-1]) / eplosses[-1])
                    pct_chngs.append(pct)

                eplosses.append(eploss)
                if len(pct_chngs) > 5:
                    trailing_chng = np.mean(pct_chngs[-5:-1])
                    if trailing_chng > -0.1:
                        print("Going to exit training at epoch: ", self.epoch)
                        break

            elif self.early_stopping == 2:
                self.periodic_eval()
                ppc_rel = self.all_errs[-1]["PostgresPlanCost-C-Relative-val"]

                if len(eplosses) >= 1:
                    pct = 100 * ((ppc_rel - eplosses[-1]) / eplosses[-1])
                    pct_chngs.append(pct)

                eplosses.append(ppc_rel)

                if self.epoch > 2 and pct_chngs[-1] > 1:
                    print(eplosses)
                    print(pct_chngs)
                    print("Going to exit training at epoch: ", self.epoch)
                    self.best_model_epoch = self.epoch - 1
                    break

        if self.best_model_epoch != -1:
            print("training done, will update our model based on validation set")
            assert len(self.model_weights) > 0
            self.net.load_state_dict(self.model_weights[self.best_model_epoch])

        adv_plot_path = self._save_adversarial_training_plot(run_plot_dir)
        if adv_plot_path is not None:
            print("Saved adversarial training plot to:", adv_plot_path)

        adv_qerr_plot_path = self._save_qerror_plot(
            self.adversarial_train_history,
            run_plot_dir,
            "latent_generator_val_qerror.png",
            "Q-Error by Epoch",
        )
        if adv_qerr_plot_path is not None:
            print("Saved latent-generator q-error plot to:", adv_qerr_plot_path)

        adv_mmd_plot_path = self._save_latent_mmd_plot(
            self.adversarial_train_history,
            run_plot_dir,
            "latent_generator_eval_latent_mmd.png",
            "Latent MMD by Eval Dataset",
        )
        if adv_mmd_plot_path is not None:
            print("Saved adversarial latent MMD plot to:", adv_mmd_plot_path)
        self._maybe_save_latent_visualization(run_plot_dir, force=True)

        self.save_model(save_dir="./saved_models", suffix_name="_epoch" + str(self.epoch))

    def _eval_ds(self, ds, samples=None):
        torch.set_grad_enabled(False)

        if self.training_opt == "swa":
            net = self.swa_net
        else:
            net = self.net

        if DEBUG_TIMES:
            torch.set_num_threads(1)
            start = time.time()
            batchsize = 2
        else:
            batchsize = self.mb_size

        # important to not shuffle the data so correct order preserved!
        # also, assuming we are not loading everything in memory for
        # evaluation stuff, therefore collate_fn set
        if "flowloss" in self.loss_func_name:
            loader = data.DataLoader(ds,
                    batch_size=len(ds), shuffle=False,
                    collate_fn = None
                    )
        else:
            loader = data.DataLoader(ds,
                    batch_size=batchsize, shuffle=False,
                    collate_fn = self.collate_fn
                    )

        allpreds = []
        allys = []

        for (xbatch,ybatch,info) in loader:
            ybatch = ybatch.to(device, non_blocking=True)

            if self.test_random_bitmap:
                # Own generator: this runs during evaluation, so drawing from
                # the global torch RNG would shift the dropout stream of the
                # training that follows -- i.e. how often you evaluate would
                # change how the model trains.
                if self.featurizer.join_features:
                    print("testing with randomized idxs")
                    idxs = torch.randperm(xbatch["join"].shape[-1],
                            generator=self._test_bitmap_gen)
                    xbatch["join"] = xbatch["join"][:,:,idxs]
                if self.featurizer.sample_bitmap and \
                        self.featurizer.table_features:
                    print("testing with randomized idxs")
                    idxs = torch.randperm(xbatch["table"].shape[-1],
                            generator=self._test_bitmap_gen)
                    xbatch["table"] = xbatch["table"][:,:,idxs]

            if self.mask_unseen_subplans:
                start = time.time()
                pf_mask = torch.from_numpy(self.featurizer.pred_onehot_mask).float()
                jf_mask = torch.from_numpy(self.featurizer.join_onehot_mask).float()
                tf_mask = torch.from_numpy(self.featurizer.table_onehot_mask).float()

                for ci,curnode in enumerate(info["node"]):
                    if not curnode in self.seen_subplans:
                        if self.featurizer.pred_features:
                            xbatch["pred"][ci] = xbatch["pred"][ci] * pf_mask
                        if self.featurizer.join_features:
                            xbatch["join"][ci] = xbatch["join"][ci] * jf_mask
                        if self.featurizer.table_features:
                            xbatch["table"][ci] = xbatch["table"][ci] * tf_mask

                # print("masking unseen subplans took: ", time.time()-start)


            if self.subplan_level_outputs:
                pred = net(xbatch).squeeze(1)
                idxs = torch.zeros(pred.shape,dtype=torch.bool)
                for i, nt in enumerate(info["num_tables"]):
                    if nt >= 10:
                        nt = 10
                    nt -= 1
                    idxs[i,nt] = True
                pred = pred[idxs]
            else:
                pred = net(xbatch).squeeze(1)

            allpreds.append(pred)
            allys.append(ybatch)

        if DEBUG_TIMES:
            print("eval ds for {} took: {}".format(len(allpreds[0])*len(allpreds),
                round((time.time()-start)*1000, 6)))
            print("excluding input layer time: ", round(net.total_fwd_time*1000, 6))

            pdb.set_trace()

        preds = torch.cat(allpreds).detach().cpu().numpy()
        ys = torch.cat(allys).detach().cpu().numpy()

        torch.set_grad_enabled(True)

        if self.heuristic_unseen_preds == "pg" and samples is not None:
            newpreds = []
            query_idx = 0
            for sample in samples:
                node_keys = list(sample["subset_graph"].nodes())
                if SOURCE_NODE in node_keys:
                    node_keys.remove(SOURCE_NODE)
                node_keys.sort()
                for subq_idx, node in enumerate(node_keys):
                    cards = sample["subset_graph"].nodes()[node]["cardinality"]
                    idx = query_idx + subq_idx
                    est_card = preds[idx]
                    # were all columns in this subplan + constants seen in the
                    # training set?
                    print(node)

                    pdb.set_trace()

            preds = np.array(newpreds)
            pdb.set_trace()

        return preds, ys

    def _get_onehot_mask(self, vec):
        tmask = ~np.array(vec, dtype="bool")
        ptrue = self.onehot_mask_truep
        pfalse = 1-self.onehot_mask_truep

        # probabilities are switched
        bools = self._mask_rng.choice(a=[False, True], size=(len(tmask),),
                p=[ptrue,pfalse])
        tmask *= bools
        tmask = ~tmask
        tmask = torch.from_numpy(tmask).float()
        return tmask

    # def _get_onehot_mask2(self, xbatch):

        # if self.onehot_dropout:
            # # doesn't depend on xbatch at all
            # tf_mask = self._get_onehot_mask(self.featurizer.table_onehot_mask)
            # jf_mask = self._get_onehot_mask(self.featurizer.join_onehot_mask)
            # pf_mask = self._get_onehot_mask(self.featurizer.pred_onehot_mask)

            # return tf_mask, jf_mask, pf_mask

        # else:
            # assert False

        # tf_mask = torch.from_numpy(tf_mask).float()
        # jf_mask = torch.from_numpy(jf_mask).float()
        # pf_mask = torch.from_numpy(pf_mask).float()
        # return tf_mask, jf_mask, pf_mask

    def _extract_info_field(self, info, field):
        if isinstance(info, dict):
            values = info[field]
            if torch.is_tensor(values):
                return values.detach().cpu().tolist()
            elif isinstance(values, np.ndarray):
                return values.tolist()
            else:
                return list(values)

        if len(info) == 0:
            return []

        if isinstance(info[0], dict):
            return [cur[field] for cur in info]

        if isinstance(info[0], list):
            # Query-grouped batches such as flowloss keep one info list per query.
            return [cur[0][field] for cur in info]

        raise TypeError(f"Unsupported info container type: {type(info)}")

    def _get_adversarial_batch_weights(self, info, losses):
        weights = getattr(self, "adv_weights", None)
        if weights is None:
            return None

        level = getattr(self, "adv_weight_level", "dataset")
        if level == "dataset":
            idx_field = "dataset_idx"
        elif level == "query":
            idx_field = "query_idx"
        else:
            raise ValueError(f"Unsupported adv_weight_level: {level}")

        batch_idxs = self._extract_info_field(info, idx_field)
        batch_weights = torch.as_tensor(
            weights[batch_idxs],
            dtype=losses.dtype,
            device=losses.device,
        )
        return batch_weights.reshape(losses.shape)

    def _reduce_losses(self, losses, info):
        if len(losses.shape) == 0:
            return losses

        batch_weights = self._get_adversarial_batch_weights(info, losses)
        if batch_weights is None:
            return losses.sum() / len(losses)

        weighted_losses = losses * batch_weights
        return weighted_losses.sum() / batch_weights.sum().clamp_min(1e-8)

    def train_one_epoch(self):
        if self.loss_func_name == "flowloss":
            torch.set_num_threads(1)

        start = time.time()
        backtimes = []
        ftimes = []
        epoch_losses = []

        for idx, (xbatch, ybatch, info) in enumerate(self.trainloader):
            # TODO: load_query_together things
            ybatch = ybatch.to(device, non_blocking=True)

            if self.random_bitmap_idx:
                idxs = torch.randperm(xbatch["join"].shape[-1],
                        generator=self._batch_transform_gen)
                xbatch["join"] = xbatch["join"][:,:,idxs]

            if self.onehot_dropout == 0:
                pass

            elif self.onehot_dropout:
                if self.featurizer.featurization_type == "combined":
                    mask = np.zeros(xbatch.shape[1])
                    mask[-1] = 1
                    mask[-3] = 1
                    mask[-4] = 1
                    mask = self._get_onehot_mask(mask)
                    xbatch = xbatch*mask
                else:
                    tf_mask = self._get_onehot_mask(self.featurizer.table_onehot_mask)
                    jf_mask = self._get_onehot_mask(self.featurizer.join_onehot_mask)
                    pf_mask = self._get_onehot_mask(self.featurizer.pred_onehot_mask)

                    if self.featurizer.pred_features:
                        xbatch["pred"] = xbatch["pred"] * pf_mask
                    if self.featurizer.join_features:
                        xbatch["join"] = xbatch["join"] * jf_mask
                    if self.featurizer.table_features:
                        xbatch["table"] = xbatch["table"] * tf_mask
            # else:
                # # tf_mask = self._get_onehot_mask(self.featurizer.table_onehot_mask)
                # # jf_mask = self._get_onehot_mask(self.featurizer.join_onehot_mask)
                # # pf_mask = self._get_onehot_mask(self.featurizer.pred_onehot_mask)
                # tf_mask, jf_mask, pf_mask = self._get_onehot_mask2(xbatch)

                # if self.featurizer.pred_features:
                    # xbatch["pred"] = xbatch["pred"] * pf_mask
                # if self.featurizer.join_features:
                    # xbatch["join"] = xbatch["join"] * jf_mask
                # if self.featurizer.table_features:
                    # xbatch["table"] = xbatch["table"] * tf_mask

            if self.subplan_level_outputs:
                pred = self.net(xbatch).squeeze(1)
                idxs = torch.zeros(pred.shape,dtype=torch.bool)
                for i, nt in enumerate(info["num_tables"]):
                    if nt >= 10:
                        nt = 10
                    nt -= 1
                    idxs[i,nt] = True
                pred = pred[idxs]
            else:
                pred = self.net(xbatch).squeeze(1)

            assert pred.shape == ybatch.shape

            if self.loss_func_name == "flowloss":
                assert self.load_query_together
                qstart = 0
                losses = []

                for cur_info in info:
                    if "query_idx" not in cur_info[0]:
                        print(cur_info)
                        pdb.set_trace()
                    qidx = cur_info[0]["query_idx"]
                    assert qidx == cur_info[1]["query_idx"]
                    subsetg_vectors, trueC_vec, opt_loss = \
                            self.flow_training_info[qidx]

                    assert len(subsetg_vectors) == 10
                    fstart = time.time()

                    cur_loss = self.loss_func(
                            pred[qstart:qstart+len(cur_info)],
                            ybatch[qstart:qstart+len(cur_info)],
                            self.featurizer.ynormalization,
                            self.featurizer.min_val,
                            self.featurizer.max_val,
                            [(subsetg_vectors, trueC_vec, opt_loss)],
                            self.normalize_flow_loss,
                            None,
                            self.cost_model)
                    ftimes.append(time.time()-fstart)
                    losses.append(cur_loss)
                    qstart += len(cur_info)

                losses = torch.stack(losses)
                loss = self._reduce_losses(losses, info)
            elif self.loss_func_name == "qloss" and \
                self.featurizer.ynormalization == "log":
                # unnormalize both pred and ybatch
                pred = self.featurizer.unnormalize_torch(pred, None)
                ybatch = self.featurizer.unnormalize_torch(ybatch, None)
                losses = self.loss_func(pred, ybatch)
                loss = self._reduce_losses(losses, info)
            else:
                losses = self.loss_func(pred, ybatch)
                loss = self._reduce_losses(losses, info)

            epoch_losses.append(loss.item())

            reg_loss = None
            if self.onehot_reg:
                for name, param in self.net.named_parameters():
                    if name == "sample_mlp1.weight":
                        mask = torch.from_numpy(~np.array(self.featurizer.table_onehot_mask,
                            dtype="bool")).float()
                    elif name == "join_mlp1.weight":
                        mask = torch.from_numpy(~np.array(self.featurizer.join_onehot_mask,
                            dtype="bool")).float()
                    elif name == "predicate_mlp1.weight":
                        mask = torch.from_numpy(~np.array(self.featurizer.pred_onehot_mask,
                            dtype="bool")).float()
                    else:
                        continue

                    reg_param = param*mask
                    if reg_loss is None:
                        reg_loss = reg_param.norm(p=2)
                    else:
                        reg_loss = reg_loss + reg_param.norm(p=2)

            elif self.reg_loss:
                reg_loss = sum(torch.linalg.norm(p, 1) for p in
                        self.net.parameters())

            if reg_loss is not None:
                assert False
                loss += self.onehot_reg_decay * reg_loss

            if self.training_opt == "swa":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.epoch > self.swa_start:
                    self.swa_net.update_parameters(self.net)
                    self.swa_scheduler.step()
            else:
                bstart = time.time()
                self.optimizer.zero_grad()
                loss.backward()
                backtimes.append(time.time()-bstart)
                if self.clip_gradient is not None:
                    clip_grad_norm_(self.net.parameters(), self.clip_gradient)
                self.optimizer.step()

        curloss = round(float(sum(epoch_losses))/len(epoch_losses),6)

        if self.epoch % 2 == 0:
            print("Epoch {} took {}, Avg Loss: {}, #samples: {}".format(self.epoch,
                round(time.time()-start, 2), curloss, len(self.trainds)))

        # print(np.mean(epoch_losses), np.max(epoch_losses),
                # np.min(epoch_losses))
        # print("Backward avg time: {}, Forward avg time: {}".format(\
                # np.mean(backtimes), np.mean(ftimes)))

        if self.use_wandb:
            wandb.log({"TrainLoss": curloss, "epoch":self.epoch})

        return curloss

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @subplan_mask: optional [[list(node), ...], ...] aligned with
        test_samples. When given, only those subplans are featurized/predicted
        (instead of every node of every query's full graph) -- essential when
        test_samples carry full graphs but only a subset of their subplans is
        the actual eval set (subquery-level --train_csv/--eval_csv splits),
        both to avoid featurizing ~all subplans (OOM) and to avoid scoring
        subplans that were in training. Passed straight through to
        format_model_test_output so pred->node alignment is preserved.
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subplan). Each key should be ' ' separated
        list of aliases / table names
        '''
        subplan_mask = kwargs.get("subplan_mask", None)
        testds = self.init_dataset(test_samples, False,
                max_num_tables = -1,
                load_padded_mscn_feats=self.load_padded_mscn_feats,
                subplan_mask=subplan_mask)

        start = time.time()
        preds, _ = self._eval_ds(testds, test_samples)

        if self.featurizer.card_type == "joinkey":
            # joinkey featurization is edge-based and ignores subplan_mask on
            # both sides (dataset + formatter), so it stays internally
            # consistent; nothing to thread through here.
            return format_model_test_output_joinkey(preds, test_samples, self.featurizer)
        else:
            return format_model_test_output(preds, test_samples, self.featurizer,
                    subplan_mask=subplan_mask)

    def get_exp_name(self):
        name = self.__str__()
        if not hasattr(self, "rand_id"):
            # Use a private RNG: this used to call random.seed(wall_clock),
            # which reseeded the *global* python RNG mid-run and silently
            # de-randomized/re-randomized everything downstream of the first
            # call to get_exp_name(). The id itself stays run-unique (it is
            # only a directory/name suffix, and two runs at the same
            # --train_seed are meant to be distinguishable on disk).
            t = 1000 * time.time() # current time in milliseconds
            self.rand_id = str(random.Random(int(t) % 2**32).getrandbits(32))
            print("Experiment name will be: ", name + self.rand_id)

        name += self.rand_id
        return name

    def num_parameters(self):
        def _calc_size(net):
            model_parameters = net.parameters()
            params = sum([np.prod(p.size()) for p in model_parameters])
            # convert to MB
            return params*4 / 1e6

        num_params = _calc_size(self.net)
        return num_params

    def __str__(self):
        return self.__class__.__name__

    def load_model(self, model_path, sample=None, map_location=None):
        if map_location is None:
            map_location = device

        checkpoint = torch.load(model_path, map_location=map_location)

        if not hasattr(self, "net"):
            if sample is None:
                raise RuntimeError("Provide a sample to init_net() before loading weights.")
            self.net, self.optimizer = self.init_net(sample)

        self.net.load_state_dict(checkpoint["model_state"])
        if hasattr(self, "optimizer") and checkpoint.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "epoch" in checkpoint and checkpoint["epoch"] is not None:
            self.epoch = checkpoint["epoch"]

        self.net.eval() #switches the PyTorch model into evaluation mode. 
        print("Loaded model from: {}".format(model_path))
        
    def save_model(self, save_dir="./", suffix_name=""):
        if not hasattr(self, "net"):
            raise RuntimeError("Model network is not initialized; train or init_net first.")

        os.makedirs(save_dir, exist_ok=True)
        exp_name = self.get_exp_name()
        if suffix_name:
            fname = "{}_{}.pt".format(exp_name, suffix_name)
        else:
            fname = "{}.pt".format(exp_name)
        save_path = os.path.join(save_dir, fname)

        payload = {
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "epoch": getattr(self, "epoch", None),
            "alg_name": self.__str__(),
        }
        torch.save(payload, save_path)
        print("Saved model to: {}".format(save_path))


class SavedPreds(CardinalityEstimationAlg):
    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        self.model_dir = kwargs["model_dir"]
        self.max_epochs = 0

    def train(self, training_samples, **kwargs):
        assert os.path.exists(self.model_dir)
        self.saved_preds = load_object_gzip(self.model_dir + "/preds.pkl")

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subquery). Each key should be ' ' separated
        list of aliases / table names
        '''
        preds = []
        for sample in test_samples:
            assert sample["name"] in self.saved_preds
            preds.append(self.saved_preds[sample["name"]])
        return preds

    def get_exp_name(self):
        old_name = os.path.basename(self.model_dir)
        name = "SavedRun-" + old_name
        return name

    def num_parameters(self):
        '''
        size of the parameters needed so we can compare across different algorithms.
        '''
        return 0

    def __str__(self):
        return "SavedAlg"

    def save_model(self, save_dir="./", suffix_name=""):
        pass

class MSSQL(CardinalityEstimationAlg):
    def __init__(self, *args, **kwargs):
        # TODO: set each of the kwargs as variables
        self.kind = kwargs["kind"]

    def test(self, test_samples, **kwargs):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            nodes = list(sample["subset_graph"].nodes())

            for alias_key in nodes:
                info = sample["subset_graph"].nodes()[alias_key]
                true_card = info["cardinality"]["actual"]
                if "expected" not in info["cardinality"]:
                    print("expected not in Postgres!")
                    pdb.set_trace()
                    continue
                est = float(info["cardinality"][self.kind])
                # err = float(info["cardinality"]["actual"]) / float(est)
                # if err >= 10000:
                    # print(info["cardinality"])
                    # print(alias_key)
                    # print(err)
                    # pdb.set_trace()

                # elif err <= 0.00001:
                    # print(info["cardinality"])
                    # print(alias_key)
                    # print(err)
                    # pdb.set_trace()

                pred_dict[(alias_key)] = est

            preds.append(pred_dict)
        return preds

    def get_exp_name(self):
        return self.__str__() + "-" + self.kind

    def __str__(self):
        return "MSSQl"

class Postgres(CardinalityEstimationAlg):
    def test(self, test_samples, **kwargs):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            nodes = list(sample["subset_graph"].nodes())

            for alias_key in nodes:
                info = sample["subset_graph"].nodes()[alias_key]
                true_card = info["cardinality"]["actual"]
                if "expected" not in info["cardinality"]:
                    print("expected not in Postgres!")
                    pdb.set_trace()
                    continue
                est = info["cardinality"]["expected"]
                pred_dict[(alias_key)] = est

            preds.append(pred_dict)
        return preds

    def get_exp_name(self):
        return self.__str__()

    def __str__(self):
        return "Postgres"

class TrueCardinalities(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            nodes = list(sample["subset_graph"].nodes())
            if SOURCE_NODE in nodes:
                nodes.remove(SOURCE_NODE)
            for alias_key in nodes:
                info = sample["subset_graph"].nodes()[alias_key]
                pred_dict[(alias_key)] = info["cardinality"]["actual"]
            preds.append(pred_dict)
        return preds

    def get_exp_name(self):
        return self.__str__()

    def __str__(self):
        return "True"

class TrueRandom(CardinalityEstimationAlg):
    def __init__(self):
        # max percentage noise added / subtracted to true values
        self.max_noise = random.randint(1,500)

    def test(self, test_samples):
        # choose noise type
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                true_card = info["cardinality"]["actual"]
                # add noise
                noise_perc = random.randint(1,self.max_noise)
                noise = (true_card * noise_perc) / 100.00
                if random.random() % 2 == 0:
                    updated_card = true_card + noise
                else:
                    updated_card = true_card - noise
                if updated_card <= 0:
                    updated_card = 1
                pred_dict[(alias_key)] = updated_card
            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_random"

class TrueRank(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            all_cards = []
            for alias_key, info in sample["subset_graph"].nodes().items():
                # pred_dict[(alias_key)] = info["cardinality"]["actual"]
                card = info["cardinality"]["actual"]
                exp = info["cardinality"]["expected"]
                all_cards.append([alias_key, card, exp])
            all_cards.sort(key = lambda x : x[1])

            for i, (alias_key, true_est, pgest) in enumerate(all_cards):
                if i == 0:
                    pred_dict[(alias_key)] = pgest
                    continue
                prev_est = all_cards[i-1][2]
                prev_alias = all_cards[i-1][0]
                if pgest >= prev_est:
                    pred_dict[(alias_key)] = pgest
                else:
                    updated_est = prev_est
                    # updated_est = prev_est + 1000
                    # updated_est = true_est
                    all_cards[i][2] = updated_est
                    pred_dict[(alias_key)] = updated_est

            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_rank"

class TrueRankTables(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            all_cards_nt = defaultdict(list)
            for alias_key, info in sample["subset_graph"].nodes().items():
                # pred_dict[(alias_key)] = info["cardinality"]["actual"]
                card = info["cardinality"]["actual"]
                exp = info["cardinality"]["expected"]
                nt = len(alias_key)
                all_cards_nt[nt].append([alias_key,card,exp])

            for _,all_cards in all_cards_nt.items():
                all_cards.sort(key = lambda x : x[1])
                for i, (alias_key, true_est, pgest) in enumerate(all_cards):
                    if i == 0:
                        pred_dict[(alias_key)] = pgest
                        continue
                    prev_est = all_cards[i-1][2]
                    prev_alias = all_cards[i-1][0]
                    if pgest >= prev_est:
                        pred_dict[(alias_key)] = pgest
                    else:
                        updated_est = prev_est
                        # updated_est = prev_est + 1000
                        # updated_est = true_est
                        all_cards[i][2] = updated_est
                        pred_dict[(alias_key)] = updated_est

            preds.append(pred_dict)
        return preds

    def __str__(self):
        return "true_rank_tables"

class Random(CardinalityEstimationAlg):
    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        preds = []
        for sample in test_samples:
            pred_dict = {}
            for alias_key, info in sample["subset_graph"].nodes().items():
                total = info["cardinality"]["total"]
                est = random.random()*total
                pred_dict[(alias_key)] = est
            preds.append(pred_dict)
        return preds

class XGBoost(CardinalityEstimationAlg):
    def __init__(self, **kwargs):
        for k, val in kwargs.items():
            self.__setattr__(k, val)

    def init_dataset(self, samples):
        ds = QueryDataset(samples, self.featurizer, False)
        X = ds.X.cpu().numpy()
        Y = ds.Y.cpu().numpy()
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        del(ds)
        return X, Y

    def load_model(self, model_dir):
        import xgboost as xgb
        model_path = model_dir + "/xgb_model.json"
        import xgboost as xgb
        self.xgb_model = xgb.XGBRegressor(objective="reg:squarederror")
        self.xgb_model.load_model(model_path)
        print("*****loaded model*****")

    def train(self, training_samples, **kwargs):
        import xgboost as xgb
        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        X,Y = self.init_dataset(training_samples)

        if self.grid_search:
            parameters = {'learning_rate':(0.001, 0.01),
                    'n_estimators':(100, 250, 500, 1000),
                    'loss': ['ls'],
                    'max_depth':(3, 6, 8, 10),
                    'subsample':(1.0, 0.8, 0.5)}

            xgb_model = GradientBoostingRegressor()
            self.xgb_model = RandomizedSearchCV(xgb_model, parameters, n_jobs=-1,
                    verbose=1)
            self.xgb_model.fit(X, Y)
            print("*******************BEST ESTIMATOR FOUND**************")
            print(self.xgb_model.best_estimator_)
            print("*******************BEST ESTIMATOR DONE**************")
        else:
            import xgboost as xgb
            self.xgb_model = xgb.XGBRegressor(tree_method=self.tree_method,
                          objective="reg:squarederror",
                          verbosity=1,
                          scale_pos_weight=0,
                          learning_rate=self.lr,
                          colsample_bytree = 1.0,
                          subsample = self.subsample,
                          n_estimators=self.n_estimators,
                          reg_alpha = 0.0,
                          max_depth=self.max_depth,
                          gamma=0)
            self.xgb_model.fit(X,Y, verbose=1)

        if hasattr(self, "result_dir") and self.result_dir is not None:
            exp_name = self.get_exp_name()
            exp_dir = os.path.join(self.result_dir, exp_name)
            self.xgb_model.save_model(exp_dir + "/xgb_model.json")

    def test(self, test_samples):
        X,Y = self.init_dataset(test_samples)
        pred = self.xgb_model.predict(X)
        return format_model_test_output(pred, test_samples, self.featurizer)

    def __str__(self):
        return self.__class__.__name__

class RandomForest(CardinalityEstimationAlg):
    def __init__(self, **kwargs):
        for k, val in kwargs.items():
            self.__setattr__(k, val)

    def init_dataset(self, samples):
        ds = QueryDataset(samples, self.featurizer, False)
        X = ds.X.cpu().numpy()
        Y = ds.Y.cpu().numpy()
        X = np.array(X, dtype=np.float32)
        Y = np.array(Y, dtype=np.float32)
        del(ds)
        return X, Y

    def load_model(self, model_dir):
        pass

    def train(self, training_samples, **kwargs):
        from sklearn.ensemble import RandomForestRegressor

        self.featurizer = kwargs["featurizer"]
        self.training_samples = training_samples

        X,Y = self.init_dataset(training_samples)

        if self.grid_search:
            pass
        else:
            self.model = RandomForestRegressor(n_jobs=-1,
                    verbose=2,
                    n_estimators = self.n_estimators,
                    max_depth = self.max_depth)
            self.model.fit(X, Y)

    def test(self, test_samples):
        X,Y = self.init_dataset(test_samples)
        pred = self.model.predict(X)
        # FIXME: why can't we just use get_query_estimates here?
        return format_model_test_output(pred, test_samples, self.featurizer)

    def __str__(self):
        return self.__class__.__name__

def joinkey_cards_to_subplan_cards(samples, joinkey_cards,
        basecard_type, basecard_tables):

    def get_card_for_edge(cure, sample):
        newtab = set(cure[0]) - set(cure[1])
        newtab = list(newtab)[0]

        rname = sample["join_graph"].nodes()[newtab]["real_name"]
        penalty = 1.0

        r1 = cur_jcards[cure]
        jk = sg.edges()[cure]["join_key_cardinality"]
        r1_join_col = list(jk.keys())[0]

        if "." in r1_join_col:
            r1_join_tab = r1_join_col[0:r1_join_col.find(".")]
        else:
            assert False

        r1_total = cards_so_far[cure[1]]
        newtab = set(cure[0]) - set(cure[1])
        assert len(newtab) == 1
        r2_alias = tuple(newtab)
        r2_total = cards_so_far[r2_alias]

        # how to find r2? ---> find an edge where it is from the first
        # one

        joinnode = [r1_join_tab, r2_alias[0]]
        joinnode.sort()
        joinnode = tuple(joinnode)

        # find the distinct key values of r2 to get to this joinnode
        r2_edges = list(sg.out_edges(joinnode))

        r2 = None
        for e in r2_edges:
            if e[1] == r2_alias:
                r2 = cur_jcards[e]
                break
        assert r2 is not None

        if r1 == 0:
            r1 += 1
        if r2 == 0:
            r2 += 1

        # choosing this because we have more confidence in our r1 and r2
        # measurements
        if r1_total < r1:
            r1_total = r1
        if r2_total < r2:
            r2_total = r2

        card = min(r1,r2) * (r1_total/r1)*(r2_total/r2)

        card *= penalty
        return card

    assert isinstance(samples[0], dict)
    preds = []
    qdir = "./results2/mscn_query_testpreds/"

    for si, sample in enumerate(samples):
        cur_jcards = joinkey_cards[si]
        sg = sample["subset_graph"]
        nodes = list(sample["subset_graph"].nodes())
        nodes.sort(key = len)
        cards_so_far = {}
        pred_dict = {}

        for node in nodes:
            if len(node) <= basecard_tables:
                if basecard_type == "actual-err":
                    curcard = sg.nodes()[node]["cardinality"]["actual"]
                    if len(node) == 2:
                        err = random.randint(1,10)
                        curcard *= err
                elif basecard_type == "mscn":
                    qfn = os.path.basename(sample["name"])
                    qfn = os.path.join(qdir, qfn)
                    assert os.path.exists(qfn)
                    with open(qfn, "rb") as f:
                        mscncards = pickle.load(f)
                    curcard = mscncards[node]
                else:
                    curcard = sg.nodes()[node]["cardinality"][basecard_type]

                if curcard == 0:
                    curcard += 1
                cards_so_far[node] = curcard
                pred_dict[(node)] = curcard
                continue

            # find any incoming edge
            connedges = list(sg.out_edges(node))
            mcard = 0
            mincard = 1e25

            # print("Number of connected edges: ", len(connedges))
            # for each possible edge we can assign a cardinality to the current
            # node
            for e0 in connedges:
                curcard = get_card_for_edge(e0, sample)
                if curcard > mcard:
                    mcard = curcard

                # if curcard < mincard:
                    # mcard = curcard

                ## simpler heuristics of choosing the best edge
                # newtab = set(e0[0]) - set(e0[1])
                # newtab = list(newtab)[0]

                # if cards_so_far[(newtab,)] > mcard:
                    # mcard = cards_so_far[(newtab,)]
                    # cure = e0

                # if cards_so_far[(newtab,)] < mincard:
                    # mincard = cards_so_far[(newtab,)]
                    # cure = e0

                # if sg.nodes()[(newtab,)]["cardinality"]["total"] > mcard:
                    # mcard = sg.nodes()[(newtab,)]["cardinality"]["total"]
                    # cure = e0

            ## simple heuristic
            # mcard = get_card_for_edge(cure)

            cards_so_far[node] = mcard
            pred_dict[(node)] = mcard

            # if len(connedges) > 4:
                # pdb.set_trace()

        preds.append(pred_dict)

    return preds

class TrueJoinKeys(CardinalityEstimationAlg):
    def __init__(self):
        pass

    def test(self, test_samples):
        assert isinstance(test_samples[0], dict)
        all_ests = []

        for si, sample in enumerate(test_samples):
            ests = {}
            sg = sample["subset_graph"]
            edge_keys = list(sample["subset_graph"].edges())
            edge_keys.sort(key = lambda x: str(x))
            subq_idx = 0
            for _, edge in enumerate(edge_keys):
                # cards = sample["subset_graph"].nodes()[node]["cardinality"]
                edgek = edge
                # idx = query_idx + subq_idx
                # est_card = featurizer.unnormalize(pred[idx], None)
                # assert est_card >= 0
                est_card = list(sg.edges()[edge]["join_key_cardinality"].values())[0]["actual"]
                ests[edgek] = est_card
                # subq_idx += 1

            all_ests.append(ests)
            # query_idx += subq_idx
        return all_ests

    def get_exp_name(self):
        return self.__str__()

    def __str__(self):
        return "TrueJoinKeys"
