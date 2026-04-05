import time
import numpy as np
import pdb
import math
import pandas as pd
import json
import sys
import torch
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

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_

from torch.optim.swa_utils import AveragedModel, SWALR

import wandb
import random
import pickle
import matplotlib.pyplot as plt

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

def format_model_test_output(pred, samples, featurizer):
    all_ests = []
    query_idx = 0
    # print("len pred: ", len(pred))

    for si, sample in enumerate(samples):
        ests = {}
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
            idxs = torch.randperm(xbatch["join"].shape[-1])
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
            if self.epoch > getattr(self, "discriminator_warmup_epochs", 5):
                train_disc = (batch_idx % 2 == 0)
                if train_disc:
                    self.opt_discriminator.zero_grad()
                    _, z_source_det = self.net.forward_with_latent(xbatch_source)
                    _, z_target_det = self.net.forward_with_latent(xbatch_target)
                    z_source_det = z_source_det.detach()
                    z_target_det = z_target_det.detach()

                    #Label smoothing          
                    labels_source = torch.full((current_batch_size, 1), 0.9, device=device)
                    labels_target = torch.full((current_batch_size, 1), 0.1, device=device)
                    #labels_source = torch.ones(current_batch_size, 1, device=device)
                    #labels_target = torch.zeros(current_batch_size, 1, device=device)

                    pred_source_disc = self.net.discriminate(z_source_det)
                    pred_target_disc = self.net.discriminate(z_target_det)

                    loss_d_source = self.bce_loss(pred_source_disc, labels_source)
                    loss_d_target = self.bce_loss(pred_target_disc, labels_target)
                    loss_d = 0.5 * (loss_d_source + loss_d_target)
                    loss_d.backward()
                    self.opt_discriminator.step()


                # Phase 3: train encoder to fool discriminator on target.
                # Run this multiple times (e.g., 2) to let the generator catch up!
                for _ in range(1):
                    self.opt_generator.zero_grad()
                    
                    # Forward pass to get the latent vector
                    _, z_target_gen = self.net.forward_with_latent(xbatch_target)
                    
                    # Trick labels (1.0)
                    trick_labels = torch.ones(current_batch_size, 1, device=device)
                    pred_target_gen = self.net.discriminate(z_target_gen)
                    
                    #loss_g = self.bce_loss(pred_target_gen, trick_labels)
                    loss_g = lambda_adv * self.bce_loss(pred_target_gen, trick_labels)
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
                    # Compare against hard 1.0 and 0.0 for accurate logging, ignoring the 0.9/0.1 smoothing
                    hard_labels_source = torch.ones_like(labels_source)
                    hard_labels_target = torch.zeros_like(labels_target)
                    
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
            "loss_d": float(np.mean(disc_losses)) if len(disc_losses) > 0 else 0.0,
            "loss_g": float(np.mean(gen_losses)) if len(gen_losses) > 0 else 0.0,
            "disc_acc": float(np.mean(disc_accs)) if len(disc_accs) > 0 else 0.0,
            "disc_acc_source": float(np.mean(disc_acc_source)) if len(disc_acc_source) > 0 else 0.0,
            "disc_acc_target": float(np.mean(disc_acc_target)) if len(disc_acc_target) > 0 else 0.0,
            "gen_fool_acc": float(np.mean(gen_fool_accs)) if len(gen_fool_accs) > 0 else 0.0,
            "epoch_seconds": round(time.time() - start, 2),
        }
        return metrics

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

        self.seen_subplans = set()
        for sample in training_samples:
            for node in sample["subset_graph"].nodes():
                self.seen_subplans.add(str(node))

        if "subplan_mask" in kwargs:
            subplan_mask = kwargs["subplan_mask"]
        else:
            subplan_mask = None

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
                # num_workers=self.num_workers
                )

        # if self.eval_epoch >= self.max_epochs and \
            # "flowloss" not in self.loss_func_name:
            # del training_samples[1:]

        self.eval_ds = {}
        self.samples = {}

        # if self.eval_epoch < self.max_epochs:
            # self.samples["train"] = training_samples
            # self.eval_ds["train"] = self.init_dataset(training_samples,
                    # self.load_query_together,
                    # max_num_tables = -1,
                    # load_padded_mscn_feats=self.load_padded_mscn_feats)


        if self.eval_epoch < self.max_epochs:
            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(kwargs["valqs"], False,
                        load_padded_mscn_feats=self.load_padded_mscn_feats)
                self.samples["val"] = kwargs["valqs"]

            # if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                # pass
            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                if len(kwargs["testqs"]) > 400:
                    ns = int(len(kwargs["testqs"]) / 10)
                    random.seed(42)
                    testqs = random.sample(kwargs["testqs"], ns)
                else:
                    testqs = kwargs["testqs"]

                self.eval_ds["test"] = self.init_dataset(testqs,
                        False,
                        load_padded_mscn_feats=self.load_padded_mscn_feats)
                self.samples["test"] = testqs

            if "evalqs" in kwargs and len(kwargs["eval_qdirs"]) > 0:
                eval_qdirs = kwargs["eval_qdirs"]

                for ei, cur_evalqs in enumerate(kwargs["evalqs"]):
                    evalqname = eval_qdirs[ei]
                    if "job" in evalqname:
                        evalqname = "JOB"
                        print("Going to remove JOB Q29 from evaluation because it takes too long for computing PPC")
                        cur_evalqs = [q for q in cur_evalqs if "29" not in
                                q["name"]]

                    elif "imdb" in evalqname:
                        # evalqname = "CEB-IMDb"
                        group_evalqs = [q for q in cur_evalqs if "group" in
                                q["sql"].lower()]
                        not_group_evalqs = [q for q in cur_evalqs if "group" not in
                                q["sql"].lower()]
                        gqname = "CEB-IMDb-Complex"
                        not_gqname = "CEB-IMDb-NoGroupNoLike"
                        self.eval_ds[gqname] = \
                                self.init_dataset(group_evalqs, False,
                                load_padded_mscn_feats=self.load_padded_mscn_feats)
                        self.true_costs[gqname] = 0.0
                        self.samples[gqname] = group_evalqs

                        self.eval_ds[not_gqname] = \
                                self.init_dataset(not_group_evalqs, False,
                                load_padded_mscn_feats=self.load_padded_mscn_feats)
                        self.true_costs[not_gqname] = 0.0
                        self.samples[not_gqname] = not_group_evalqs

                        continue

                    elif "stats" in evalqname:
                        evalqname = "Stats-CEB"

                    print("{}, num eval queries: {}".format(evalqname,
                        len(cur_evalqs)))

                    if len(cur_evalqs) == 0:
                        continue

                    self.eval_ds[evalqname] = self.init_dataset(cur_evalqs,
                            False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats)
                    self.true_costs[evalqname] = 0.0
                    self.samples[evalqname] = cur_evalqs

        # TODO: initialize self.num_features
        self.net, self.optimizer = self.init_net(self.trainds[0])

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
            if self.epoch % self.eval_epoch == 0:
                self.periodic_eval()

            self.train_one_epoch()

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

        self.seen_subplans = set()
        for sample in training_samples:
            for node in sample["subset_graph"].nodes():
                self.seen_subplans.add(str(node))

        if "subplan_mask" in kwargs:
            subplan_mask = kwargs["subplan_mask"]
        else:
            subplan_mask = None

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
        )

        self.eval_ds = {}
        self.samples = {}

        if self.eval_epoch < self.max_epochs:
            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(
                    kwargs["valqs"], False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                )
                self.samples["val"] = kwargs["valqs"]

            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                if len(kwargs["testqs"]) > 400:
                    ns = int(len(kwargs["testqs"]) / 10)
                    random.seed(42)
                    testqs = random.sample(kwargs["testqs"], ns)
                else:
                    testqs = kwargs["testqs"]

                self.eval_ds["test"] = self.init_dataset(
                    testqs, False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                )
                self.samples["test"] = testqs

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

                        self.eval_ds[not_gqname] = self.init_dataset(
                            not_group_evalqs, False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats,
                        )
                        self.true_costs[not_gqname] = 0.0
                        self.samples[not_gqname] = not_group_evalqs
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

        self.net, self.optimizer = self.init_net(self.trainds[0])
        if not hasattr(self.net, "forward_with_latent") or \
                not hasattr(self.net, "discriminate"):
            raise RuntimeError(
                "train_with_new_discriminator requires an MSCN-style network "
                "with forward_with_latent() and discriminate()."
            )
        self._ensure_latent_discriminator_ready()
        self._init_new_discriminator_optimizers()

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

        run_result_dir = kwargs.get("result_dir", "./results")
        run_plot_dir = os.path.join(run_result_dir, self.get_exp_name())

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
        )

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

        for self.epoch in range(0, total_epochs):
            should_eval = (self.epoch % self.eval_epoch == 0)
            if self.epoch % self.eval_epoch == 0:
                self.periodic_eval()

            epoch_metrics = self.train_one_epoch_with_new_discriminator(self.target_loader)

            if should_eval and "val" in self.eval_ds:
                val_preds, val_ys = self._eval_ds(self.eval_ds["val"], self.samples["val"])
                val_loss = self._compute_mean_loss_from_numpy(val_preds, val_ys)
                if val_loss is not None:
                    epoch_metrics["val_loss"] = val_loss

            self.adversarial_train_history.append(epoch_metrics)
            self.model_weights.append(copy.deepcopy(self.net.state_dict()))

            if self.epoch % 2 == 0:
                print(
                    "Epoch {} took {}s, reg_loss={}, recon_loss={}, phase1_loss={}, disc_loss={}, gen_loss={}, disc_acc={}, source_acc={}, target_acc={}, fool_acc={}, val_loss={}".format(
                        self.epoch,
                        epoch_metrics["epoch_seconds"],
                        round(epoch_metrics["loss_reg"], 6),
                        round(epoch_metrics.get("loss_recon", 0.0), 6),
                        round(epoch_metrics.get("loss_phase1", epoch_metrics["loss_reg"]), 6),
                        round(epoch_metrics["loss_d"], 6),
                        round(epoch_metrics["loss_g"], 6),
                        round(epoch_metrics["disc_acc"], 6),
                        round(epoch_metrics["disc_acc_source"], 6),
                        round(epoch_metrics["disc_acc_target"], 6),
                        round(epoch_metrics["gen_fool_acc"], 6),
                        round(epoch_metrics.get("val_loss", float("nan")), 6),
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
                if "val_loss" in epoch_metrics:
                    wandb_payload["ValLoss"] = epoch_metrics["val_loss"]
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

        self.save_model(save_dir="./saved_models", suffix_name="_epoch" + str(self.epoch))

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

        self.seen_subplans = set()
        for sample in training_samples:
            for node in sample["subset_graph"].nodes():
                self.seen_subplans.add(str(node))

        if "subplan_mask" in kwargs:
            subplan_mask = kwargs["subplan_mask"]
        else:
            subplan_mask = None

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
        )

        self.eval_ds = {}
        self.samples = {}

        if self.eval_epoch < self.max_epochs:
            if "valqs" in kwargs and len(kwargs["valqs"]) > 0:
                self.eval_ds["val"] = self.init_dataset(
                    kwargs["valqs"], False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                )
                self.samples["val"] = kwargs["valqs"]

            if "testqs" in kwargs and len(kwargs["testqs"]) > 0:
                if len(kwargs["testqs"]) > 400:
                    ns = int(len(kwargs["testqs"]) / 10)
                    random.seed(42)
                    testqs = random.sample(kwargs["testqs"], ns)
                else:
                    testqs = kwargs["testqs"]

                self.eval_ds["test"] = self.init_dataset(
                    testqs, False,
                    load_padded_mscn_feats=self.load_padded_mscn_feats,
                )
                self.samples["test"] = testqs

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

                        self.eval_ds[not_gqname] = self.init_dataset(
                            not_group_evalqs, False,
                            load_padded_mscn_feats=self.load_padded_mscn_feats,
                        )
                        self.true_costs[not_gqname] = 0.0
                        self.samples[not_gqname] = not_group_evalqs
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

        run_result_dir = kwargs.get("result_dir", "./results")
        run_plot_dir = os.path.join(run_result_dir, self.get_exp_name())

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

            if should_eval and "val" in self.eval_ds:
                val_preds, val_ys = self._eval_ds(self.eval_ds["val"], self.samples["val"])
                val_loss = self._compute_mean_loss_from_numpy(val_preds, val_ys)
                if val_loss is not None:
                    epoch_metrics["val_loss"] = val_loss

            self.adversarial_train_history.append(epoch_metrics)
            self.model_weights.append(copy.deepcopy(self.net.state_dict()))

            if self.epoch % 2 == 0:
                print(
                    "Epoch {} took {}s, reg_loss={}, disc_loss={}, gen_loss={}, disc_acc={}, source_acc={}, fake_acc={}, fool_acc={}, val_loss={}".format(
                        self.epoch,
                        epoch_metrics["epoch_seconds"],
                        round(epoch_metrics["loss_reg"], 6),
                        round(epoch_metrics["loss_d"], 6),
                        round(epoch_metrics["loss_g"], 6),
                        round(epoch_metrics["disc_acc"], 6),
                        round(epoch_metrics["disc_acc_source"], 6),
                        round(epoch_metrics["disc_acc_fake"], 6),
                        round(epoch_metrics["gen_fool_acc"], 6),
                        round(epoch_metrics.get("val_loss", float("nan")), 6),
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
                if "val_loss" in epoch_metrics:
                    wandb_payload["ValLoss"] = epoch_metrics["val_loss"]
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
                if self.featurizer.join_features:
                    print("testing with randomized idxs")
                    idxs = torch.randperm(xbatch["join"].shape[-1])
                    xbatch["join"] = xbatch["join"][:,:,idxs]
                if self.featurizer.sample_bitmap and \
                        self.featurizer.table_features:
                    print("testing with randomized idxs")
                    idxs = torch.randperm(xbatch["table"].shape[-1])
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
        bools = np.random.choice(a=[False, True], size=(len(tmask),),
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
                idxs = torch.randperm(xbatch["join"].shape[-1])
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

    def test(self, test_samples, **kwargs):
        '''
        @test_samples: [sql_rep objects]
        @ret: [dicts]. Each element is a dictionary with cardinality estimate
        for each subset graph node (subplan). Each key should be ' ' separated
        list of aliases / table names
        '''
        testds = self.init_dataset(test_samples, False,
                max_num_tables = -1,
                load_padded_mscn_feats=self.load_padded_mscn_feats)

        start = time.time()
        preds, _ = self._eval_ds(testds, test_samples)

        if self.featurizer.card_type == "joinkey":
            return format_model_test_output_joinkey(preds, test_samples, self.featurizer)
        else:
            return format_model_test_output(preds, test_samples, self.featurizer)

    def get_exp_name(self):
        name = self.__str__()
        if not hasattr(self, "rand_id"):
            t = 1000 * time.time() # current time in milliseconds
            random.seed(int(t) % 2**32)
            self.rand_id = str(random.getrandbits(32))
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
