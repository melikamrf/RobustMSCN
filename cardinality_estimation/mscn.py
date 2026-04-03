import torch
import numpy as np
from query_representation.utils import *
from .dataset import QueryDataset, pad_sets, to_variable
from .nets import *
from .algs import *
# import cardinality_estimation.algs as algs

from torch.utils import data
from torch.nn.utils.clip_grad import clip_grad_norm_
import wandb

class MSCN(NN):

    def _infer_decoder_output_dim(self, sample_x):
        explicit_dim = getattr(self, "decoder_output_dim", None)
        if explicit_dim is not None:
            return explicit_dim

        if not getattr(self, "enable_decoder", False):
            return None

        if not getattr(self, "load_padded_mscn_feats", False):
            raise RuntimeError(
                "enable_decoder requires load_padded_mscn_feats=1 or an "
                "explicit decoder_output_dim so the reconstruction target has "
                "a stable size."
            )

        output_dim = 0
        for key in ["table", "pred", "join", "flow", "tmask", "pmask", "jmask"]:
            output_dim += int(sample_x[key].numel())
        return output_dim

    def _get_auxiliary_component_kwargs(self, sample_x):
        return {
            "enable_latent_interface": getattr(self,
                "enable_latent_interface", False),
            "enable_discriminator": getattr(self,
                "enable_discriminator", False),
            "enable_generator": getattr(self, "enable_generator", False),
            "enable_decoder": getattr(self, "enable_decoder", False),
            "discriminator_hidden_dims": getattr(self,
                "discriminator_hidden_dims", None),
            "discriminator_dropout": getattr(self,
                "discriminator_dropout", 0.1),
            "generator_noise_dim": getattr(self,
                "generator_noise_dim", None),
            "generator_hidden_dims": getattr(self,
                "generator_hidden_dims", None),
            "generator_dropout": getattr(self,
                "generator_dropout", 0.1),
            "decoder_output_dim": self._infer_decoder_output_dim(sample_x),
            "decoder_hidden_dims": getattr(self, "decoder_hidden_dims", None),
            "decoder_dropout": getattr(self, "decoder_dropout", 0.0),
        }

    def init_dataset(self, samples, load_query_together,
            max_num_tables = -1,
            load_padded_mscn_feats=False,
            subplan_mask=None):
        ds = QueryDataset(samples, self.featurizer,
                load_query_together,
                max_num_tables = max_num_tables,
                load_padded_mscn_feats=load_padded_mscn_feats,
                subplan_mask=subplan_mask)

        return ds

    def _init_net(self, sample):
        if self.load_query_together:
            sample = sample[0]

        if len(sample[0]["table"]) == 0:
            sfeats = 0
        else:
            sfeats = len(sample[0]["table"][0])

        if len(sample[0]["pred"]) == 0:
            pfeats = 0
        else:
            pfeats = len(sample[0]["pred"][0])

        if len(sample[0]["join"]) == 0:
            jfeats = 0
        else:
            jfeats = len(sample[0]["join"][0])

        if self.subplan_level_outputs:
            n_out = 10
        else:
            n_out = 1

        if self.featurizer.ynormalization in ["selectivity-log"]:
            use_sigmoid = False
        elif "whitening" in self.featurizer.ynormalization:
            use_sigmoid = False
        else:
            use_sigmoid = True

        if self.loss_func_name == "flowloss":
            net = SetConvFlow(sfeats,
                    pfeats, jfeats,
                    len(sample[0]["flow"]),
                    self.hidden_layer_size,
                    n_out=n_out,
                    num_hidden_layers = self.num_hidden_layers,
                    dropouts=[self.inp_dropout, self.hl_dropout,
                        self.comb_dropout],
                    use_sigmoid = use_sigmoid,
                    **self._get_auxiliary_component_kwargs(sample[0]))
        else:
            net = SetConv(sfeats,
                    pfeats, jfeats,
                    len(sample[0]["flow"]),
                    self.hidden_layer_size,
                    self.other_hid_units,
                    n_out=n_out,
                    num_hidden_layers = self.num_hidden_layers,
                    dropouts=[self.inp_dropout, self.hl_dropout,
                        self.comb_dropout],
                    use_sigmoid = use_sigmoid,
                    **self._get_auxiliary_component_kwargs(sample[0]))

        return net

class MSCN_JoinKeyCards(NN):

    def _infer_decoder_output_dim(self, sample_x):
        explicit_dim = getattr(self, "decoder_output_dim", None)
        if explicit_dim is not None:
            return explicit_dim

        if not getattr(self, "enable_decoder", False):
            return None

        if not getattr(self, "load_padded_mscn_feats", False):
            raise RuntimeError(
                "enable_decoder requires load_padded_mscn_feats=1 or an "
                "explicit decoder_output_dim so the reconstruction target has "
                "a stable size."
            )

        output_dim = 0
        for key in ["table", "pred", "join", "flow", "tmask", "pmask", "jmask"]:
            output_dim += int(sample_x[key].numel())
        return output_dim

    def _get_auxiliary_component_kwargs(self, sample_x):
        return {
            "enable_latent_interface": getattr(self,
                "enable_latent_interface", False),
            "enable_discriminator": getattr(self,
                "enable_discriminator", False),
            "enable_generator": getattr(self, "enable_generator", False),
            "enable_decoder": getattr(self, "enable_decoder", False),
            "discriminator_hidden_dims": getattr(self,
                "discriminator_hidden_dims", None),
            "discriminator_dropout": getattr(self,
                "discriminator_dropout", 0.1),
            "generator_noise_dim": getattr(self,
                "generator_noise_dim", None),
            "generator_hidden_dims": getattr(self,
                "generator_hidden_dims", None),
            "generator_dropout": getattr(self,
                "generator_dropout", 0.1),
            "decoder_output_dim": self._infer_decoder_output_dim(sample_x),
            "decoder_hidden_dims": getattr(self, "decoder_hidden_dims", None),
            "decoder_dropout": getattr(self, "decoder_dropout", 0.0),
        }

    def init_dataset(self, samples, load_query_together,
            max_num_tables = -1,
            load_padded_mscn_feats=False,
            subplan_mask=None):
        ds = QueryDataset(samples, self.featurizer,
                load_query_together,
                load_padded_mscn_feats=self.load_padded_mscn_feats,
                join_key_cards=True)

        return ds

    def _init_net(self, sample):
        if self.load_query_together:
            sample = sample[0]

        if len(sample[0]["table"]) == 0:
            sfeats = 0
        else:
            sfeats = len(sample[0]["table"][0])

        if len(sample[0]["pred"]) == 0:
            pfeats = 0
        else:
            pfeats = len(sample[0]["pred"][0])

        if len(sample[0]["join"]) == 0:
            jfeats = 0
        else:
            jfeats = len(sample[0]["join"][0])

        if self.subplan_level_outputs:
            n_out = 10
        else:
            n_out = 1

        if self.featurizer.ynormalization == "selectivity-log":
            use_sigmoid = False
        else:
            use_sigmoid = True

        if self.loss_func_name == "flowloss":
            net = SetConvFlow(sfeats,
                    pfeats, jfeats,
                    len(sample[0]["flow"]),
                    self.hidden_layer_size,
                    n_out=n_out,
                    dropouts=[self.inp_dropout, self.hl_dropout,
                        self.comb_dropout],
                    use_sigmoid = use_sigmoid,
                    **self._get_auxiliary_component_kwargs(sample[0]))
        else:
            net = SetConv(sfeats,
                    pfeats, jfeats,
                    len(sample[0]["flow"]),
                    self.hidden_layer_size,
                    self.other_hid_units,
                    n_out=n_out,
                    dropouts=[self.inp_dropout, self.hl_dropout,
                        self.comb_dropout],
                    use_sigmoid = use_sigmoid,
                    **self._get_auxiliary_component_kwargs(sample[0]))

        return net

    def test(self, test_samples, **kwargs):
        testds = self.init_dataset(test_samples, False)

        start = time.time()
        preds, _ = self._eval_ds(testds, test_samples)

        return format_model_test_output_joinkey(preds, test_samples, self.featurizer)

class MSCNCaptum(NN):

    def init_dataset(self, samples, load_query_together,
            max_num_tables = -1,
            load_padded_mscn_feats=False,
            **kwargs,
            ):
        ds = QueryDataset(samples, self.featurizer,
                load_query_together,
                max_num_tables = max_num_tables,
                load_padded_mscn_feats=load_padded_mscn_feats)

        return ds

    def _init_net(self, sample):
        if self.load_query_together:
            sample = sample[0]

        if len(sample[0]["table"]) == 0:
            sfeats = 0
        else:
            sfeats = len(sample[0]["table"][0])

        if len(sample[0]["pred"]) == 0:
            pfeats = 0
        else:
            pfeats = len(sample[0]["pred"][0])

        if len(sample[0]["join"]) == 0:
            jfeats = 0
        else:
            jfeats = len(sample[0]["join"][0])

        if self.subplan_level_outputs:
            n_out = 10
        else:
            n_out = 1

        if self.featurizer.ynormalization in ["selectivity-log"]:
            use_sigmoid = False
        elif "whitening" in self.featurizer.ynormalization:
            use_sigmoid = False
        else:
            use_sigmoid = True

        net = SetConvCaptum(sfeats,
                pfeats, jfeats,
                len(sample[0]["flow"]),
                self.hidden_layer_size,
                # self.other_hid_units,
                n_out=n_out,
                num_hidden_layers = self.num_hidden_layers,
                dropouts=[self.inp_dropout, self.hl_dropout,
                    self.comb_dropout])
                # use_sigmoid = use_sigmoid)

        return net
