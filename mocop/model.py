from typing import Iterable, List, Optional, Set, Tuple, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig, OmegaConf
from torch import nn as nn
from torch.nn import functional as F

from layers import GatedGraphConvolution
from metrics import accuracy
from steps import _supervised_metric, _validation_epoch_end


class MultiLayerPerceptron(nn.Module):
    """Standard multi-layer perceptron with non-linearity and potentially dropout.

    Parameters
    ----------
    num_input_features : int
        input dimension
    num_classes : int, optional
        Number of output classes. If not specified (or None), MLP does not have a final layer.
    hidden_layer_dimensions : List[int], optional
        list of hidden layer dimensions. If not provided, class is a linear model
    nonlin : Union[str, nn.Module]
        name of a nonlinearity in torch.nn, or a pytorch Module. default is relu
    p_dropout : float
        dropout probability for dropout layers. default is 0.0
    """

    def __init__(
        self,
        num_input_features: int,
        num_classes: Optional[int] = None,
        hidden_layer_dimensions: Optional[List[int]] = None,
        nonlin: Union[str, nn.Module] = "ReLU",
        p_dropout: float = 0.0,
    ):
        super(MultiLayerPerceptron, self).__init__()
        if hidden_layer_dimensions is None:
            hidden_layer_dimensions = []
        if isinstance(hidden_layer_dimensions, ListConfig):
            hidden_layer_dimensions = OmegaConf.to_object(hidden_layer_dimensions)
        if isinstance(nonlin, str):
            nonlin = getattr(torch.nn, nonlin)()

        hidden_layer_dimensions = [dim for dim in hidden_layer_dimensions if dim != 0]
        layer_inputs = [num_input_features] + hidden_layer_dimensions
        modules = []
        for i in range(len(hidden_layer_dimensions)):
            modules.extend(
                [
                    nn.Dropout(p=p_dropout),
                    nn.Linear(layer_inputs[i], layer_inputs[i + 1]),
                ]
            )
            if i < (len(hidden_layer_dimensions) - 1):
                modules.append(nonlin)

        self.module = nn.Sequential(*modules)
        if num_classes is None:
            self.has_final_layer = False
        else:
            self.has_final_layer = True
            if num_classes > 1:
                self.output_shape = (num_classes,)
            else:
                self.output_shape = ()
            output_size = num_classes
            self.final_nonlin = nonlin
            self.final_dropout = nn.Dropout(p=p_dropout)
            self.final = nn.Linear(layer_inputs[-1], output_size)

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run forward pass up to penultimate layer"""
        outputs = self.module(inputs)
        return outputs

    def forward(self, x_a: torch.Tensor, **kwargs) -> torch.Tensor:
        outputs = self.module(x_a)
        if self.has_final_layer:
            outputs = self.final_nonlin(outputs)
            outputs = self.final_dropout(outputs)
            outputs = self.final(outputs)
            outputs = torch.reshape(outputs, outputs.shape[:-1] + self.output_shape)
        return outputs


class DualInputEncoder(pl.LightningModule):
    def __init__(
        self,
        encoder_a: Optional[MultiLayerPerceptron],
        encoder_b: Optional[MultiLayerPerceptron],
        supervised_head_dim=[64, 2],
        non_lin_proj: bool = False,
        dim=128,
        temperature=10,
    ):
        super().__init__()
        self.encoder_a = encoder_a
        self.encoder_b = encoder_b
        self.dim = dim
        self.supervised_head_dim = supervised_head_dim
        self.add_module("supervised_head", None)
        self.add_module("h_a", None)
        self.add_module("h_b", None)
        self.temperature = temperature

        if non_lin_proj:
            self.proj_func = F.relu
        else:
            self.proj_func = lambda x: x
        self.optimizer = None

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def forward(self, x_a=None, x_b=None):
        if x_a is not None:
            emb_a_ = self.encoder_a(x_a)
            if self.h_a is None:
                self.h_a = MultiLayerPerceptron(
                    num_input_features=emb_a_.size(-1),
                    hidden_layer_dimensions=[self.dim],
                ).to(emb_a_)
            emb_a = F.normalize(self.h_a(self.proj_func(emb_a_)))  # [B, F]

        if x_b is not None:
            emb_b = self.encoder_b(x_b)
            if self.h_b is None:
                self.h_b = MultiLayerPerceptron(
                    num_input_features=emb_b.size(-1),
                    hidden_layer_dimensions=[self.dim],
                ).to(emb_b)
            emb_b = F.normalize(self.h_b(self.proj_func(emb_b)))  # [B, F]
        logits = torch.matmul(emb_a, emb_b.T)  # [B, B]

        if self.supervised_head is None:
            self.supervised_head = MultiLayerPerceptron(
                num_input_features=emb_a_.size(-1),
                hidden_layer_dimensions=self.supervised_head_dim,
            )
        supervised_logits = self.supervised_head(emb_a_)
        return logits, supervised_logits

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        supervised_labels = batch["labels"]
        logits, supervised_logits = self.forward(**inputs)

        if step_name == "train":
            logits = logits * self.temperature
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        labels = torch.arange(len(logits)).to(device=logits.device).long()

        loss_a = criterion(logits, labels)
        loss_b = criterion(logits.T, labels)
        loss = (loss_a + loss_b) / 2
        total = len(logits)

        logs = {}
        mask = supervised_labels != -1
        if mask.sum() != 0:
            mask = supervised_labels != -1
            supervised_criterion = nn.BCEWithLogitsLoss(reduction="none")
            supervised_loss = supervised_criterion(supervised_logits, supervised_labels)
            supervised_loss = torch.masked_select(supervised_loss, mask).mean()
            loss += supervised_loss

            supervised_outputs = torch.sigmoid(supervised_logits)

            logs.update(_supervised_metric(supervised_labels, supervised_outputs))
            logs["supervised_loss"] = supervised_loss.cpu().detach()

        logits = logits.detach()
        labels = labels.detach()

        topk = (1, 5, 10)
        acc_a = accuracy(logits.detach(), labels.detach(), topk=topk)
        acc_b = accuracy(logits.detach().t(), labels.detach(), topk=topk)
        for k, acc_ak, acc_bk in zip(topk, acc_a, acc_b):
            suffix = f"_top{k}" if k != 1 else ""
            acc_ak = acc_ak.cpu().item()
            acc_bk = acc_bk.cpu().item()
            logs.update(
                {
                    f"acc_a{suffix}": acc_ak,
                    f"acc_b{suffix}": acc_bk,
                    f"acc{suffix}": (acc_ak + acc_bk) / 2,
                }
            )

        correct_a = logits.detach().argmax(dim=1).eq(labels).sum().cpu().item()
        correct_b = logits.detach().argmax(dim=0).eq(labels).sum().cpu().item()
        logs.update(
            {
                "loss": loss.cpu().detach(),
                "loss_a": loss_a.cpu().detach(),
                "loss_b": loss_b.cpu().detach(),
                "acc_a_old": correct_a / total,
                "acc_b_old": correct_b / total,
                "acc_old": (correct_a + correct_b) / 2 / total,
            }
        )

        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
        if step_name == "val" and "supervised_loss" in logs:
            batch_dictionary["outputs"] = supervised_outputs
            batch_dictionary["labels"] = supervised_labels
        return batch_dictionary

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )

        for k, v in batch_dictionary["log"].items():
            self.logger.experiment.add_scalar(k, v, self.global_step)

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        return self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )

    def validation_epoch_end(self, validation_step_outputs):
        _validation_epoch_end(self, validation_step_outputs)


class LightningGGNNRegression(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = GatedGraphNeuralNetwork(**kwargs)
        self.optimizer = None
        self.scheduler = None

    def configure_optimizers(self):
        optimizers = {"optimizer": self.optimizer}
        if self.scheduler is not None:
            optimizers["lr_scheduler"] = self.scheduler
        return optimizers

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        supervised_labels = batch["labels"]
        logits = self.model.forward(**inputs)
        criterion = nn.MSELoss()

        logs = {}
        supervised_loss = criterion(logits, supervised_labels)
        loss = supervised_loss

        logs["supervised_loss"] = supervised_loss.cpu().detach()

        logs.update({"loss": loss.cpu().detach()})

        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
        if step_name == "val":
            batch_dictionary["outputs"] = logits
            batch_dictionary["labels"] = supervised_labels
        return batch_dictionary

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )

        for k, v in batch_dictionary["log"].items():
            self.logger.experiment.add_scalar(k, v, self.global_step)

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        return self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def validation_epoch_end(self, validation_step_outputs):
        _validation_epoch_end(self, validation_step_outputs, is_regression=True)


class LightningGGNN(pl.LightningModule):
    def __init__(self, freeze=False, **kwargs):
        super().__init__()
        self.model = GatedGraphNeuralNetwork(**kwargs)
        self.optimizer = None
        self.scheduler = None
        if freeze:
            self.model.transfer(freeze=True)

    def configure_optimizers(self):
        optimizers = {"optimizer": self.optimizer}
        if self.scheduler is not None:
            optimizers["lr_scheduler"] = self.scheduler
        return optimizers

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_scheduler(self, scheduler, scheduler_config):
        self.scheduler = scheduler_config
        self.scheduler["scheduler"] = scheduler

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        inputs = batch["inputs"]
        supervised_labels = batch["labels"]
        logits = self.model.forward(**inputs)

        logs = {}
        mask = supervised_labels != -1
        if mask.sum() != 0:
            supervised_criterion = nn.BCEWithLogitsLoss(reduction="none")
            supervised_loss = supervised_criterion(logits, supervised_labels)
            supervised_loss = torch.masked_select(supervised_loss, mask).mean()
            loss = supervised_loss

            supervised_outputs = torch.sigmoid(logits)

            logs.update(_supervised_metric(supervised_labels, supervised_outputs))
            logs["supervised_loss"] = supervised_loss.cpu().detach()

        logs.update({"loss": loss.cpu().detach()})

        logs = {f"{step_name}/{k}": v for k, v in logs.items()}

        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
        if step_name == "val":
            batch_dictionary["outputs"] = supervised_outputs
            batch_dictionary["labels"] = supervised_labels
        return batch_dictionary

    def training_step(self, train_batch, batch_idx):
        batch_dictionary = self._step(
            batch=train_batch, batch_idx=batch_idx, step_name="train"
        )

        for k, v in batch_dictionary["log"].items():
            self.logger.experiment.add_scalar(k, v, self.global_step)

        return batch_dictionary

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        return self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def validation_epoch_end(self, validation_step_outputs):
        _validation_epoch_end(self, validation_step_outputs)


class GatedGraphNeuralNetwork(nn.Module):
    """A variant of the graph neural network family that utilizes GRUs
    to control the flow of information between layers.
    Each GatedGraphConvolution operation follows the formulations below:
    .. math:: H^{(L_i)} = A H^{(L-1)} W^{(L)}
    .. math:: H^{(L)} = GRU(H^{(L-1)}, H^{(L_i)})
    The current implementation also facilites transfer learning with the "transfer" method.
    The method replaces the last fully connected layer in the trained model object
    with a reinitialized layer that has a specified output dimension.
    Example:
    >>> # here we instantiate a model with output dimension 1
    >>> model = GatedGraphNeuralNetwork(n_edge=1, in_dim=10, n_conv=5, fc_dims=[1024, 1])
    >>> # now we reinitializes the last layer to have output dimension of 50
    >>> model.transfer(out_dim=50)
    Gated Graph Sequence Neural Networks: https://arxiv.org/abs/1511.05493
    Neural Message Passing for Quantum Chemistry: https://arxiv.org/abs/1704.01212
    """

    def __init__(
        self,
        n_edge: int,
        in_dim: int,
        n_conv: int,
        fc_dims: Iterable[int],
        p_dropout: float = 0.2,
    ) -> None:
        """Gated graph neural network with support for transfer learning
        Parameters
        ----------
        n_edge : int
            Number of edges in input graphs.
        in_dim : int
            Number of features per node in input graphs.
        n_conv : int
            Number of gated graph convolution layers.
        fc_dims : Iterable[int]
            Fully connected layers dimensions.
        """
        super(GatedGraphNeuralNetwork, self).__init__()

        self.conv_layers, self.fc_layers = self._build_layers(
            in_dim=in_dim, n_edge=n_edge, fc_dims=fc_dims, n_conv=n_conv
        )

        self.dropout = nn.Dropout(p=p_dropout)
        self.reset_parameters()

    @staticmethod
    def _build_layers(in_dim, n_edge, fc_dims, n_conv):
        conv_layers = []

        for i in range(n_conv):
            l = GatedGraphConvolution(in_dim=in_dim, out_dim=in_dim, n_edge=n_edge)
            conv_layers.append(l)

        fc_layers = []
        num_fc_layers = len(fc_dims)
        fc_dims.insert(0, in_dim)
        for i, (in_dim, out_dim) in enumerate(zip(fc_dims[:-1], fc_dims[1:])):
            l = nn.Linear(in_dim, out_dim)

            if i < (num_fc_layers - 2):
                l = nn.Sequential(l, nn.ReLU())
            elif i == (num_fc_layers - 2):
                l = nn.Sequential(l, nn.Tanh())

            fc_layers.append(l)

        return nn.ModuleList(conv_layers), nn.ModuleList(fc_layers)

    def reset_parameters(self):
        for l in self.conv_layers:
            l.reset_parameters()

        for k, v in self.state_dict().items():
            if "fc_layers" in k:
                if "weight" in k:
                    nn.init.xavier_uniform_(v)
                elif "bias" in k:
                    nn.init.zeros_(v)

    def encode(self, x: List[torch.Tensor]) -> torch.Tensor:
        """Encode featurized batched input.
        This is done by forward propagating up to the second to last layer in the network.
        Parameters
        ----------
        x : List[torch.Tensor]
            List of batch input torch.Tensor [adj_mat, node_feat, atom_vec ]
        Returns
        -------
        torch.Tensor
            Encoded inputs
        """
        adj, node_feat, atom_vec = x

        for layer in self.conv_layers:
            node_feat = layer(adj, node_feat)
            node_feat = self.dropout(node_feat)
        output = torch.mul(node_feat, atom_vec)

        output = output.sum(1)

        for layer in self.fc_layers[:-1]:
            output = layer(output)
            output = self.dropout(output)

        return output

    def forward(self, x_a: List[torch.Tensor], **kwargs) -> torch.Tensor:
        """Run forward pass on batched input.
        Parameters
        ----------
        x : List[torch.Tensor]
            List of batch input torch.Tensor [adj_mat, node_feat, atom_vec]
        Returns
        -------
        torch.Tensor
            Model output
        """
        output = self.encode(x_a)
        output = self.fc_layers[-1](output)
        return output

    def transfer(self, out_dim: Union[list, int] = None, freeze: bool = False) -> None:
        """Replace the last fully connected layer with a newly initialized layer
        with out_dim as output dimension. Use freeze=True to freeze the pre-trained
        network and use it as a featurizer.
        Parameters
        ----------
        out_dim : Union[list,int]
            Output dimension of the new fully connected layer
        freeze : bool, optional
            Freeze the weights of the pretrained network, by default False
        """
        # only transfer learn on graph level
        self.dropout = nn.Dropout(p=0.1)

        # freeze parameters if necessary
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

        # new final fc layer overriding freeze
        if out_dim is None:
            out_dim = self.fc_layers[-1].out_features

        if isinstance(out_dim, int):
            out_dim = [out_dim]

        in_dim = self.fc_layers[-1].in_features
        out_dim.insert(0, in_dim)
        del self.fc_layers[-1]

        for i, (in_dim, out_dim_) in enumerate(zip(out_dim[:-1], out_dim[1:])):
            layer = nn.Linear(in_dim, out_dim_)
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.fc_layers.append(layer)

        return None
