# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import Namespace
from contextlib import suppress
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset
from datasets.utils.continual_dataset import ContinualDataset

from utils.conf import get_device
from utils.kornia_utils import to_kornia_transform
from utils.magic import persistent_locals

with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]
    AVAIL_OPTIMS = ['sgd', 'adam', 'adamw']

    @property
    def current_task(self):
        """
        Returns the index of current task.
        """
        return self._current_task

    @property
    def n_classes_current_task(self):
        """
        Returns the number of classes in the current task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_classes_current_task'):
            return self._n_classes_current_task
        else:
            return -1

    @property
    def n_seen_classes(self):
        """
        Returns the number of classes seen so far.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_seen_classes'):
            return self._n_seen_classes
        else:
            return -1

    @property
    def n_remaining_classes(self):
        """
        Returns the number of classes remaining to be seen.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_remaining_classes'):
            return self._n_remaining_classes
        else:
            return -1

    @property
    def n_past_classes(self):
        """
        Returns the number of classes seen up to the PAST task.
        Returns -1 if task has not been initialized yet.
        """
        if hasattr(self, '_n_past_classes'):
            return self._n_past_classes
        else:
            return -1

    @property
    def cpt(self):
        """
        Returns the raw number of classes per task.
        Warning: return value might be either an integer or a list of integers.
        """
        return self._cpt

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.dataset = get_dataset(self.args)
        self.N_CLASSES = self.dataset.N_CLASSES
        self.N_TASKS = self.dataset.N_TASKS
        self.SETTING = self.dataset.SETTING
        self._cpt = self.dataset.N_CLASSES_PER_TASK
        self._current_task = 0

        try:
            self.weak_transform = to_kornia_transform(transform.transforms[-1].transforms)
            self.normalization_transform = to_kornia_transform(self.dataset.get_normalization_transform())
        except BaseException:
            print("Warning: could not initialize kornia transforms.")
            self.weak_transform = None
            self.normalization_transform = None

        if self.net is not None:
            self.opt = self.get_optimizer()
        else:
            print("Warning: no default model for this dataset. You will have to specify the optimizer yourself.")
            self.opt = None
        self.device = get_device()

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')

        if self.args.label_perc != 1 and 'cssl' not in self.COMPATIBILITY:
            print('WARNING: label_perc is not explicitly supported by this model -> training may break')

    def load_buffer(self, buffer):
        """
        Default way to handle load buffer.
        """
        assert buffer.examples.shape[0] == self.args.buffer_size, "Buffer size mismatch. Expected {} got {}".format(
            self.args.buffer_size, buffer.examples.shape[0])
        self.buffer = buffer

    def get_optimizer(self):
        # check if optimizer is in torch.optim
        supported_optims = {optim_name.lower(): optim_name for optim_name in dir(optim) if optim_name.lower() in self.AVAIL_OPTIMS}
        opt = None
        if self.args.optimizer.lower() in supported_optims:
            if self.args.optimizer.lower() == 'sgd':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.net.parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd,
                                                                                    momentum=self.args.optim_mom,
                                                                                    nesterov=self.args.optim_nesterov == 1)
            elif self.args.optimizer.lower() == 'adam' or self.args.optimizer.lower() == 'adamw':
                opt = getattr(optim, supported_optims[self.args.optimizer.lower()])(self.net.parameters(), lr=self.args.lr,
                                                                                    weight_decay=self.args.optim_wd)

        if opt is None:
            raise ValueError('Unknown optimizer: {}'.format(self.args.optimizer))
        return opt

    def _compute_offsets(self, task):
        cpt = self.N_CLASSES // self.N_TASKS
        offset1 = task * cpt
        offset2 = (task + 1) * cpt
        return offset1, offset2

    def get_debug_iters(self):
        """
        Returns the number of iterations to be used for debugging.
        Default: 3
        """
        return 5

    def begin_task(self, dataset: ContinualDataset) -> None:
        """
        Prepares the model for the current task.
        Executed before each task.
        """
        pass

    def end_task(self, dataset: ContinualDataset) -> None:
        """
        Prepares the model for the next task.
        Executed after each task.
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if 'cssl' not in self.COMPATIBILITY:  # drop unlabeled data if not supported
            labeled_mask = args[1] != -1
            if labeled_mask.sum() == 0:
                return 0
            args = [arg[labeled_mask] if isinstance(arg, torch.Tensor) and arg.shape[0] == args[0].shape[0] else arg for arg in args]
        if 'wandb' in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def meta_begin_task(self, dataset):
        self._n_classes_current_task = self._cpt if isinstance(self._cpt, int) else self._cpt[self._current_task]
        self._n_seen_classes = self._cpt * (self._current_task + 1) if isinstance(self._cpt, int) else sum(self._cpt[:self._current_task + 1])
        self._n_remaining_classes = self.N_CLASSES - self._n_seen_classes
        self._n_past_classes = self._cpt * self._current_task if isinstance(self._cpt, int) else sum(self._cpt[:self._current_task])
        self.begin_task(dataset)

    def meta_end_task(self, dataset):
        self._current_task += 1
        self.end_task(dataset)

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor, epoch: int = None) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals, extra=None):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            tmp = {k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                   for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')}
            tmp.update(extra or {})
            wandb.log(tmp)
