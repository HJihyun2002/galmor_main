import torch
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer
from datasets import get_dataset
from utils.batch_norm import bn_track_stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.colors as mcolors
import warnings
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
# warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.manifold._t_sne")


class GALMOR(ContinualModel):
    NAME = 'GALMOR'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser() -> ArgumentParser:
        parser = ArgumentParser(description='A continual learning method: GALMOR')
        add_rehearsal_args(parser)
        return parser

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size)
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.dataset = get_dataset(args)
        self.class_means = None
        self.class_representative_samples = {}
        self.all_feature_means = {}
        self.test_loaders = []
        # self.alpha_list = []
        # self.beta_list = []

    def forward(self, x):
        """
        Implementing the forward propagation process.

        Parameters:
        x (torch.Tensor): Input the feature data of the sample.

        Return:
        torch.Tensor: Used for subsequent loss calculations or category predictions.
        """
        return 0

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        present = labels.unique()

        real_batch_size = inputs.shape[0]
        self.class_means = None

        self.opt.zero_grad()

        logits = self.net(inputs)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        # if self.seen_so_far.max() < (self.num_classes - 1):
        #     mask[:, self.seen_so_far.max():] = 1
        # If the current task index is greater than 0,
        # fill the portions of the logits with a mask of 0 with the smallest value representable by the current data type to ignore the influence of these categories.
        if self.current_task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        # loss_re = torch.tensor(0.)

        inputs1 = inputs
        labels1 = labels

        if not self.buffer.is_empty():
            # sample from buffer
            buf_inputs, buf_labels, buf_indexes = self.buffer.get_data(
                self.args.minibatch_size, transform=self.transform, device=self.device, return_index=True)
            inputs1 = torch.cat((inputs, buf_inputs))
            labels1 = torch.cat((labels, buf_labels))
            if self.current_task > 0:
                loss_re = self.loss(self.net(buf_inputs), buf_labels)

        if self.current_task > 0:
            # Compute gradients
            loss.backward(retain_graph=True)
            grad_current = torch.cat([p.grad.view(-1) for p in self.net.parameters()])
            self.opt.zero_grad()

            loss_re.backward(retain_graph=True)
            grad_replay = torch.cat([p.grad.view(-1) for p in self.net.parameters()])
            self.opt.zero_grad()

            grad_norm_current = torch.norm(grad_current)
            grad_norm_replay = torch.norm(grad_replay)
            alpha = grad_norm_current / (grad_norm_current + grad_norm_replay)
            beta = grad_norm_replay / (grad_norm_current + grad_norm_replay)
            # self.alpha_list.append(alpha.item())
            # self.beta_list.append(beta.item())
            loss = alpha * loss + beta * loss_re
            # print("alpha:", alpha, "beta:", beta)

        '''
        Calculate the sample importance score.
        
        Get: feature_scores.
        '''

        loss.backward()
        self.opt.step()

        if not self.buffer.is_empty():
            self.buffer.update_scores(buf_indexes, feature_scores.detach()[real_batch_size:])
        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels[:real_batch_size],
                             feature_scores=feature_scores.detach()[:real_batch_size])

        return loss.item()

    def end_task(self, dataset) -> None:
        self.class_means = None
        self.compute_intra_class_coverage()
        self.visualize_feature_clusters(task_id=self.current_task)

    @torch.no_grad()
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.

        This function calculates the mean feature vector for each class that has been seen so far,
        using data from the buffer and current inputs. The computed means are stored in `self.class_means`
        for later use in classification or other tasks.

        Returns:
            None: This function does not return anything. It stores the result in `self.class_means`.
        """

    @torch.no_grad()
    def visualize_feature_clusters(self, task_id, save_path=None):
        """
        Visualize the feature clustering of the current model across all task class by applying t-SNE dimensionality reduction
        and displaying the results on a two-dimensional plane.
        Includes samples from the buffer and the current task's test set, represented in different styles respectively.

        Parameters:
            task_id (int): Current task number.
            save_path (str, optional): Image save path.

        Returns:
            None: The result will be saved as an image file.
        """

    @torch.no_grad()
    def compute_intra_class_coverage(self):
        """
        Calculate the feature Coverage for each category for use in ablation experiments.

        Parameters:
            No explicit parameters.

        Returns:
            None: The results will be printed directly.
        """
