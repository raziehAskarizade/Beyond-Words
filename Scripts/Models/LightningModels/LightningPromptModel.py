
import torch
import torchmetrics
from Scripts.Models.LightningModels.LightningModels import BaseLightningModel
from Scripts.Models.LossFunctions.loss_helpers import HeteroLossArgs


class LightningPromptModel(BaseLightningModel):
    
    def __init__(self, model, optimizer=None, loss_func=None, learning_rate=0.01, batch_size=64, lr_scheduler=None, user_lr_scheduler=False, min_lr=0.0):
        super(LightningPromptModel, self).__init__(model, optimizer, loss_func, learning_rate, batch_size=batch_size, lr_scheduler=lr_scheduler, user_lr_scheduler=user_lr_scheduler, min_lr=min_lr)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.test_acc = torchmetrics.Accuracy(task="binary")

    def training_step(self, data_batch, *args, **kwargs):    
        data, labels, prompts, target_prompts = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        h_out_features = HeteroLossArgs(out_features[0], out_features[1])
        label_features = HeteroLossArgs(labels.view(out_features[0].shape), data.x_dict)
        loss = self.loss_func(h_out_features, label_features)
        self.log('train_loss', loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
        
        predicted_labels = out_features[0] if out_features[0].shape[1] < 2 else torch.argmax(out_features[0], dim=1)
        self.train_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('train_acc', self.train_acc, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, data_batch, *args, **kwargs):
        data, labels = data_batch
        data = data.to(self.device)
        labels = labels.to(self.device)
        out_features = self(data)
        h_out_features = HeteroLossArgs(out_features[0], out_features[1])
        label_features = HeteroLossArgs(labels.view(out_features[0].shape), data.x_dict)
        loss = self.loss_func(h_out_features, label_features)
        self.log('val_loss', loss, batch_size=self.batch_size, on_epoch=True, on_step=False)
        predicted_labels = out_features[0] if out_features[0].shape[1] < 2 else torch.argmax(out_features[0], dim=1)
        self.val_acc(predicted_labels, data_batch[1].view(predicted_labels.shape))
        self.log('val_acc', self.val_acc, prog_bar=True, on_epoch=True, on_step=True, batch_size=self.batch_size)

    def _get_loss_func(self, loss_func):
        return loss_func \
            if loss_func is not None else \
            torch.nn.BCELoss()
