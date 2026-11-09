import torch
from transformers import get_scheduler, get_cosine_schedule_with_warmup

from methods.models import get_model
from methods.optimizers import get_optimizers


class Algorithm(torch.nn.Module):
    def __init__(self, config, num_classes, num_attributes, group_sizes, num_steps):
        super().__init__()
        self.config = config
        self.model_type = config.train.model_type
        self.data_type = (
            "text"
            if self.model_type in ["bert", "roberta", "deberta"]
            else "images"
        )
        self.dataset = config.data.dataset
        self.num_classes = num_classes
        self.num_attributes = num_attributes
        self.group_sizes = group_sizes
        self.num_steps = (
            num_steps if num_steps is not None else config.train.num_steps
        )

    def _init_model(self):
        raise NotImplementedError

    def _compute_loss(self, i, x, y, a, step):
        raise NotImplementedError

    def update(self, minibatch, step):
        raise NotImplementedError

    def return_feats(self, x):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """Empirical risk minimization."""

    def __init__(self, config, num_classes, num_attributes, group_sizes, num_steps):
        super().__init__(
            config, num_classes, num_attributes, group_sizes, num_steps
        )
        self.network = get_model(config)
        self._init_model()

    def _init_model(self):
        config = self.config
        self.clip_grad = self.data_type == "text"
        if self.model_type in ["resnet18", "resnet50"]:
            self.optimizer = get_optimizers["sgd"](
                self.network,
                lr=config.train.lr,
                weight_decay=config.train.weight_decay,
            )
            self.lr_scheduler = None
        elif self.model_type == "vit":
            self.optimizer = get_optimizers["adamw"](
                self.network,
                lr=1e-5,
                weight_decay=1e-3,
            )
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(0.1 * self.num_steps),
                num_training_steps=self.num_steps,
            )
        elif self.model_type in ["bert", "roberta", "deberta"]:
            self.network.zero_grad()
            self.optimizer = get_optimizers["bert_adamw"](
                self.network,
                lr=3e-5,
                weight_decay=0.01,
            )
            self.lr_scheduler = get_scheduler(
                "linear",
                optimizer=self.optimizer,
                num_warmup_steps=0,
                num_training_steps=self.num_steps,
            )
        else:
            raise ValueError(f"Unsupported model type {self.model_type}")
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    def _compute_loss(self, i, x, y, a, step):
        return self.loss(self.predict(x), y).mean()

    def update(self, minibatch, step):
        all_i, all_x, all_y, all_a = minibatch
        loss = self._compute_loss(all_i, all_x, all_y, all_a, step)

        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.data_type == "text":
            self.network.zero_grad()

        return {"loss": loss.item()}

    def return_feats(self, x):
        return self.network(x)[0]

    def predict(self, x):
        return self.network(x)[1]
