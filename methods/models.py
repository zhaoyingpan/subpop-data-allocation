import torch
import torchvision
from torch import nn
from transformers import ViTModel, BertModel, RobertaModel, DebertaV2Model
from torchvision.models import ResNet18_Weights, ResNet50_Weights

def get_model(config):
    if config.data.dataset in ["CivilComments", "CivilComments_subset"]:
        if config.train.model_type in ["bert", "roberta", "deberta"]:
            return TextClassifier(dataset=config.data.dataset, model_type=config.train.model_type)
        else:
            raise ValueError(f"Unsupported backbone: {config.train.model_type}")
    else:
        if config.train.model_type in ["resnet18", "resnet50"]:
            return Classifier_resnet(model_type=config.train.model_type)
        elif config.train.model_type == "vit":
            return VisionTransformer()
        else:
            raise ValueError(f"Unsupported backbone: {config.train.model_type}")

class Classifier_resnet(nn.Module):
    def __init__(self, model_type='resnet18'):
        super(Classifier_resnet, self).__init__()

        if model_type == 'resnet18':
            self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif model_type == 'resnet50':
            self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        feat_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        self.fc = nn.Linear(feat_dim, 2)

    def forward(self, x):
        f = self.resnet(x)
        flat_f = torch.flatten(f, 1)
        out = self.fc(flat_f)
        return flat_f, out


class VisionTransformer(nn.Module):
    def __init__(self):
        super(VisionTransformer, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.fc = nn.Linear(768, 2)
    
    def forward(self, x):
        z = self.vit(x)
        m = z.last_hidden_state
        g = m[:, 0]
        y = self.fc(g)
        return g, y


class FeatureWrapper(torch.nn.Module):

    def __init__(self, model_type):
        super().__init__()
        self.model_type = model_type
        if self.model_type == "bert":
            self.model = BertModel.from_pretrained("bert-base-uncased")
        elif self.model_type == "roberta":
            self.model = RobertaModel.from_pretrained("roberta-base")
        elif self.model_type == "deberta":
            self.model = DebertaV2Model.from_pretrained("microsoft/deberta-v3-base")
        else:
            raise ValueError(f"Unsupported backbone: {self.model_type}")

        self.n_outputs = self.model.config.hidden_size
        classifier_dropout = getattr(self.model.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        kwargs = {
            'input_ids': x[:, :, 0],
            'attention_mask': x[:, :, 1]
        }
        if self.model_type == "bert" and x.shape[-1] == 3:
            kwargs['token_type_ids'] = x[:, :, 2]
        output = self.model(**kwargs)
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            return self.dropout(output.pooler_output)
        else:
            return self.dropout(output.last_hidden_state[:, 0, :])


class TextClassifier(nn.Module):
    def __init__(self, dataset, model_type):
        super(TextClassifier, self).__init__()
        
        self.model = FeatureWrapper(model_type)

        if dataset in ["CivilComments", "CivilComments_subset"]:
            self.fc = nn.Linear(self.model.n_outputs, 2)
    
    def forward(self, x):
        z = self.model(x)
        y = self.fc(z)
        return z, y

