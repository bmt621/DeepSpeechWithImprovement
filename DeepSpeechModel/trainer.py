from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils import data

from utils import data_processor


class Trainer:
    """allows for training deepspeech model"""

    def __init__(self,path_to_save_model):
        self.path_to_save = path_to_save_model

        pass


    def set_trainer(self):
        pass


    def train(self):
        pass






learning_rate = 5e-4
batch_size = 20
epochs = 50
hparams = {
        "n_cnn_layers": 3,
        "n_rnn_layers": 5,
        "rnn_dim": 512,
        "n_class": 29,
        "n_feats": 128,
        "stride":2,
        "dropout": 0.1,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}


train_dataset = None
test_dataset = None

train_loader = data.DataLoader(dataset=train_dataset,
                                batch_size=hparams['batch_size'],
                                shuffle=True,
                                collate_fn=lambda x: data_processor(x, True),
                                **kwargs)


test_loader = data.DataLoader(dataset = test_dataset,
                              batch_size = hparams['batch_size'],
                              shuffle=False,
                              collate_fn=lambda x: data_processor(x, False),
                              **kwargs)


