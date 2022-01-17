import pdb
import copy
from signjoey.vocabulary import PAD_ID
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules import BiLSTM, BiLSTMLayer, TemporalConv
from signjoey.encoders import TransformerEncoder
from signjoey.decoders import TransformerDecoder, RecurrentDecoder

def make_src_mask(lgt):
    batch_len = len(lgt)
    mask = torch.zeros([batch_len, lgt[0]], dtype=int)
    for idx, l in enumerate(lgt):
        for i in range(l):
            mask[idx][i] = 1
    return mask.unsqueeze(1)

def make_txt_mask(lgt):
    """
        Create text mask from a sequence of length (not necessarily sorted)
        Input:
        lgt: sequence of length ([3,4,1,2...]): 1d Tensor

        Output:
        A 3 dimension Tensor of mask [B, M, M] with B is mini-batch length,

    """
    m = torch.max(lgt)
    txt_mask = []
    for _, l in enumerate(lgt):
        msk = torch.zeros((m,m), dtype=bool)
        for i in range(l):
            for j in range(i+1):
                msk[i][j] = True
        txt_mask.append(msk.unsqueeze(0))
    
    return torch.cat(txt_mask, dim=0)

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SLTVACModel(nn.Module):
    def __init__(
        self,
        num_classes,
        vocab_num_classes,
        c2d_type,
        conv_type,
        use_bn=False,
        encoder_type='Transformers',
        embedding_size='300',
        hidden_size=512,
        gloss_dict=None,
        vocab_dict=None,
        encoder_arg=None,
        decoder_arg=None,
        loss_weights=None,
        phase="train"
    ):

        """
        Create a model for SLTVAC

        :param num_class: number of glosses
        :param vocab_num_classes: number of words
        :param c2d_type: 2d convolution type
        :param conv_type: 1d convolution type
        :param use_bn: use batch normalization for encoder
        :param tm_type: encoder type
        :param decoder_type: decoder type
        :param hidden_size: dimension of hidden feature
        :param gloss_dict: gloss dictionary
        :param loss_weight: initial loss weights

        To be implemented
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """

        super(SLTVACModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.vocab_num_classes = vocab_num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        # self.recognition = utils.Decode(gloss_dict, num_classes, 'beam')
        if encoder_type == "BILSTM":
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        else:
            self.temporal_model = TransformerEncoder(**encoder_arg)

        self.embedding = nn.Sequential(
            [
                nn.Linear(embedding_size, hidden_size),
                nn.Softmax(dim=2)
            ]
        )


        self.decoder_module = TransformerDecoder(**decoder_arg, vocab_size=vocab_num_classes)

        self.classifier = nn.Linear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)   
        self.phase = phase
        self.do_recognition = self.loss_weights["recognition_loss_weight"] > 0
        self.do_translation = self.loss_weights["translation_loss_weight"] > 0

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, sentence, sentence_len):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        # Encoder

        if type(self.temporal_model) is BiLSTM:
            tm_outputs = self.temporal_model(x, lgt)
        else:
            tm_outputs = torch.transpose(self.temporal_model(torch.transpose(x, 0, 1), None, make_src_mask(lgt)), 0, 1)

        outputs = self.classifier(tm_outputs['predictions'])

        # Implement decoder
        sentence_outputs, _, _, _ = self.decoder_module(
            tm_outputs.transpose(0, 1),
            sentence,
            src_mask = make_src_mask(lgt),
            trg_mask = make_txt_mask([l - 1 for l in sentence_len])
        )

        # pred = None if self.training \
        #     else self.recognition.decode(outputs, lgt, batch_first=False, probs=False)
        # conv_pred = None if self.training \
        #     else self.recognition.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "sentence_logits": sentence_outputs,
        }

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        self.loss['translation'] = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
        return self.loss

    def criterion_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)
        return loss


    def loss_calculation(self, ret_dict, label, label_lgt, translation):
        """
        Calculate the loss for SLTVAC model.
        """
        loss = 0

        if self.loss_weights["recognition_loss_weight"] > 0:
            w = self.loss_weights["recognition_loss_weight"]
            loss += w * self.loss_weights['ConvCTC'] * self.loss['CTCLoss'](ret_dict["conv_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            loss += w * self.loss_weights['SeqCTC'] * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            loss += w * self.loss_weights['Dist'] * self.loss['distillation'](ret_dict["conv_logits"],
                                                           ret_dict["sequence_logits"].detach(),
                                                           use_blank=False)

        if self.loss_weights["translation_loss_weight"] > 0:
            loss += self.loss_weights["translation_loss_weight"] * self.loss['translation'](torch.transpose(ret_dict["sentence_logits"], 1, 2), translation[:][1:])

        return loss

    def output_inference(self, ret_dict):
        new_ret_dict = ret_dict


        pred = self.decoder.decode(ret_dict["sequence_logits"], ret_dict["feat_len"], batch_first=False, probs=False)
        conv_pred = self.decoder.decode(ret_dict["conv_logit"], ret_dict["feat_len"], batch_first=False, probs=False)

        new_ret_dict["conv_sents"] = conv_pred
        new_ret_dict["recognized_sents"] = pred

        return new_ret_dict
        

def subsequent_mask(size: int):
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0

if __name__ == "__main__":
    # x = torch.randn((3, 3, 3))
    # print(x)
    # x = ~x
    # print(x)
    # x = torch.randn((2, 30, 3))
    # lgt = torch.LongTensor([30,25])
    # print(make_mask(lgt).size())
    # model = SLRModel(1024,"resnet18",2,True)
    # preds = model(x,[130,125])
    # m = torch.zeros((8, 8), dtype=bool)
    # n = torch.ones((8, 8), dtype=bool)
    # p = torch.cat([m.unsqueeze(0),n.unsqueeze(0)])
    # print(p.size())
    # print(p)
    # for i in range(5):
    #     for j in range(i+1):
    #         m[i][j] = True
    # print(m.size())
    # a = [1,2,3,4]
    # a = a[:-1]
    # print(a)
    # lgt = torch.LongTensor([3,5,4,7,2])
    # txt_mask = make_txt_mask(lgt)
    # print(txt_mask.size())

    # msk = subsequent_mask(6)
    # print(msk)

    # model = TransformerDecoder()
    # x = torch.randn((2, 30, 512))
    # out1, out2 = model(x)
    # print(out1.size)
