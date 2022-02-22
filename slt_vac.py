import pdb
import copy
from signjoey.vocabulary import BOS_ID, EOS_ID, PAD_ID
import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.criterions import SeqKD
from modules.embeddings import WordEmbedding
from modules import BiLSTM, BiLSTMLayer, TemporalConv
from signjoey.encoders import TransformerEncoder
from signjoey.decoders import TransformerDecoder, RecurrentDecoder
from utils.masks import make_src_mask, make_txt_mask
from signjoey.search import beam_search, transformer_greedy
from utils.vocab import arrays_to_sentences
from utils.decode import Decode

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
        encoder_type='BiLSTM',
        embedding_size=300,
        hidden_size=512,
        gloss_dict=None,
        vocab_dict=None,
        vocab_dict_reverse=None,
        encoder_arg=None,
        decoder_arg=None,
        loss_weights=None,
        phase="train",
        device=None,
        pretrained_embedding=None
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
        self.device = device
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
        self.gloss_dict = gloss_dict
        self.vocab_dict = vocab_dict
        self.vocab_dict_reverse = vocab_dict_reverse
        # self.recognition = utils.Decode(gloss_dict, num_classes, 'beam')
        if encoder_type == "BiLSTM":
            self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                              num_layers=2, bidirectional=True)
        else:
            self.temporal_model = TransformerEncoder(**encoder_arg)
        

        self.embedding = WordEmbedding(embedding_size, hidden_size, pretrained_embedding, device)


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
        feature_maps = self.get_feature_maps(x)            
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x, feature_maps

    def encode(self, x, len_x):
        """Encode phase output"""
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise, feature_maps = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        # Encoder
        src_mask = self.device.data_to_device(make_src_mask(lgt))

        if type(self.temporal_model) is BiLSTMLayer:
            tm_outputs = self.temporal_model(x, lgt)['predictions']
        else:            
            tm_outputs = torch.transpose(self.temporal_model(torch.transpose(x, 0, 1), None, src_mask)[0], 0, 1)
        
        outputs = self.classifier(tm_outputs)

        return {
            "framewise_features": framewise,
            "feature_maps": feature_maps,
            "visual_features": x,
            "feat_len": lgt,
            "conv_logits": conv1d_outputs['conv_logits'],
            "sequence_logits": outputs,
            "src_mask": src_mask,
            "encoder_output":tm_outputs.transpose(0, 1)
        }

    def decode(self, encoder_output, src_mask, beam_size, beam_alpha):
        if beam_size > 1:
            sentence = beam_search(
                decoder=self.decoder_module,
                size=beam_size,
                bos_index=BOS_ID,
                eos_index=EOS_ID,
                pad_index=PAD_ID,
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                max_output_length=512,
                alpha=beam_alpha,
                embed=self.embedding
            )
        else:
            sentence = transformer_greedy(
                decoder=self.decoder_module,
                bos_index=BOS_ID,
                eos_index=EOS_ID,
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                max_output_length=512,      
                embed=self.embedding                         
            )
        
        return sentence
        
    def forward(self, x, len_x, sentence, sentence_len):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise, feature_maps = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']

        # Encoder
        src_mask = self.device.data_to_device(make_src_mask(lgt))

        if type(self.temporal_model) is BiLSTMLayer:
            tm_outputs = self.temporal_model(x, lgt)['predictions']
        else:
            
            tm_outputs = torch.transpose(self.temporal_model(torch.transpose(x, 0, 1), None, src_mask)[0], 0, 1)

        outputs = self.classifier(tm_outputs)

        # Implement decoder
        sentence_embedding = self.embedding(sentence)
        txt_mask = self.device.data_to_device(make_txt_mask(torch.Tensor([l for l in sentence_len])))
        sentence_outputs, _, _, _ = self.decoder_module(
            sentence_embedding,
            tm_outputs.transpose(0, 1),
            src_mask = src_mask,
            trg_mask = txt_mask
        )

        # pred = None if self.training \
        #     else self.recognition.decode(outputs, lgt, batch_first=False, probs=False)
        # conv_pred = None if self.training \
        #     else self.recognition.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "feature_maps": feature_maps,
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

    def loss_calculation(self, ret_dict, label, label_lgt, translation):
        """
        Calculate the loss for SLTVAC model.
        """
        loss = 0
        batch_size = translation.size(0)
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

        pad = torch.full([batch_size , 1], PAD_ID, dtype=torch.int)
        hyp = torch.cat([translation[:, 1:] , pad.to(self.device.output_device)], dim=1).long()
        if self.loss_weights["translation_loss_weight"] > 0:
            loss += self.loss_weights["translation_loss_weight"] * self.loss['translation'](torch.transpose(ret_dict["sentence_logits"], 1, 2), hyp)

        return loss

    def output_inference(
        self,
        do_recognition=True,
        recognition_beam_width=1,
        do_translation=True,
        translation_beam_width=1,
        translation_beam_alpha=1,
        encoder_output=None,
        conv_logits=None,
        encoder_lgt=None,
        src_mask=None,
    ):

        if do_recognition:
            output = self.classifier(encoder_output).transpose(0, 1)
            if recognition_beam_width > 1:
                recognition = Decode(self.gloss_dict, self.num_classes, 'beam', beam_width=recognition_beam_width)
            else:
                recognition = Decode(self.gloss_dict, self.num_classes, 'max')
            temporal_gloss = recognition.decode(output, encoder_lgt, batch_first=False, probs=False)
            conv_gloss = recognition.decode(conv_logits, encoder_lgt, batch_first=False, probs=False)
        else:
            temporal_gloss, conv_gloss = None, None
        if do_translation:
            translation, _ = self.decode(encoder_output, src_mask, translation_beam_width, translation_beam_alpha)
            tokenized_translation = arrays_to_sentences(vocab_dict=self.vocab_dict_reverse, arrays=translation)
        else:
            tokenized_translation = None
        
        ret_dict = dict()
        ret_dict["recognized_sents"] = temporal_gloss
        ret_dict['conv_sents'] = conv_gloss
        ret_dict['translations'] = tokenized_translation
        return ret_dict
        

    def get_feature_maps(self, x):
        feature_map_model = torch.nn.Sequential(*list(self.conv2d.children())[:-2])
        feature_maps =feature_map_model(x)
        return feature_maps

def subsequent_mask(size: int):
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0

# if __name__ == "__main__":
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
