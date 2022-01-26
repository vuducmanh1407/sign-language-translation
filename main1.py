import os

from zmq import device

from preprocess.dataset_preprocess import vocab_dict_update

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import pdb
import sys
import cv2
import yaml
import torch
import random
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict

faulthandler.enable()
import utils
from seq_scripts import seq_train, seq_eval, seq_test, seq_feature_generation


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)
        self.device = utils.GpuDataParallel()
        self.recorder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)
        self.dataset = {}
        self.data_loader = {}
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.vocab_dict = np.load(self.arg.dataset_info['vocab_path'], allow_pickle=True).item()
        self.word_embedding = np.load(self.arg.dataset_info['embedding_path'], allow_pickle=True).item()
        self.vocab_dict_reverse = {value:key for key, value in self.vocab_dict.items()}
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.arg.model_args['vocab_num_classes'] = len(self.vocab_dict)
        self.model, self.optimizer = self.loading()
        self.do_recognition = arg.loss_weights["recognition_loss_weight"] > 0.0
        self.do_translation = arg.loss_weights["translation_loss_weight"] > 0.0

    def start(self):
        if self.arg.phase == 'train':
            self.recorder.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            seq_model_list = []
            loss_dict = {}
            metrics_dict = {}
            for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
                save_model = epoch % self.arg.save_interval == 0
                eval_model = epoch % self.arg.eval_interval == 0
                # train end2end model
                loss = seq_train(self.data_loader['train'], self.model, self.optimizer,
                          self.device, epoch, self.recorder)
                loss_dict[epoch] = loss
                if eval_model:
                    metrics = seq_eval(cfg=self.arg, loader=self.data_loader['dev'], model=self.model, device=self.device, epoch=epoch,
                                    recorder=self.recorder, do_recognition=self.do_recognition, do_translation=self.do_translation)
                    metrics_dict[epoch] = metrics
                if save_model:
                    model_path = "{}dev_epoch{}_model.pt".format(self.arg.work_dir, epoch)
                    seq_model_list.append(model_path)
                    print("seq_model_list", seq_model_list)
                    self.save_model(epoch, model_path)
            
            # save figures for visualization
            # TO DO
            np.save(f"./{self.arg.work_dir}loss_dict.npy", loss_dict)
            np.save(f"./{self.arg.work_dir}metrics_dict.npy", metrics_dict)
        elif self.arg.phase == 'test':
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError('Please appoint --load-weights.')
            self.recorder.print_log('Model:   {}.'.format(self.arg.model))
            self.recorder.print_log('Weights: {}.'.format(self.arg.load_weights))
            # train_wer = seq_eval(self.arg, self.data_loader["train_eval"], self.model, self.device,
            #                      "train", 6667, self.arg.work_dir, self.recorder, self.arg.evaluate_tool)
            # dev_wer = seq_eval(self.arg, self.data_loader["dev"], self.model, self.device,
            #                    "dev", 6667, self.arg.work_dir, self.recorder, self.arg.evaluate_tool)
            # test_wer = seq_eval(self.arg, self.data_loader["test"], self.model, self.device,
            #                     "test", 6667, self.arg.work_dir, self.recorder, self.arg.evaluate_tool)
            seq_test(cfg=self.arg, dev_loader=self.data_loader['dev'], test_loader=self.data_loader['test'], model=self.model,
            device=self.device, output_path=self.arg.work_dir, recorder=self.recorder, do_recognition=self.do_recognition,
            do_translation=self.do_translation)
            self.recorder.print_log('Evaluation Done.\n')
        elif self.arg.phase == "features":
            for mode in ["train", "dev", "test"]:
                seq_feature_generation(
                    self.data_loader[mode + "_eval" if mode == "train" else mode],
                    self.model, self.device, mode, self.arg.work_dir, self.recorder
                )
        elif self.arg.phase == "inference": # inference phase 
            if self.arg.load_weights is None and self.arg.load_checkpoints is None:
                raise ValueError('Please appoint --load-weights.')
            # TODO: Build dataloader for 1 video

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def save_model(self, epoch, save_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state(),
        }, save_path)

    def loading(self):
        self.device.set_device(self.arg.device)
        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            vocab_dict=self.vocab_dict,
            encoder_arg=self.arg.encoder,
            decoder_arg=self.arg.decoder,
            loss_weights=self.arg.loss_weights,
            device=self.device,
            pretrained_embedding=self.word_embedding
        )
        optimizer = utils.Optimizer(model, self.arg.optimizer_args)

        if self.arg.load_weights:
            self.load_model_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self.load_checkpoint_weights(model, optimizer)
        model = self.model_to_device(model)
        print("Loading model finished.")
        if self.arg.phase in ['test','train','features']: # only
            self.load_data()
        return model, optimizer

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        # weights = self.modified_weights(state_dict['model_state_dict'])
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_checkpoint_weights(self, model, optimizer):
        self.load_model_weights(model, self.arg.load_checkpoints)
        state_dict = torch.load(self.arg.load_checkpoints)

        if len(torch.cuda.get_rng_state_all()) == len(state_dict['rng_state']['cuda']):
            print("Loading random seeds...")
            self.rng.set_rng_state(state_dict['rng_state'])
        if "optimizer_state_dict" in state_dict.keys():
            print("Loading optimizer parameters...")
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            optimizer.to(self.device.output_device)
        if "scheduler_state_dict" in state_dict.keys():
            print("Loading scheduler parameters...")
            optimizer.scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        self.arg.optimizer_args['start_epoch'] = state_dict["epoch"] + 1
        self.recorder.print_log("Resuming from checkpoint: epoch {self.arg.optimizer_args['start_epoch']}")

    def load_data(self):
        print("Loading data")
        self.feeder = import_class(self.arg.feeder)
        dataset_list = zip(["train", "train_eval", "dev", "test"], [True, False, False, False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["prefix"] = self.arg.dataset_info['dataset_root']
            arg["mode"] = mode.split("_")[0]
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, vocab_dict=self.vocab_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    mod = importlib.import_module(components[0])
    mod = getattr(mod, components[1])
    return mod


if __name__ == '__main__':
    sparser = utils.get_parser()
    p = sparser.parse_args()
    # p.config = "baseline_iter.yaml"
    if p.config is not None:
        with open(p.config, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        # for k in default_arg.keys():
        #     if k not in key:
        #         print('WRONG ARG: {}'.format(k))
        #         assert (k in key)
        sparser.set_defaults(**default_arg)
    args = sparser.parse_args()
    with open(f"./configs/{args.dataset}.yaml", 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)
    print(args)
    processor = Processor(args)
    # utils.pack_code("./", args.work_dir)
    processor.start()
