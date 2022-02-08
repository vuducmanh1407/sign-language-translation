import os
import pdb
import sys
import copy
import torch
import numpy as np
import pickle as pickle
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from evaluation.slr_eval.wer_calculation import evaluate
from utils.vocab import arrays_to_sentences
from utils.metrics import *


def seq_train(loader, model, optimizer, device, epoch_idx, recorder):
    model.train()
    loss_value = []
    clr = [group['lr'] for group in optimizer.optimizer.param_groups]
    for batch_idx, data in enumerate(loader):
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        label = device.data_to_device(data[2])
        label_lgt = device.data_to_device(data[3])
        translation = device.data_to_device(data[4])
        translation_lgt = device.data_to_device(data[5])
        ret_dict = model(vid, vid_lgt, translation, translation_lgt)
        loss = model.loss_calculation(ret_dict, label, label_lgt, translation)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            print(data[-1])
            continue
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.rnn.parameters(), 5)
        optimizer.step()
        loss_value.append(loss.item())
        if batch_idx % recorder.log_interval == 0:
            recorder.print_log(
                '\tEpoch: {}, Batch({}/{}) done. Loss: {:.8f}  lr:{:.6f}'
                    .format(epoch_idx, batch_idx, len(loader), loss.item(), clr[0]))
    optimizer.scheduler.step()
    recorder.print_log('\tMean training loss: {:.10f}.'.format(np.mean(loss_value)))
    return loss_value


def seq_eval(cfg, loader, model, device, epoch, recorder, do_recognition=True, do_translation=True):
    model.eval()
    total_info = []
    encoder_ret_dicts = []
    for batch_idx, data in enumerate(tqdm(loader)):
        recorder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])


        with torch.no_grad():
            ret_dict = model.encode(vid, vid_lgt)
            encoder_ret_dicts.append(ret_dict)

        total_info += [file_name.split("|") for file_name in data[-1]] 

    total_gloss = []
    conv_gloss = []
    total_translation = []

    for idx, ret_dict in enumerate(encoder_ret_dicts):
        output_ret_dict = model.output_inference(
            do_recognition=True,
            recognition_beam_width=1,
            do_translation=True,
            translation_beam_width=1,
            translation_beam_alpha=1,
            encoder_output=ret_dict["encoder_output"],
            conv_logits=ret_dict["conv_logits"],
            encoder_lgt=ret_dict["feat_len"],
            src_mask=ret_dict["src_mask"],
        )

        total_gloss.extend(output_ret_dict["recognized_sents"])
        conv_gloss.extend(output_ret_dict["conv_sents"])
        total_translation.extend(output_ret_dict['translations'])

    metrics = metrics_calculation(total_sent=total_gloss, total_conv_sent=conv_gloss, total_translation=total_translation, total_info=total_info)

    recorder.print_log(
        "Epoch: {}\n"
        "WER {:3.2f}\t(DEL: {:3.2f},\tINS: {:3.2f},\tSUB: {:3.2f})\n\t"
        "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\n\t"
        "CHRF {:.2f}\t"
        "ROUGE {:.2f}".format(
            epoch,
            metrics["wer"] if do_recognition else -1,
            metrics["wer_scores"]["del_rate"]
            if do_recognition
            else -1,
            metrics["wer_scores"]["ins_rate"]
            if do_recognition
            else -1,
            metrics["wer_scores"]["sub_rate"]
            if do_recognition
            else -1,
            metrics["bleu"] if do_translation else -1,
            metrics["bleu_scores"]["bleu1"]
            if do_translation
            else -1,
            metrics["bleu_scores"]["bleu2"]
            if do_translation
            else -1,
            metrics["bleu_scores"]["bleu3"]
            if do_translation
            else -1,
            metrics["bleu_scores"]["bleu4"]
            if do_translation
            else -1,
            metrics["chrf"]
            if do_translation
            else -1,
            metrics["rouge"]
            if do_translation
            else -1
        )
    )

    return metrics

def seq_test(cfg, dev_loader, test_loader, model, device, output_path, recorder, do_recognition=True, do_translation=True):

    recognition_beam_sizes = cfg.testing.get("recognition_beam_sizes", [1])
    translation_beam_sizes = cfg.testing.get("translation_beam_sizes", [1])
    translation_beam_alphas = cfg.testing.get("translation_beam_alphas", [-1])

    model.eval()

    recorder.print_log(f"Test phase:")
    
    # Load dev set
    dev_encoder_ret_dicts = []
    dev_total_info = []
    for batch_idx, data in enumerate(tqdm(dev_loader)):
        recorder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])


        with torch.no_grad():
            ret_dict = model.encode(vid, vid_lgt)
            dev_encoder_ret_dicts.append(ret_dict)

        dev_total_info += [file_name.split("|") for file_name in data[-1]]      

    # Load test set
    test_total_info = []
    test_encoder_ret_dicts = []
    for batch_idx, data in enumerate(tqdm(test_loader)):
        recorder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])


        with torch.no_grad():
            ret_dict = model.encode(vid, vid_lgt)
            test_encoder_ret_dicts.append(ret_dict)

        test_total_info += [file_name.split("|") for file_name in data[-1]]      

    # Evaluation
    dev_best_total_gloss = []
    dev_best_conv_gloss = []
    dev_best_total_translation = []


    dev_recognition_results = dict()
    dev_best_wer_score = float("inf")
    dev_best_recognition_beam_size = 1
    dev_best_recognition_result = None
    if do_recognition:
        for rbw in recognition_beam_sizes:
            recorder.print_log("[DEV] Partition [RECOGNITION] experiment [BW]: {:d}".format(rbw))
            total_sent = []
            total_conv_sent = []
            for idx, ret_dict in enumerate(dev_encoder_ret_dicts):
                output_ret_dict = model.output_inference(
                    do_recognition=True,
                    recognition_beam_width=rbw,
                    do_translation=False,
                    translation_beam_width=1,
                    translation_beam_alpha=1,
                    encoder_output=ret_dict["encoder_output"],
                    encoder_lgt=ret_dict["feat_len"],
                    conv_logits=ret_dict["conv_logits"],
                    src_mask=ret_dict["src_mask"],
                )

                total_sent.extend(output_ret_dict["recognized_sents"])
                total_conv_sent.extend(output_ret_dict["conv_sents"])

            metrics_rbw = metrics_calculation(total_sent=total_sent, total_conv_sent=total_conv_sent, total_translation=None, total_info=dev_total_info)
            dev_recognition_results[rbw] = metrics_rbw

            if dev_recognition_results[rbw]["wer"] < dev_best_wer_score:
                dev_best_wer_score = dev_recognition_results[rbw]["wer"]
                dev_best_recognition_beam_size = rbw
                dev_best_recognition_result = dev_recognition_results[rbw]
                dev_best_total_gloss = total_sent
                dev_best_conv_gloss = total_conv_sent              

                recorder.print_log(
                    "[DEV] partition [RECOGNITION] results:\n\t"
                    "New Best CTC Decode Beam Size: {:d}\n\t"
                    "WER {:3.2f}\t(DEL: {:3.2f},\tINS: {:3.2f},\tSUB: {:3.2f})".format(
                        dev_best_recognition_beam_size,
                        dev_best_recognition_beam_size,
                        dev_best_recognition_result["wer_scores"][
                            "del_rate"
                        ],
                        dev_best_recognition_result["wer_scores"][
                            "ins_rate"
                        ],
                        dev_best_recognition_result["wer_scores"][
                            "sub_rate"
                        ]
                    )
                )
       
    dev_translation_results = {}
    dev_best_bleu_score = float("-inf")
    dev_best_translation_beam_size = 1
    dev_best_translation_alpha = 1
    if do_translation:
        for tbw in translation_beam_sizes:
            dev_translation_results[tbw] = {}
            for ta in translation_beam_alphas:
                total_translation = []
                for idx, ret_dict in enumerate(dev_encoder_ret_dicts):
                    output_ret_dict = model.output_inference(
                        do_recognition=False,
                        recognition_beam_width=dev_best_recognition_beam_size,
                        do_translation=True,
                        translation_beam_width=tbw,
                        translation_beam_alpha=ta,
                        encoder_output=ret_dict["encoder_output"],
                        encoder_lgt=ret_dict["feat_len"],
                        src_mask=ret_dict["src_mask"],
                    )
                    total_translation.extend(output_ret_dict['translations'])

                metrics_rbw_ra = metrics_calculation(total_sent=None, total_conv_sent=None, total_translation=total_translation, total_info=dev_total_info)
                dev_translation_results[tbw][ta] = metrics_rbw_ra
                if (
                    dev_translation_results[tbw][ta]["bleu"]
                    > dev_best_bleu_score
                ):
                    dev_best_bleu_score = dev_translation_results[tbw][ta]["bleu"]
                    dev_best_translation_beam_size = tbw
                    dev_best_translation_alpha = ta
                    dev_best_translation_result = dev_translation_results[tbw][ta]
                    dev_best_total_translation = total_translation
                    recorder.print_log(
                            "[DEV] partition [Translation] results:\n\t"
                            "New Best Translation Beam Size: {:d} and Alpha: {:d}\n\t"
                            "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\n\t"
                            "CHRF {:.2f}\t"
                            "ROUGE {:.2f}".format(
                            dev_best_translation_beam_size,
                            dev_best_translation_alpha,
                            dev_best_translation_result["bleu"],
                            dev_best_translation_result["bleu_scores"][
                                "bleu1"
                            ],
                            dev_best_translation_result["bleu_scores"][
                                "bleu2"
                            ],
                            dev_best_translation_result["bleu_scores"][
                                "bleu3"
                            ],
                            dev_best_translation_result["bleu_scores"][
                                "bleu4"
                            ],
                            dev_best_translation_result["chrf"],
                            dev_best_translation_result["rouge"],
                        )
                    )


    recorder.print_log(
        "[DEV] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: {:d}\n\t"
        "Best Translation Beam Size: {:d} and Alpha: {:d}\n\t"
        "WER {:3.2f}\t(DEL: {:3.2f},\tINS: {:3.2f},\tSUB: {:3.2f})\n\t"
        "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\n\t"
        "CHRF {:.2f}\t"
        "ROUGE {:.2f}".format(
            dev_best_recognition_beam_size if do_recognition else -1,
            dev_best_translation_beam_size if do_translation else -1,
            dev_best_translation_alpha if do_translation else -1,
            dev_best_recognition_result["wer"] if do_recognition else -1,
            dev_best_recognition_result["wer_scores"]["del_rate"]
            if do_recognition
            else -1,
            dev_best_recognition_result["wer_scores"]["ins_rate"]
            if do_recognition
            else -1,
            dev_best_recognition_result["wer_scores"]["sub_rate"]
            if do_recognition
            else -1,
            dev_best_translation_result["bleu"] if do_translation else -1,
            dev_best_translation_result["bleu_scores"]["bleu1"]
            if do_translation
            else -1,
            dev_best_translation_result["bleu_scores"]["bleu2"]
            if do_translation
            else -1,
            dev_best_translation_result["bleu_scores"]["bleu3"]
            if do_translation
            else -1,
            dev_best_translation_result["bleu_scores"]["bleu4"]
            if do_translation
            else -1,
            dev_best_translation_result["chrf"] if do_translation else -1,
            dev_best_translation_result["rouge"] if do_translation else -1,
        )
    )

    # Work with test set
    test_total_gloss = []
    test_total_conv_gloss = []
    test_total_translation = []
    for idx, ret_dict in enumerate(test_encoder_ret_dicts):
        output_ret_dict = model.output_inference(
            do_recognition=True,
            recognition_beam_width=dev_best_recognition_beam_size,
            do_translation=True,
            translation_beam_width=dev_best_translation_beam_size,
            translation_beam_alpha=dev_best_translation_alpha,
            encoder_output=ret_dict["encoder_output"],
            encoder_lgt=ret_dict["feat_len"],
            src_mask=ret_dict["src_mask"],
        )

        test_total_gloss.extend(output_ret_dict["recognized_sents"])
        test_total_conv_gloss.extend(output_ret_dict["conv_sents"])
        test_total_translation.extend(output_ret_dict['translations'])

    test_best_result = metrics_calculation(total_sent=test_total_gloss, total_conv_sent=test_total_conv_gloss, total_translation=test_total_translation, total_info=test_total_info)

    recorder.print_log(
        "[TEST] partition [Recognition & Translation] results:\n\t"
        "Best CTC Decode Beam Size: {:d}\n\t"
        "Best Translation Beam Size: %d and Alpha: {:d}\n\t"
        "WER {:3.2f}\t(DEL: {:3.2f},\tINS: {:3.2f},\tSUB: {:3.2f})\n\t"
        "BLEU-4 {:.2f}\t(BLEU-1: {:.2f},\tBLEU-2: {:.2f},\tBLEU-3: {:.2f},\tBLEU-4: {:.2f})\n\t"
        "CHRF {:.2f}\t"
        "ROUGE {:.2f}".format(
            dev_best_recognition_beam_size if do_recognition else -1,
            dev_best_translation_beam_size if do_translation else -1,
            dev_best_translation_alpha if do_translation else -1,
            test_best_result["wer"] if do_recognition else -1,
            test_best_result["wer_scores"]["del_rate"]
            if do_recognition
            else -1,
            test_best_result["wer_scores"]["ins_rate"]
            if do_recognition
            else -1,
            test_best_result["wer_scores"]["sub_rate"]
            if do_recognition
            else -1,
            test_best_result["bleu"] if do_translation else -1,
            test_best_result["bleu_scores"]["bleu1"]
            if do_translation
            else -1,
            test_best_result["bleu_scores"]["bleu2"]
            if do_translation
            else -1,
            test_best_result["bleu_scores"]["bleu3"]
            if do_translation
            else -1,
            test_best_result["bleu_scores"]["bleu4"]
            if do_translation
            else -1,
            test_best_result["chrf"] if do_translation else -1,
            test_best_result["rouge"] if do_translation else -1,
        )
    )


    def _write_to_file(file_path: str, sequence_ids, hypotheses):
        with open(file_path, mode="w", encoding="utf-8") as out_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                out_file.write(seq + "|" + hyp + "\n")

    if output_path is not None:
        if do_recognition:
            dev_total_gls_output_path_set = "{}.BW_{:03d}.{}.total.gls".format(
                output_path, dev_best_recognition_beam_size, "dev"
            )
            _write_to_file(
                dev_total_gls_output_path_set,
                [info[0] for info in dev_total_info],
                dev_best_total_gloss,
            )

            dev_conv_gls_output_path_set = "{}.BW_{:03d}.{}.conv.gls".format(
                output_path, dev_best_recognition_beam_size, "dev"
            )
            _write_to_file(
                dev_conv_gls_output_path_set,
                [info[0] for info in dev_total_info],
                dev_best_conv_gloss,
            )

            test_total_gls_output_path_set = "{}.BW_{:03d}.{}.total.gls".format(
                output_path, dev_best_recognition_beam_size, "test"
            )
            _write_to_file(
                test_total_gls_output_path_set,
                [info[0] for info in test_total_info],
                test_total_gloss,
            )

            test_conv_gls_output_path_set = "{}.BW_{:03d}.{}.conv.gls".format(
                output_path, dev_best_recognition_beam_size, "test"
            )
            _write_to_file(
                test_conv_gls_output_path_set,
                [info[0] for info in test_total_info],
                test_total_gloss,
            )

        if do_translation:
            if dev_best_translation_beam_size > -1:
                dev_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "dev",
                )
                test_txt_output_path_set = "{}.BW_{:02d}.A_{:1d}.{}.txt".format(
                    output_path,
                    dev_best_translation_beam_size,
                    dev_best_translation_alpha,
                    "test",
                )
            else:
                dev_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "dev"
                )
                test_txt_output_path_set = "{}.BW_{:02d}.{}.txt".format(
                    output_path, dev_best_translation_beam_size, "test"
                )

            _write_to_file(
                dev_txt_output_path_set,
                [info[0] for info in dev_total_info],
                dev_best_total_translation,
            )
            _write_to_file(
                test_txt_output_path_set,
                [info[0] for info in test_total_info],
                test_total_translation,
            )

        with open(output_path + ".dev_results.pkl", "wb") as out:
            pickle.dump(
                {
                    "recognition_results": dev_recognition_results
                    if do_recognition
                    else None,
                    "translation_results": dev_translation_results
                    if do_translation
                    else None,
                },
                out,
            )
        with open(output_path + ".test_results.pkl", "wb") as out:
            pickle.dump(test_best_result, out)
    



def seq_feature_generation(loader, model, device, mode, work_dir, recorder):
    model.eval()

    src_path = os.path.abspath(f"{work_dir}{mode}")
    tgt_path = os.path.abspath(f"./features/{mode}")
    if not os.path.exists("./features/"):
    	os.makedirs("./features/")

    if os.path.islink(tgt_path):
        curr_path = os.readlink(tgt_path)
        if work_dir[1:] in curr_path and os.path.isabs(curr_path):
            return
        else:
            os.unlink(tgt_path)
    else:
        if os.path.exists(src_path) and len(loader.dataset) == len(os.listdir(src_path)):
            os.symlink(src_path, tgt_path)
            return

    for batch_idx, data in tqdm(enumerate(loader)):
        recorder.record_timer("device")
        vid = device.data_to_device(data[0])
        vid_lgt = device.data_to_device(data[1])
        with torch.no_grad():
            ret_dict = model(vid, vid_lgt)
        if not os.path.exists(src_path):
            os.makedirs(src_path)
        start = 0
        for sample_idx in range(len(vid)):
            end = start + data[3][sample_idx]
            filename = f"{src_path}/{data[-1][sample_idx].split('|')[0]}_features.npy"
            save_file = {
                "label": data[2][start:end],
                "features": ret_dict['framewise_features'][sample_idx][:, :vid_lgt[sample_idx]].T.cpu().detach(),
            }
            np.save(filename, save_file)
            start = end
        assert end == len(data[2])
    os.symlink(src_path, tgt_path)


def write2file(path, info, output):
    filereader = open(path, "w")
    for sample_idx, sample in enumerate(output):
        for word_idx, word in enumerate(sample):
            filereader.writelines(
                "{} 1 {:.2f} {:.2f} {}\n".format(info[sample_idx],
                                                 word_idx * 1.0 / 100,
                                                 (word_idx + 1) * 1.0 / 100,
                                                 word[0]))
