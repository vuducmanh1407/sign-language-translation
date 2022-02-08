from signjoey.metrics import *
from evaluation.slt_vac_eval.python_wer_evaluation import *

def metrics_calculation(total_sent, total_conv_sent, total_translation, total_info):
    """
    total_info elements position:
    [0]: file name
    [-2]: gloss
    [-1]: translation
    """

    if total_sent is not None:
        wer_stat = list_wer_calculation([info[0] for info in total_info], [info[-2].split(" ") for info in total_info], total_sent, total_conv_sent)    
        gls_wer_score = wer_list(hypotheses=[" ".join(sent) for sent in total_sent], references=[info[-2] for info in total_info])
    
    if total_conv_sent is not None:
        conv_gls_wer_score = wer_list(hypotheses=[" ".join(sent) for sent in total_conv_sent], references=[info[-2] for info in total_info])   

    if total_translation is not None:
        txt_bleu = bleu(references=[info[-1] for info in total_info], hypotheses=[" ".join(t) for t in total_translation])
        txt_chrf = chrf(references=[info[-1] for info in total_info], hypotheses=[" ".join(t) for t in total_translation])
        txt_rouge = rouge(references=[info[-1] for info in total_info], hypotheses=[" ".join(t) for t in total_translation])

    valid_scores = {}
    if total_sent is not None:
        valid_scores["wer_stat"] = wer_stat
        valid_scores["wer"] = gls_wer_score["wer"]
        valid_scores["wer_scores"] = gls_wer_score
    elif total_sent is None and total_conv_sent is not None:
        valid_scores["wer"] = conv_gls_wer_score["wer"]
        valid_scores["wer_scores"] = conv_gls_wer_score

    if total_translation is not None:
        valid_scores["bleu"] = txt_bleu["bleu4"]
        valid_scores["bleu_scores"] = txt_bleu
        valid_scores["chrf"] = txt_chrf
        valid_scores["rouge"] = txt_rouge
    
    return valid_scores