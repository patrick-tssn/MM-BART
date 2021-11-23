import os
import json
import logging
import random
import time
import copy
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import copy
import numpy as np
import setproctitle

import torch
from torch._C import device
import torch.nn.functional as F

from transformers import  BartTokenizer, BartConfig
from VideoBART import VideoBARTDecoderGenerationModel, VideoBARTGenerationModel
from dataset import get_dataset, build_input_from_segments


BART_SPECIAL_TOKENS = ["<s>", "</s>", "<speaker1>", "<speaker2>","<cap>", "<video>", "<pad>"]
BART_SPECIAL_TOKENS_DICT = {'bos_token': "<s>", 'eos_token': "</s>", 'additional_special_tokens': ["<speaker1>", "<speaker2>", "<video>", "<cap>"], 'pad_token': "<pad>"}
MODEL_INPUTS = ["input_ids", "token_type_ids","lm_labels"]
PADDED_INPUTS = ["input_ids", "token_type_ids","lm_labels"]

def beam_search(caption, history, tokenizer, model, args, current_output=None, video=None):
    if current_output is None:
        current_output = []
    hyplist = [([], 0., current_output)]
    best_state = None
    comp_hyplist = []
    
    SPECIAL_TOKENS = BART_SPECIAL_TOKENS
    from bart_dataset import build_input_from_segments

    for i in range(args.max_length):
        new_hyplist = []
        argmin = 0
        for out, lp, st in hyplist:
            encoder_instance, decoder_instance, instance, _ = build_input_from_segments(caption, history, st, tokenizer,  with_eos=False, drop_caption=False, model=args.model)
            eos = []
            eos.append(model.config.decoder_start_token_id)
            eos = torch.tensor(eos, device=args.device).unsqueeze(0)
            encoder_input_ids = torch.tensor(encoder_instance["input_ids"], device=args.device).unsqueeze(0) 
            encoder_token_type_ids = torch.tensor(encoder_instance["token_type_ids"], device=args.device).unsqueeze(0)
            decoder_token_type_ids = torch.tensor(decoder_instance["token_type_ids"], device=args.device).unsqueeze(0)
            if video is not None:
                if args.vasm:
                    encoder_token_type_ids = torch.cat([torch.ones((video.size(0), video.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), encoder_token_type_ids], dim=1)
                    decoder_token_type_ids = torch.cat([eos, torch.ones((video.size(0), video.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), eos, decoder_token_type_ids], dim=1)
                    dec_ids = torch.tensor(decoder_instance["input_ids"], device=args.device).unsqueeze(0)
                    dec_ids = torch.cat([eos, dec_ids], dim=1)
                    eos = eos.expand((video.size(0), 1, video.size(2)))
                    dec_video_ids = torch.cat([eos, video], dim=1)
                    logits = model(encoder_input_ids, video, token_type_ids=encoder_token_type_ids, decoder_input_ids = dec_ids, decoder_video_ids=dec_video_ids, decoder_token_type_ids=decoder_token_type_ids,train=0)
                else:
                    encoder_token_type_ids = torch.cat([torch.ones((video.size(0), video.size(1))).long().to(args.device) * tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[-2]), encoder_token_type_ids], dim=1)
                    decoder_token_type_ids = torch.cat([eos, decoder_token_type_ids], dim=1)
                    dec_ids = torch.tensor(decoder_instance["input_ids"], device=args.device).unsqueeze(0)
                    dec_ids = torch.cat([eos, dec_ids], dim=1)
                    dec_video_ids = video
                    logits = model(encoder_input_ids, video, token_type_ids=encoder_token_type_ids, decoder_input_ids = dec_ids, decoder_video_ids=None, decoder_token_type_ids=decoder_token_type_ids,train=0)
            else:
                decoder_token_type_ids = torch.cat([eos, decoder_token_type_ids], dim=1)
                dec_ids = torch.tensor(decoder_instance["input_ids"], device=args.device).unsqueeze(0)
                dec_ids = torch.cat([eos, dec_ids], dim=1)
                logits = model(encoder_input_ids, None, token_type_ids=encoder_token_type_ids, decoder_input_ids = dec_ids, decoder_token_type_ids=decoder_token_type_ids)
                
            logits = logits[0] # (bz, seq_len, vocab_size)
            logp = F.log_softmax(logits, dim=-1)[:, -1, :] # (bz, 1, vocab_size)
            lp_vec = logp.cpu().data.numpy() + lp
            lp_vec = np.squeeze(lp_vec) # (vocab_size)
            if i >= args.min_length:
                new_lp = lp_vec[tokenizer.eos_token_id] + args.penalty * (len(out) + 1) 
                comp_hyplist.append((out, new_lp))
                if best_state is None or best_state < new_lp:
                    best_state = new_lp
            count = 1
            for o in np.argsort(lp_vec)[::-1]:
                if o == tokenizer.unk_token_id or o == tokenizer.eos_token_id:
                    continue
                new_lp = lp_vec[o]
                if len(new_hyplist) == args.beam_size:
                    if new_hyplist[argmin][1] < new_lp:
                        new_st = copy.deepcopy(st)
                        new_st.append(int(o))
                        new_hyplist[argmin] = (out + [o], new_lp, new_st) 
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0]
                    else:
                        break
                else:
                    new_st = copy.deepcopy(st)
                    new_st.append(int(o))
                    new_hyplist.append((out + [o], new_lp, new_st))
                    if len(new_hyplist) == args.beam_size:
                        argmin = min(enumerate(new_hyplist), key=lambda h: h[1][1])[0] 
                count += 1
        hyplist = new_hyplist 
    if len(comp_hyplist) > 0:
        maxhyps = sorted(comp_hyplist, key=lambda h: -h[1])[:1] 
        return maxhyps
    else:
        return [([], 0)]

# Evaluation routine
def generate_response(model, data, dataset, args, ref_data=None):
    result_dialogs = []
    model.eval()
    with torch.no_grad():
        qa_id = 0
        for idx, dialog in enumerate(data['dialogs']):
            vid = dialog['image_id']
            out_dialog = dialog['dialog'][-1:]
            pred_dialog = {'image_id': vid,
                           'dialog': copy.deepcopy(out_dialog)}
            result_dialogs.append(pred_dialog)

            vgg = np.load("/data/avsd/vggish_testset/"+vid+".npy")
            i3d_flow = np.load("/data/avsd/i3d_flow_testset/"+vid+".npy")
            i3d_rgb = np.load("/data/avsd/i3d_rgb_testset/"+vid+".npy")

            # test sample way
            # sample_step = i3d_flow.shape[0] // vgg.shape[0]
            # if sample_step == 0:
            #     sample_step = 1
            sample_step = 1
            sample_i3d_flow = i3d_flow[range(1, i3d_flow.shape[0], sample_step)]
            sample_i3d_rgb = i3d_rgb[range(1, i3d_rgb.shape[0], sample_step)]

            vgg = torch.from_numpy(vgg).float().to(args.device)
            i3d_flow = torch.from_numpy(sample_i3d_flow).float().to(args.device)
            i3d_rgb = torch.from_numpy(sample_i3d_rgb).float().to(args.device)
            min_length = min([i3d_flow.size(0), i3d_rgb.size(0), vgg.size(0)])
            i3d = torch.cat([i3d_flow[:min_length], i3d_rgb[:min_length], vgg[:min_length]], dim=1).unsqueeze(0)

            for t, qa in enumerate(out_dialog):
                if args.log:
                    logging.info('%d %s_%d' % (qa_id, vid, t))
                    logging.info('QS: ' + qa['question'])
                # prepare input data
                start_time = time.time()
                qa_id += 1
                
                
                current_output = []
                current_output.append(model.config.decoder_start_token_id)
                if args.video:
                    hypstr = beam_search(dataset[idx]["caption"], dataset[idx]["history"], tokenizer, model, args, current_output=None, video=i3d)
                else:
                    hypstr = beam_search(dataset[idx]["caption"], dataset[idx]["history"], tokenizer, model, args, current_output=None, video=None)
                hypstr = hypstr[0][0]
                hypstr=tokenizer.decode(hypstr, skip_special_tokens=True)
                if args.log:
                    logging.info('HYP: ' + hypstr)
                pred_dialog['dialog'][t]['answer'] = hypstr
                if args.log:
                    logging.info('ElapsedTime: %f' % (time.time() - start_time))
                    logging.info('-----------------------')


    return {'dialogs': result_dialogs}


if __name__ =="__main__":
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (gpt or gpt2)")
    parser.add_argument("--max_history", type=int, default=3, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--beam_search", default=True, help="Set to use beam search instead of sampling")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--penalty", type=float, default=0.3, help="elngth penalty")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--task", type=str, default='8')
    parser.add_argument("--test_set", type=str, default="/share2/wangyx/data/avsd/test_set4DSTC8-AVSD.json")
    parser.add_argument("--lbl_test_set", type=str, default="/share2/wangyx/data/avsd/dstc7avsd_eval/data/lbl_undisclosedonly_test_set4DSTC7-AVSD.json")
    
    parser.add_argument("--ckptid", type=str, help='ckpt selected for test') # ckptid
    parser.add_argument("--gpuid", type=str, default='0', help='gpu id') # gpuid
    parser.add_argument("--log", type=bool, default=False, help='if logging info') # if print response
    parser.add_argument('--exp_set', type=str, default='test')
    parser.add_argument('--log_set', type=str, default='', help='log file name')
    parser.add_argument('--video', type=int, default=1) # if add video
    parser.add_argument('--vasm', type=int, default=0) # if add video reconstruct loss
    args = parser.parse_args()
    

    exp_set = args.exp_set
    if args.task == '8':
        args.test_set = "/data/avsd/test_set4DSTC8-AVSD.json"
    elif args.task == '7':
        args.test_set = "/data/avsd/test_set4DSTC7-AVSD.json"
    model_checkpoint = 'log/' + args.model + args.log_set + '/'
    output_dir = 'results/' + args.model + exp_set
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = output_dir + '/task_' + args.task + '_result_' + args.ckptid  + '_' + str(args.beam_size) + '_' + str(args.min_length) + '_' + str(args.penalty) + '.json'
    
    if args.device == 'cuda':
        args.device = 'cuda:' + args.gpuid  
    # setproctitle.setproctitle("task_{}_ckpt_{}_beam_{}_min_{}_pen_{:.1f}".format(args.task, args.ckptid, args.beam_size, args.min_length, args.penalty))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    for arg in vars(args):
        print("{}={}".format(arg, getattr(args, arg)))
    logging.basicConfig(level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s')
    logging.info('Loading model params from ' + model_checkpoint)
    
    if 'bart' in args.model:
        tokenizer_class = BartTokenizer
        model_class = VideoBARTGenerationModel
        model_config = BartConfig.from_pretrained(model_checkpoint)
        SPECIAL_TOKENS = BART_SPECIAL_TOKENS
        SPECIAL_TOKENS_DICT = BART_SPECIAL_TOKENS_DICT
    else:
        print('No pre-trained model: {}!'.format(args.model))
    tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
    tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT)
    model = model_class.from_pretrained(model_checkpoint+"checkpoint_mymodel_" + args.ckptid + ".pt", config=model_config)
    model.to(args.device)
    model.eval()


    if args.log:
        logging.info('Loading test data from ' + args.test_set)
    test_data = json.load(open(args.test_set,'r'))
    test_dataset = get_dataset(tokenizer, args.test_set, undisclosed_only=True, n_history=args.max_history)
    # generate sentences
    if args.log:
        logging.info('-----------------------generate--------------------------')
    start_time = time.time()
    result = generate_response(model, test_data, test_dataset, args)
    if args.log:
        logging.info('----------------')
        logging.info('wall time = %f' % (time.time() - start_time))
    if output_path:
        if args.log:
            logging.info('writing results to ' + output_path)
        json.dump(result, open(output_path, 'w'), indent=4)
    if args.log:
        logging.info('done')
