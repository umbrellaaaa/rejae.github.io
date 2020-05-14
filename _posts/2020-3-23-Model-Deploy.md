---
layout:     post
title:      深度学习模型部署
subtitle:   
date:       2020-3-23
author:     RJ
header-img: 
catalog: true
tags:
    - job

---
<p id = "build"></p>
---

## 前言
之前参考github上，基于Flask搭建了一个简单的NER和写诗模型的部署。没有考虑太多问题，做的也很简陋。

现在公司需要一个流式数据实时的处理英中机器翻译。现在我的模型是通过文件输入的，然后输出翻译结果文件。调用一次后，就结束了，这显然不行。

需要了解数据流相关的处理，模型那边应该随时在线，和循环等待输入类似。

## redis
公司采用的是redis内存数据库，接受到请求后按队列处理。

## django




## 模型部署

```python
# -*- coding:utf8 -*-
#copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""
import os
import time
from collections import namedtuple
import fileinput
import torch
from fairseq import checkpoint_utils, options, tasks, utils
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import redis
import simplejson as json
import fastBPE

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')
bpe = fastBPE.fastBPE('processed/all.en.bpe.codes', 'processed/dict.en.txt')


def preprocess(file_path, out_path):
    print("preprocess  - %s, %s" %(file_path, out_path))
    with open(file_path, 'r', encoding='utf-8') as f:
        with open(out_path, 'w', encoding='utf-8') as f_out:
            data = f.read().splitlines() # paragraph
            comp = re.compile('[^-^,^.^$^%^A-Z^a-z^0-9^ ]')
            wordnet_lemmatizer = WordNetLemmatizer()
            for item in data:
                if len(item)==0:
                    continue
                word_tokens = word_tokenize(item)
                sentence = ' '.join([it.replace('-',' - ').replace('.',' . ') for it in word_tokens])
                sentence = comp.sub(' ', sentence)
                f_out.write(sentence+' \n')


def split_by_period(string):
    data = string.split('. ')
    content=[]
    for item in data:
        if len(item.strip())>1:
            temp=item.replace('\n',' ').replace(',',' , ').replace('-',' - ')
            content.append(temp.strip()+'.')
    return content


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            content = split_by_period(src_str)
            for item in content:
                buffer.append(item.strip())
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    print('line is::::',lines)
    lines = bpe.apply(lines)
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    print('token is :',tokens)
    # replace unk token to #  
 
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    utils.import_user_module(args)
    # 检测输入、临时和输出文件夹是否存在，存在：清空，不存在：创建
    basedir = str(args.input).split("_")[0]
    gpu_use = str(args.input).split("_")[1]
    input_dir = "%s/input" %(basedir)
    temp_dir = "%s/temp" %(basedir)
    output_dir = "%s/output" %(basedir)
    if os.path.isdir(input_dir):
        os.system("rm -fr %s/*" % (input_dir))
    else:
        os.system("mkdir -p %s" % (input_dir))
    if os.path.isdir(temp_dir):
        os.system("rm -fr %s/*" % (temp_dir))
    else:
        os.system("mkdir -p %s" % (temp_dir))
    if os.path.isdir(output_dir):
        os.system("rm -fr %s/*" % (output_dir))
    else:
        os.system("mkdir -p %s" % (output_dir))

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    if use_cuda:
        torch.cuda.set_device(int(gpu_use))

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=True,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(args)

    # Hack to support GPT-2 BPE
    if args.remove_bpe == 'gpt2':
        from fairseq.gpt2_bpe.gpt2_encoding import get_encoder
        decoder = get_encoder(
            'fairseq/gpt2_bpe/encoder.json',
            'fairseq/gpt2_bpe/vocab.bpe',
        )
        encode_fn = lambda x: ' '.join(map(str, decoder.encode(x)))
    else:
        decoder = None
        encode_fn = lambda x: x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)
    print(align_dict)
    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    # 处理输入数据
    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Type the input sentence and press return:')
    start_id = 0

    redisconn =redis.StrictRedis(host='127.0.0.1',port='6379')
    while True:
        # 检测文件夹中是否有文件？
        tran_file = ""
        taskID = ""
        if redisconn.llen('list_trans_text') > 0:
            data = redisconn.rpop('list_trans_text')
            if data is None:
                time.sleep(0.02)
                continue

            obj = json.loads(data)
            taskID = obj['taskID']
            tran_file = "%s.txt"%(taskID)

            # need_trans_files = os.listdir(input_dir)
            # if len(need_trans_files) > 0:
            #     for tran_file in need_trans_files:

            if not os.path.exists("%s/%s" % (input_dir, tran_file)):
                print("%s/%s not found!" % (input_dir, tran_file))
                continue
            # 保存翻译的中间结果
            m_f = open("%s/trans_%s" % (temp_dir, tran_file), 'w', encoding='utf-8')
            # 打开输入文件
            for inputs in buffered_read("%s/%s" % (input_dir, tran_file), args.buffer_size):
                results = []
                for batch in make_batches(inputs, args, task, max_positions, encode_fn):
                    src_tokens = batch.src_tokens
                    src_lengths = batch.src_lengths
                    # print(src_tokens)
                    if use_cuda:
                        src_tokens = src_tokens.cuda()
                        src_lengths = src_lengths.cuda()

                    sample = {
                        'net_input': {
                            'src_tokens': src_tokens,
                            'src_lengths': src_lengths,
                        },
                    }
                    translations = task.inference_step(generator, models, sample)
                    # print('translation:',type(translations),translations)
                    for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                        src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                        results.append((start_id + id, src_tokens_i, hypos))

                # sort output to match input order
                for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                        print('S-{}\t{}'.format(id, src_str))

                    # Process top predictions
                    for hypo in hypos[:min(len(hypos), args.nbest)]:
                        # print('type(hypos)',type(hypos),hypos)
                        # print('hype alignment',hypo['alignment'])
                        # print('the inputs is :',inputs)
                        hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                            hypo_tokens=hypo['tokens'].int().cpu(),
                            src_str=src_str,
                            #input_str=inputs[0],    # 考虑替换成 args.input
                            alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                            align_dict=align_dict,
                            tgt_dict=tgt_dict,
   
                        )
                        if decoder is not None:
                            hypo_str = decoder.decode(map(int, hypo_str.strip().split()))
                        print('H-{}\t{}\t{}'.format(id, hypo['score'], hypo_str))
                        # m_f.write('src:'+src_str+'\n')
                        trunk = len(hypo_str)
                        if '#' in hypo_str:
                            trunk = hypo_str.index('#')
                        hypo_str = hypo_str[:trunk]
                        # m_f.write('result:'+ hypo_str.replace('@','')+'\n')
                        m_f.write(hypo_str.replace('@', '').replace(' ', '')+'\n')
                        print('P-{}\t{}'.format(
                            id,
                            ' '.join(map(lambda x: '{:.4f}'.format(x), hypo['positional_scores'].tolist()))
                        ))
                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

                # update running id counter
                start_id += len(inputs)
            m_f.close()
            # 处理结果文件
            # print("main deal ------ ")
            cmd = 'rm -fr "%s/%s"' % (input_dir, tran_file)
            # print(cmd)
            os.system(cmd)
            cmd = 'rm -fr "%s/%s_pre"' % (input_dir, tran_file)
            # print(cmd)
            os.system(cmd)
            cmd = 'mv -f "%s/trans_%s" "%s/%s"' % (temp_dir, tran_file, output_dir, tran_file)
            # print(cmd)
            os.system(cmd)
            time.sleep(0.02)

        # else:
        #     print("No File In %s" % (input_dir, ))
        #     time.sleep(3)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()


```