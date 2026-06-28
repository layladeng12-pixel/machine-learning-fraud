#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file  : batch_predict.py
@desc  : 批量预测并评估模型性能

使用方法:
    # 仅预测（无标签）
    python batch_predict.py --input my_dataset/processed\ data/test_no_label.csv

    # 预测并评估（需要原标签）
    python batch_predict.py --input my_dataset/processed\ data/test_no_label.csv \
                            --label_file my_dataset/processed\ data/test.csv
"""
import argparse
import json
import os

import pandas as pd
import torch
import tokenizers
from pypinyin import pinyin, Style
from tokenizers import BertWordPieceTokenizer
from torch.nn import functional as F
from torch.utils.data import DataLoader
from functools import partial

from models.modeling_glycebert import GlyceBertForSequenceClassification


class PinyinMapper:
    """拼音映射工具类"""
    def __init__(self, chinese_bert_path):
        self.config_path = os.path.join(chinese_bert_path, 'config')
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def convert_sentence_to_pinyin_ids(self, sentence, tokenizer_output):
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True,
                            errors=lambda x: [['not chinese'] for _ in x])
        pinyin_locs = {}
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)
        return pinyin_ids


def predict_batch(model, texts, tokenizer, pinyin_mapper, device, max_length=256, batch_size=16):
    """批量预测"""
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_input_ids = []
            batch_pinyin_ids = []

            for text in batch_texts:
                bert_output = tokenizer.encode(text)
                bert_tokens = bert_output.ids

                if len(bert_tokens) > max_length:
                    bert_tokens = bert_tokens[:max_length - 1] + [bert_output.ids[-1]]

                pinyin_tokens = pinyin_mapper.convert_sentence_to_pinyin_ids(text, bert_output)
                pinyin_tokens = pinyin_tokens[:len(bert_tokens)]

                batch_input_ids.append(bert_tokens)
                batch_pinyin_ids.append([p for p in pinyin_tokens])

            # padding到相同长度
            max_len = max(len(x) for x in batch_input_ids)
            for j in range(len(batch_input_ids)):
                pad_len = max_len - len(batch_input_ids[j])
                batch_input_ids[j].extend([0] * pad_len)
                batch_pinyin_ids[j].extend([[0]*8 for _ in range(pad_len)])

            input_ids = torch.LongTensor(batch_input_ids).to(device)
            pinyin_ids = torch.LongTensor([[p for p in pt] for pt in batch_pinyin_ids]).to(device)
            pinyin_ids = pinyin_ids.view(len(batch_input_ids), -1, 8)

            outputs = model(input_ids, pinyin_ids)
            logits = outputs[0]
            probs = F.softmax(logits, dim=-1)

            preds = torch.argmax(probs, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs[:, 1].cpu().numpy())

    return all_preds, all_probs


def main():
    parser = argparse.ArgumentParser(description="批量预测诈骗电话")
    parser.add_argument("--bert_path", type=str, default="model/ChineseBERT-base")
    parser.add_argument("--model_path", type=str,
                        default="output/fraud_detection/best_model.bin")
    parser.add_argument("--input", type=str, required=True,
                        help="待预测数据路径（仅含clean_text）")
    parser.add_argument("--label_file", type=str, default=None,
                        help="原标签文件（用于评估，可选）")
    parser.add_argument("--output", type=str, default="my_dataset/prediction_result.csv",
                        help="预测结果保存路径")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print(f"\n加载数据: {args.input}")
    df = pd.read_csv(args.input)
    texts = df['clean_text'].tolist()
    print(f"样本数: {len(texts)}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    vocab_file = os.path.join(args.bert_path, 'vocab.txt')
    tokenizer = BertWordPieceTokenizer(vocab_file)
    pinyin_mapper = PinyinMapper(args.bert_path)

    model = GlyceBertForSequenceClassification.from_pretrained(args.bert_path)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    # 预测
    print("\n正在预测...")
    preds, probs = predict_batch(model, texts, tokenizer, pinyin_mapper, device, batch_size=args.batch_size)

    # 保存结果
    result_df = pd.DataFrame({
        'clean_text': texts,
        'prediction': preds,
        'fraud_probability': probs
    })
    result_df.to_csv(args.output, index=False)
    print(f"预测结果已保存: {args.output}")

    # 如果有标签，进行评估
    if args.label_file:
        print(f"\n加载原标签: {args.label_file}")
        label_df = pd.read_csv(args.label_file)
        true_labels = label_df['label'].tolist()

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

        accuracy = accuracy_score(true_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, preds, average='binary', zero_division=0
        )

        print("\n" + "=" * 50)
        print("模型评估结果")
        print("=" * 50)
        print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"精确率: {precision:.4f} ({precision*100:.2f}%)")
        print(f"召回率: {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1分数: {f1:.4f} ({f1*100:.2f}%)")

        cm = confusion_matrix(true_labels, preds)
        # 处理单一类别的情况：补全2x2混淆矩阵
        if cm.shape == (1, 1):
            if true_labels[0] == 0:
                cm_full = [[cm[0][0], 0], [0, 0]]
            else:
                cm_full = [[0, 0], [0, cm[0][0]]]
        else:
            cm_full = cm.tolist()
        print(f"\n混淆矩阵:")
        print(f"              预测非诈骗  预测诈骗")
        print(f"实际非诈骗       {cm_full[0][0]:>5}     {cm_full[0][1]:>5}")
        print(f"实际诈骗         {cm_full[1][0]:>5}     {cm_full[1][1]:>5}")
        print("=" * 50)


if __name__ == '__main__':
    main()
