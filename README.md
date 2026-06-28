# ChineseBERT 诈骗通话识别与基于社会工程策略的诈骗通话文本改写检验鲁棒性

## 项目简介

本项目实现了ChineseBERT对识别诈骗通话场景的训练和微调，基于大语言模型的诈骗通话文本改写，并利用 ChineseBERT 对改写后的数据集进行训练和预测，以检验ChineseBERT模型的鲁棒性。

项目主要包含以下三个部分：


* ChineseBERT 模型训练
* * 数据集改写
* 模型预测与结果评估

---

## 1. ChineseBERT 模型训练

由于 ChineseBERT 预训练模型体积较大，因此本仓库**未上传模型参数文件**。

请先前往 ChineseBERT 官方仓库下载预训练模型：

> https://github.com/ShannonAI/ChineseBert

下载完成后，请按照项目代码中的路径要求，将预训练模型放置到对应目录。

根据诈骗数据集进行ChineseBERT模型的训练和微调，可使用以下命令：
```bash
cd /workspace/ChineseBert && python train_fraud_detection.py \
    --train_data "my_dataset/processed data/train.csv" \
    --val_data "my_dataset/processed data/val.csv" \
    --batch_size 4 \
--num_epochs 3
```
注意：上述命令仅为示例。
得到最优模型保存于output即可，由于模型参数的原因，并未在此上传实验训练得到的最优模型。

---

## 2. 数据集改写

项目仅提供诈骗通话文本改写代码的其中一个示例，所有的改写都基于大模型完成，因此仅需要在示例的基础上改变Prompt即可。整体能够达到在保持原始语义基本一致的前提下，对文本表达方式进行调整，生成新的诈骗通话数据的效果。

运行改写代码：

```bash
python urgency_test_generator.py
```

运行结束后，生成使用制造紧迫感策略的改写数据，可以将Prompt改成基于”建立信任“”情感操纵“策略或者递进式策略的提示词，改写成不同的数据集。

---

## 3. 模型预测

训练完成后，可以利用batch_predict.py针对原始测试集和改写后的测试集进行批量预测和模型评估：

```bash
cd /workspace/ChineseBert && python batch_predict.py \
    --input "/workspace/ChineseBert/fraud_aug/aug_dataset/trust_test_dataset.csv" \
    --label_file "/workspace/ChineseBert/fraud_aug/aug_dataset/trust_test_dataset.csv" \
    --model_path output/fraud_detection/best_model.bin \
--output my_dataset/trust_test_result.csv
```

预测结果将输出至指定目录。

---
