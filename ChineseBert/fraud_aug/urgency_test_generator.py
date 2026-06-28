import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


# 配置区域
API_KEY = ""

INPUT_FILE = "/workspace/ChineseBert/my_dataset/processed_data/test.csv"

OUTPUT_FILE = "/workspace/ChineseBert/fraud_aug/aug_dataset/urgency_test_dataset.csv"


SAMPLE_SIZE = 1000

# 并发数
MAX_WORKERS = 5

MODEL_NAME = "deepseek-chat"


# 初始化客户端


client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com"
)

# Prompt

def build_prompt(text):

    return f"""
下面是一段诈骗通话文本：

{text}

任务：

仅对原文本进行局部增强。

增强策略：

通过加入时间压力、紧急情况、立即处理、后果警告等表达，增强诈骗者制造紧迫感的效果。

可以采用以下方式：

强调时间限制
强调必须立即处理
强调延误后果
强调系统即将执行某种操作
强调当前处于关键处理阶段

严格要求：

保留原诈骗类型
保留原公司名称
保留原人物身份
保留原事件背景
保留原诈骗流程
保留原诈骗诱导动作
不得删除关键诈骗行为
不得创造新的诈骗故事
不得创造新的公司
不得创造新的活动
不得增加新的诈骗方式
不得改变诈骗目的

禁止新增：

- 名额有限
- 候补机制
- 专属额度
- 排队资格
- 内部通道
- 专属名额
- 系统释放给他人
允许：

仅在原句前后增加少量体现紧迫感的话术。

长度要求：

增强内容控制在原文长度的10%-30%以内
不得大幅扩写

禁止：

输出标题
输出分析
输出解释
输出“增强后文本”
输出“以下是”

禁止加入：

身份说明扩展
客户回访
老客户关系建立
正常寒暄
公司背景介绍
长篇信任建立内容

这些属于建立信任策略。

禁止加入：

同情表达
请求帮助
情绪化语言
愧疚感引导
亲情压力
道德施压

这些属于情感操纵策略。
紧迫感只能通过以下方式体现：

- 尽快处理
- 账户风险
- 包裹即将退回
- 兑奖期限
- 审核即将结束

禁止新增：

- 名额有限
- 候补机制
- 专属额度
- 排队资格
- 内部通道
- 专属名额
- 系统释放给他人

直接输出最终通话内容。
"""


# 清洗模型输出
def clean_output(text):

    patterns = [
        r"以下是.*?:",
        r"增强后.*?:",
        r"建立信任.*?:",
        r"Trust Building.*?:"
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )

    return text.strip()


# 质量检查


def quality_check(original, generated):

    if len(generated) < 20:
        return False

    # 防止模型重新编故事
    if len(generated) > len(original) * 2:
        return False

    bad_words = [
        "增强后文本",
        "以下是",
        "建立信任",
        "Trust Building",
        "策略说明"
    ]

    for word in bad_words:
        if word in generated:
            return False

    return True


# 调用模型

def rewrite(text):

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": build_prompt(text)
            }
        ],
        temperature=0.5
    )

    result = response.choices[0].message.content

    return clean_output(result)

# 单条处理

def process_row(row):

    try:

        original_text = row["clean_text"]

        new_text = rewrite(original_text)

        if not quality_check(
            original_text,
            new_text
        ):
            return None

        return {
            "clean_text": new_text,
            "label": row["label"],
            "fraud_type": row["fraud_type"],
            "augment_type": "urgency"
        }

    except Exception as e:

        print("出错：", e)

        return None

# 主程序

def main():

    df = pd.read_csv(INPUT_FILE)

    # 只保留诈骗样本
    sample_df = df[df["label"] == 1].copy()

    print("开始生成...")
    print("诈骗样本数：", len(sample_df))

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                process_row,
                row
            )
            for _, row in sample_df.iterrows()
        ]

        completed = 0

        for future in as_completed(futures):

            result = future.result()

            completed += 1

            print(
                f"完成 {completed}/{len(futures)}"
            )

            if result is not None:
                results.append(result)

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("生成完成")
    print("有效样本数：", len(result_df))
    print("保存文件：", OUTPUT_FILE)

if __name__ == "__main__":
    main()