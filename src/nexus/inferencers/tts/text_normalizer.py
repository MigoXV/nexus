"""
TTS 文本正则化工具
用于将文本分割成适合 TTS 处理的句子块
"""

import re
from typing import List

# 用于分句的标点符号（移除逗号，只保留句号、问号、感叹号等）
SENTENCE_DELIMITERS = re.compile(r'([。！？.!?；;]+)')


def split_text_by_punctuation(text: str) -> List[str]:
    """
    根据标点符号分割文本为句子块。
    
    :param text: 输入文本
    :return: 分割后的句子列表
    """
    if not text:
        return []
    
    # 先去掉开头的换行符，再将其余换行符替换为句号
    text = text.lstrip('\n\r')
    text = text.replace('\n', '。').replace('\r', '')
    
    # 使用标点符号分割，保留标点
    parts = SENTENCE_DELIMITERS.split(text)
    
    sentences = []
    current = ""
    for i, part in enumerate(parts):
        if not part:
            continue
        current += part
        # 如果是标点符号，则当前句子完成
        if SENTENCE_DELIMITERS.match(part):
            if current.strip():
                sentences.append(current.strip())
            current = ""
    
    # 处理末尾没有标点的部分
    if current.strip():
        sentences.append(current.strip())
    
    return sentences


def normalize_for_tts(text: str) -> str:
    """
    对文本进行 TTS 前的正则化处理。
    
    :param text: 输入文本
    :return: 正则化后的文本
    """
    if not text:
        return ""
    
    # 将换行符替换为句号
    text = text.replace('\n', '。').replace('\r', '')
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 可以在这里添加更多的文本正则化规则
    # 例如：数字转文字、缩写展开等
    
    return text
