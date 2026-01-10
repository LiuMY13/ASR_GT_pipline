import re
import unicodedata


def text_normalize(text: str) -> str:
    """
    文本规范化函数，用于公平 CER/WER 计算。
    规则：
      - 全角转半角
      - 去除空白
      - 去除所有标点符号（中文/英文）
      - 英文转小写
      - 保留中文、字母、数字、空格
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. 全角转半角 (NFKC)
    text = unicodedata.normalize("NFKC", text)

    # 2. 转小写（处理英文）
    text = text.lower()

    # 3. 只保留：中文 + 英文字母 + 数字
    # \u4e00-\u9fff: 中文
    # a-z: 英文（已小写）
    # 0-9: 数字
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", text)

    # 4. 合并多个空格为单个空格，并去除首尾空格
    # text = re.sub(r"\s+", " ", text).strip()

    return text
