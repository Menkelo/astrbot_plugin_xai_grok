import re


def _strip_leading_command(text: str, command: str) -> str:
    """
    去掉开头命令本体（支持 / ! 前缀），保留其后全部内容（含比例、换行等）
    例：
      /视频 鹅在游泳 1:1 -> 鹅在游泳 1:1
      !画图\n猫咪 9:16 -> 猫咪 9:16
    """
    s = (text or "").strip()
    if not s:
        return ""

    # 只移除开头命令，不动后续文本
    pattern = rf"^\s*[\\/!]?{re.escape(command)}(?:\s+|$)"
    return re.sub(pattern, "", s, count=1, flags=re.IGNORECASE).strip()


def extract_prompt_after_command(message_str: str, command: str) -> str:
    """
    强化版命令提取：
    - 支持 /cmd、!cmd、cmd
    - 保留后续完整内容（包括 1:1 / 16:9 / 换行）
    - 若首行未匹配，退化为按行扫描，找到命令行后拼接剩余内容
    """
    text = (message_str or "").strip()
    if not text:
        return ""

    # 1) 快速路径：整段从命令开头
    direct = _strip_leading_command(text, command)
    if direct != text:
        return normalize_spaces_keep_ratio(direct)

    # 2) 兼容多行：逐行找首个命令
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(rf"^[\\/!]?{re.escape(command)}(?:\s+|$)", stripped, flags=re.IGNORECASE):
            first_line_rest = _strip_leading_command(stripped, command)
            tail = "\n".join(lines[i + 1:]).strip()
            merged = f"{first_line_rest}\n{tail}".strip() if tail else first_line_rest
            return normalize_spaces_keep_ratio(merged)

    # 3) 都没命中，原样返回（保险）
    return normalize_spaces_keep_ratio(text)


def normalize_spaces(text: str) -> str:
    """
    普通空白归一化：把连续空白折叠为1个空格
    """
    return re.sub(r"\s+", " ", (text or "")).strip()


def normalize_spaces_keep_ratio(text: str) -> str:
    """
    空白归一化（加强）：
    - 折叠多余空白
    - 修复比例写法中的空格：1 : 1 / 16 ： 9 -> 1:1 / 16:9
    """
    s = normalize_spaces(text)

    # 统一中文冒号为英文冒号
    s = s.replace("：", ":")

    # 压缩比例两侧空白： "16 : 9" -> "16:9"
    s = re.sub(r"(?<!\d)(\d{1,2})\s*:\s*(\d{1,2})(?!\d)", r"\1:\2", s)

    # 压缩尺寸写法： "1024 x 1792" / "1024×1792" -> "1024x1792"
    s = re.sub(r"(?<!\d)(\d{2,5})\s*[xX×]\s*(\d{2,5})(?!\d)", r"\1x\2", s)

    return s