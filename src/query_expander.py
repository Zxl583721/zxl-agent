COURSE_QUERY_HINTS = (
    (("雷达方程", "作用距离", "距离方程"), ("第1讲", "原理组成", "历史", "雷达方程")),
    (("噪声", "检测", "门限", "阈值", "虚警", "相参积累"), ("第2讲", "信号检测", "检测概率", "虚警概率")),
    (("测距", "测速", "测角", "多普勒", "精度"), ("第3讲", "测量精度", "距离", "速度", "角度")),
    (("杂波", "海杂波", "地杂波", "表面杂波"), ("第4讲", "杂波理论", "目标检测")),
    (("传播", "损耗", "折射", "波导", "绕射"), ("第5讲", "传播损耗", "大气折射", "传播因子")),
    (("天线", "相控阵", "波束", "阵列", "扫描"), ("第6讲", "天线", "相控阵技术", "方向图")),
    (("发射", "接收", "接收机", "动态范围", "相位噪声"), ("第7讲", "发射技术", "接收技术")),
    (("试验", "测试", "靶场", "测量雷达", "跟踪雷达"), ("第8讲", "雷达试验", "试验方案")),
    (("对抗", "干扰", "抗干扰", "电子战", "DRFM"), ("第9讲", "对抗技术", "ECCM", "干扰信号")),
    (("课程安排", "参考", "要求"), ("前言", "课程安排", "参考及要求")),
)

SYNONYM_GROUPS = (
    ("门限", "阈值", "检测门限"),
    ("虚警", "虚警概率", "误警"),
    ("检测概率", "发现概率"),
    ("测速", "速度测量", "多普勒"),
    ("测角", "角度测量", "方位角", "俯仰角"),
    ("测距", "距离测量", "时延"),
    ("传播损耗", "路径损耗", "衰减"),
    ("波导", "大气波导", "非标准传播"),
    ("相控阵", "阵列天线", "电子扫描"),
    ("动态范围", "线性范围", "接收机动态范围"),
    ("干扰", "压制干扰", "欺骗干扰", "电子干扰"),
    ("对抗", "抗干扰", "ECCM", "电子战"),
)


def expand_keyword_queries(query: str, max_queries: int = 3) -> list[str]:
    """Build deterministic keyword-search variants for a retrieval question."""
    query = str(query).strip()
    if not query:
        return []

    expansions = []
    keyword_terms = collect_matching_terms(query)
    if keyword_terms:
        expansions.append(compose_query(query, keyword_terms))

    course_terms = collect_course_terms(query)
    if course_terms:
        expansions.append(compose_query(query, course_terms))

    merged_terms = unique_terms([*keyword_terms, *course_terms])
    if merged_terms:
        expansions.append(compose_query(query, merged_terms))

    return unique_terms(expansions)[:max_queries]


def collect_matching_terms(query: str) -> list[str]:
    terms = []
    for group in SYNONYM_GROUPS:
        if any(term.lower() in query.lower() for term in group):
            terms.extend(group)
    return unique_terms(terms)


def collect_course_terms(query: str) -> list[str]:
    terms = []
    for markers, hints in COURSE_QUERY_HINTS:
        if any(marker.lower() in query.lower() for marker in markers):
            terms.extend(hints)
    return unique_terms(terms)


def compose_query(query: str, terms: list[str]) -> str:
    terms_text = " ".join(term for term in terms if term and term not in query)
    if not terms_text:
        return query
    return f"{query}\n检索扩展词：{terms_text}"


def unique_terms(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = str(item).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
