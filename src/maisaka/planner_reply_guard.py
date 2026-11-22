"""Planner 未调用工具时的回复意图检测。

Planner 的正文不会展示给任何人，只有 reply 工具才能真正发言。
模型偶尔会在正文里写下“直接回复即可”“调用 reply”之后不调用任何工具就结束，
这里用规则识别这种情况，供主循环追加一次提醒后重试。

私聊还有另一种漏回复：模型只写分析，并声明本轮不调用工具，但对方刚发来的消息
既没有 reply，也没有 wait 或表情。这种正文没有“要回复”的字样，上面的意图规则
认不出。私聊在这种情况下仍提醒一次；群聊不走这条路径，以免改变发言频率。
"""

from typing import Sequence

import re

from src.maisaka.context.messages import ModelOutputContextMessage, SessionBackedMessage

# 结论通常写在最后：先看最后一小段有没有明确的“调用 reply”，再看稍长的末尾有没有较弱的发言表述
_FINAL_CLAUSE_CHARS = 100
_CONCLUSION_TAIL_CHARS = 220

# 明确写出要走 reply 的说法，例如“调用 reply”“直接reply即可”“准备发送回复”
_EXPLICIT_REPLY_PATTERN = re.compile(
    r"(?<![不别])(?<!不再)(?:调用|使用|通过|用|直接|走)\s*`?reply`?"
    r"|调用工具\s*[:：]\s*`?reply"
    r"|`?reply`?\s*(?:回复|工具)?\s*(?:即可|就行|就好)"
    r"|准备发送(?:一条)?(?:简短)?(?:清晰)?(?:的)?回复|现在执行回复"
    r"|直接(?:回复|回|接话|文本回应|生成回复)(?:即可|就行|就好)?[。.]?\s*$"
)

# 明确写出不发言的说法，出现在最后一小段时不再视为要回复
_EXPLICIT_NO_REPLY_PATTERN = re.compile(
    r"(?:不|无需|不用|不必|别)\s*(?:再)?(?:调用|使用|用|发|进行|需要)?\s*`?(?:reply|回复|发言)"
    r"|保持(?:沉默|安静|静默|潜水|观望|观察)|潜水|先不发|不发新回复"
)

# 较弱的发言表述，例如“回一句”“回复方向”“计划：…调侃”
_REPLY_INTENT_PATTERN = re.compile(
    r"(?:回复|回应|接话|接梗|吐槽)(?:即可|就行|就好|就够)"
    r"|(?:下一步|计划|决定|决策|结论|选择|方向)\s*[:：][^\n。]{0,40}(?:回复|回一句|接一句|发一条|回应|接住|吐槽|调侃|怼)"
    r"|(?:简短|轻量|轻轻)?(?:回|接|发)[^\n，。；]{0,8}?一[句条]"
    r"|回复方向|reply_reference|建议回复"
)

# 已经对这条用户消息做出可见动作或明确等待，私聊不再把“只分析”当成漏回复
_PRIVATE_TURN_SETTLED_TOOLS = frozenset({"reply", "send_emoji", "wait"})

# 私聊正文明确选择不发言或等待。不把“不需要工具调用”算进去：那句话只说明没调工具，
# 可见发言本身必须走 reply。
_PRIVATE_REPLY_DECLINE_PATTERN = re.compile(
    r"(?:不|无需|不用|不必|没必要|不该|不应|不宜|不需要|先不|暂不|不再|不强行|不硬)\s*"
    r"(?:再)?(?:追加)?(?:进行|发出|发送)?\s*`?(?:reply|回复|发言|接话|回应)"
    r"|不(?:发|调用|使用)\s*`?reply`?"
    r"|保持(?:沉默|安静|静默|观望|观察)"
    r"|(?<!不)(?:继续|先|选择|暂停|安静)?等待"
    r"|先看看|先观察|先放着"
    r"|已(?:经)?(?:回复|发出|发送|完成回复)"
)

# 表示不发言、已经发过或还要再看情况的说法，出现时较弱的发言表述不作数
_NO_REPLY_PATTERN = re.compile(
    r"(?:不|无需|不用|不必|没必要|不该|不应|不宜|不需要|先不|暂不|不再|不强行|不硬)\s*(?:再)?(?:追加)?(?:进行)?(?:回复|发言|接话|回应|插话|开口|补话|发送|发新回复|发)"
    r"|保持(?:沉默|安静|静默|观望|观察|潜水)|潜水|静观|观望|不主动"
    r"|(?:继续|先|选择|暂停|安静)?等待|等(?:一|他|她|对方|群友|群里|有人|\S{1,8}?(?:回|接|出来|反应))"
    r"|结束(?:本轮|思考|这轮)|本轮结束|这轮结束|先放着|收手|停手|不动"
    r"|已(?:经)?(?:回复|发出|发送|完成回复|回应|接住|接话)|刚(?:刚)?(?:回|发|说)(?:了|完|过)"
    r"|若|如果|等\S{0,10}再|再(?:决定|看|说|考虑|判断)"
)


def has_unsent_reply_intent(planner_content: str) -> bool:
    """判断未调用工具的 Planner 正文是否表达了“要回复”的结论。

    Args:
        planner_content: Planner 本轮输出的可见正文。

    Returns:
        bool: 正文末尾表达了发言意图且没有同时表达不发言时返回 True。
    """

    content = planner_content.strip()
    if not content:
        return False

    final_clause = content[-_FINAL_CLAUSE_CHARS:]
    if _EXPLICIT_REPLY_PATTERN.search(final_clause) and not _EXPLICIT_NO_REPLY_PATTERN.search(final_clause):
        return True

    conclusion_tail = content[-_CONCLUSION_TAIL_CHARS:]
    if _NO_REPLY_PATTERN.search(conclusion_tail):
        return False
    return _REPLY_INTENT_PATTERN.search(conclusion_tail) is not None


def declines_visible_reply(planner_content: str) -> bool:
    """判断正文末尾是否明确选择不发言或继续等待。"""

    content = planner_content.strip()
    if not content:
        return False
    return _PRIVATE_REPLY_DECLINE_PATTERN.search(content[-_CONCLUSION_TAIL_CHARS:]) is not None


def _is_other_person_message(message: SessionBackedMessage) -> bool:
    """真实用户消息。机器人自己的回声不算一条还等着回复的来信。"""

    if message.source_kind != "user":
        return False
    original_message = message.original_message
    if original_message is None:
        return True
    # 延迟导入，避免回复守卫在加载时拉起聊天包。
    from src.chat.utils.utils import is_bot_self

    user_info = original_message.message_info.user_info
    if is_bot_self(original_message.platform, user_info.user_id):
        return False
    return True


def has_unanswered_direct_user_message(history: Sequence[object]) -> bool:
    """从最新历史往回看，对方消息之后是否还没有 reply、表情或 wait。"""

    for message in reversed(tuple(history)):
        if isinstance(message, ModelOutputContextMessage):
            tool_names = {tool_call.func_name for tool_call in message.tool_calls}
            if tool_names & _PRIVATE_TURN_SETTLED_TOOLS:
                return False
            continue
        if not isinstance(message, SessionBackedMessage):
            continue
        if message.source_kind == "guided_reply":
            return False
        if _is_other_person_message(message):
            return True
    return False


def should_retry_private_reply_omission(
    planner_content: str,
    history: Sequence[object],
    *,
    is_group_chat: bool,
    focus_active: bool,
) -> bool:
    """私聊里对方消息仍未被处理，且模型只留下分析时，允许提醒一次。

    群聊和 Focus 不使用这条规则。正文已经写明不发言或等待时也不重试。
    """

    if is_group_chat or focus_active:
        return False
    if declines_visible_reply(planner_content):
        return False
    return has_unanswered_direct_user_message(history)
