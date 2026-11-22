from datetime import datetime
from types import SimpleNamespace

import pytest

from src.common.data_models.llm_service_data_models import LLMResponseResult
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.config.config import global_config
from src.llm_models.payload_content.tool_option import ToolCall
from src.maisaka.context.messages import (
    SessionBackedMessage,
    ToolResultMessage,
    build_model_output_context_messages,
)
from src.maisaka.planner_reply_guard import (
    declines_visible_reply,
    has_unanswered_direct_user_message,
    has_unsent_reply_intent,
    should_retry_private_reply_omission,
)
from src.maisaka.tool_result_condenser import build_tool_result_condense_task, should_condense_tool_result

# 合成的失败形态：模型声明不调用工具，正文也没有“要回复”的说法。
_ANALYSIS_ONLY = (
    "本轮不需要工具调用。\n对方刚发送了私聊消息。现有上下文已经足够理解场面，没有额外资料需要查找。"
)


def _user_message(
    text: str,
    *,
    source_kind: str = "user",
    original_message: object | None = None,
) -> SessionBackedMessage:
    return SessionBackedMessage(
        raw_message=MessageSequence([TextComponent(text)]),
        visible_text=text,
        timestamp=datetime(2026, 11, 22, 12, 0, 0),
        message_id="synthetic-user",
        original_message=original_message,
        source_kind=source_kind,
    )


def _model_turn(content: str, tool_calls: list[ToolCall] | None = None) -> list[object]:
    result = LLMResponseResult.from_portable_output(response=content, tool_calls=tool_calls)
    return list(build_model_output_context_messages(result.output_items))


def _tool_result(tool_name: str, call_id: str) -> ToolResultMessage:
    return ToolResultMessage(
        content=f"{tool_name} 完成",
        timestamp=datetime(2026, 11, 22, 12, 0, 1),
        tool_call_id=call_id,
        logical_turn_id="synthetic-turn",
        tool_name=tool_name,
    )


@pytest.mark.parametrize(
    "planner_content",
    [
        "牛牛在吐槽版本。\n\n不需要调用工具，直接回复即可。",
        "可以顺着调侃一句。\n\n调用工具：无必要，直接通过reply发送回复即可。",
        "决定：调用reply，回复SCh的消息（msg_id: \"1884926678\"），语气带点调侃。",
        "下一步动作：\n- 用 reply 把结论给出去。\n\n准备发送回复。",
        "计划：简短回喵七一句，自嘲刚才的撤回事件，把气氛收住。",
    ],
)
def test_detects_reply_intent_without_tool_call(planner_content: str) -> None:
    assert has_unsent_reply_intent(planner_content)


@pytest.mark.parametrize(
    "planner_content",
    [
        "",
        "群里没有新消息，也没人接话。保持观望，不发言。",
        "**决策：不调用任何工具，不进行回复。**",
        "计划：不发reply，也不贴表情，继续等下一个明确递话的话头。",
        "刚才已经回过“好耶”，此时再插进去反而刻意。\n\n下一步计划：等待群友接话，有新消息再判断是否回复。",
        "下一步计划：\n- 不再调用 reply，等二憨补充测试信息。",
    ],
)
def test_ignores_decisions_not_to_reply(planner_content: str) -> None:
    assert not has_unsent_reply_intent(planner_content)


def test_analysis_only_text_is_not_an_explicit_reply_intent() -> None:
    """现有实验开关只认正文里的回复意图，认不出“不调用工具”的分析。"""

    assert not has_unsent_reply_intent(_ANALYSIS_ONLY)
    assert not declines_visible_reply(_ANALYSIS_ONLY)
    assert not declines_visible_reply("本轮不需要工具调用。")
    assert not declines_visible_reply("本轮不需要工具调用。\n如果只看最近一句，对方是在打招呼。")


@pytest.mark.parametrize(
    "planner_content",
    [
        "保持观望，不发言。",
        "计划：不发reply，先等待对方补充。",
        "已经回复过了，这轮不再发言。",
    ],
)
def test_explicit_silence_or_wait_is_not_a_private_omission(planner_content: str) -> None:
    history = [_user_message("你好")]

    assert declines_visible_reply(planner_content)
    assert not should_retry_private_reply_omission(
        planner_content,
        history,
        is_group_chat=False,
        focus_active=False,
    )


def test_private_query_then_analysis_still_needs_a_reply() -> None:
    history: list[object] = [_user_message("你好，收到请回一句。")]
    history.extend(
        _model_turn(
            "先查一下再决定。",
            [
                ToolCall(call_id="call-profile", func_name="query_person_profile", args={"name": "对方"}),
                ToolCall(call_id="call-memory", func_name="query_memory", args={"query": "近况"}),
            ],
        )
    )
    history.append(_tool_result("query_person_profile", "call-profile"))
    history.append(_tool_result("query_memory", "call-memory"))

    assert has_unanswered_direct_user_message(history)
    assert should_retry_private_reply_omission(
        _ANALYSIS_ONLY,
        history,
        is_group_chat=False,
        focus_active=False,
    )


def test_private_analysis_after_reply_does_not_retry() -> None:
    """已经调用过 reply 的私聊，后面的纯分析轮应结束，而不是再逼一次回复。"""

    history: list[object] = [_user_message("你好")]
    history.extend(
        _model_turn(
            "先查画像。",
            [ToolCall(call_id="call-profile", func_name="query_person_profile", args={})],
        )
    )
    history.extend(
        _model_turn(
            "可以回一句。",
            [
                ToolCall(
                    call_id="call-reply",
                    func_name="reply",
                    args={"msg_id": "synthetic-user", "reply_reference": "问候"},
                )
            ],
        )
    )
    history.append(_user_message("已发出", source_kind="guided_reply"))

    assert not has_unanswered_direct_user_message(history)
    assert not should_retry_private_reply_omission(
        _ANALYSIS_ONLY,
        history,
        is_group_chat=False,
        focus_active=False,
    )


def test_wait_after_user_message_is_not_an_omission() -> None:
    history: list[object] = [_user_message("我先去忙了")]
    history.extend(_model_turn("先等等。", [ToolCall(call_id="call-wait", func_name="wait", args={"seconds": 30})]))

    assert not has_unanswered_direct_user_message(history)
    assert not should_retry_private_reply_omission(
        _ANALYSIS_ONLY,
        history,
        is_group_chat=False,
        focus_active=False,
    )


def test_group_and_focus_do_not_use_private_omission_retry() -> None:
    history = [_user_message("大家好")]

    assert not should_retry_private_reply_omission(
        _ANALYSIS_ONLY,
        history,
        is_group_chat=True,
        focus_active=False,
    )
    assert not should_retry_private_reply_omission(
        _ANALYSIS_ONLY,
        history,
        is_group_chat=False,
        focus_active=True,
    )


def test_bot_echo_is_not_an_unanswered_user_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.chat.utils.utils.is_bot_self", lambda platform, user_id: user_id == "bot-self")
    echo = _user_message(
        "机器人刚发出的内容",
        original_message=SimpleNamespace(
            platform="qq",
            message_info=SimpleNamespace(user_info=SimpleNamespace(user_id="bot-self")),
        ),
    )

    assert not has_unanswered_direct_user_message([echo])
    assert not should_retry_private_reply_omission(
        _ANALYSIS_ONLY,
        [echo],
        is_group_chat=False,
        focus_active=False,
    )


def test_condenser_only_targets_long_successful_query_results() -> None:
    long_result = "x" * global_config.experimental.tool_result_condense_threshold

    assert should_condense_tool_result("maibot_release_info", long_result, success=True)
    assert not should_condense_tool_result("maibot_release_info", long_result[:-1], success=True)
    assert not should_condense_tool_result("maibot_release_info", long_result, success=False)
    assert not should_condense_tool_result("view_forward_message", long_result, success=True)


def test_condense_task_is_self_contained() -> None:
    tool_call = ToolCall(call_id="call-1", func_name="maibot_plugin_list", args={"keyword": "pixiv"})

    task_text = build_tool_result_condense_task(tool_call, "插件A\n插件B", "想找一个 pixiv 相关的插件推荐给群友")

    assert "工具名称：maibot_plugin_list" in task_text
    assert '"keyword": "pixiv"' in task_text
    assert "想找一个 pixiv 相关的插件推荐给群友" in task_text
    assert "<原始结果>\n插件A\n插件B\n</原始结果>" in task_text


def test_condenser_threshold_follows_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config.experimental, "tool_result_condense_threshold", 800)

    assert should_condense_tool_result("maibot_release_info", "x" * 800, success=True)
    assert not should_condense_tool_result("maibot_release_info", "x" * 799, success=True)
