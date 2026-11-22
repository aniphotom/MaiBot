"""Maisaka 推理引擎测试。"""

from datetime import datetime
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, Mock

import pytest

from src.chat.heart_flow.heartFC_utils import CycleDetail
from src.common.data_models.llm_service_data_models import LLMResponseResult
from src.common.data_models.message_component_data_model import MessageSequence, TextComponent
from src.config.config import global_config
from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation
from src.llm_models.model_client.base_client import GenerationAttempt, GenerationTrace
from src.llm_models.payload_content.context_item import (
    ContextItemMeta,
    ProviderActivityItem,
)
from src.llm_models.payload_content.native_tool import NativeToolCallSummary
from src.llm_models.payload_content.tool_option import ToolCall
from src.maisaka.chat_loop_service import ChatResponse, MaisakaChatLoopService
from src.maisaka.context.messages import ReferenceMessageType, SessionBackedMessage
from src.maisaka.context.usage import (
    SECTION_INSTANT_NOTICE,
    SECTION_MESSAGES,
    SECTION_SYSTEM_PROMPT,
    ContextSectionUsage,
)
from src.maisaka.display.prompt_cli_renderer import PromptCLIVisualizer
from src.maisaka.focus import focus_mode_manager
from src.maisaka.mode_policy import is_idle_cycle_reason
from src.maisaka.monitor.events import (
    _serialize_planner_block,
    _serialize_request_block,
    _serialize_tool_results,
)
from src.maisaka.reasoning_engine import (
    PLANNER_UNSENT_REPLY_HINT,
    STOP_AFTER_EXECUTION_PAUSE_REASON,
    CycleRuntimeState,
    MaisakaReasoningEngine,
)


class _ToolRegistryStub:
    """按顺序返回预设工具结果。"""

    def __init__(self, results: list[ToolExecutionResult]) -> None:
        self._results = list(results)
        self.invoked_tool_names: list[str] = []

    async def list_tools(self, context: object) -> list[object]:
        del context
        return []

    async def invoke(self, invocation: ToolInvocation, context: ToolExecutionContext) -> ToolExecutionResult:
        del context
        self.invoked_tool_names.append(invocation.tool_name)
        return self._results.pop(0)


def _build_tool_engine(results: list[ToolExecutionResult]) -> tuple[MaisakaReasoningEngine, SimpleNamespace]:
    """构造仅用于工具批次测试的推理引擎。"""

    runtime = SimpleNamespace(
        _tool_registry=_ToolRegistryStub(results),
        session_id="session-test",
        chat_stream=SimpleNamespace(
            is_group_session=True,
            group_id="group-test",
            user_id="",
            platform="test",
        ),
        is_action_tool_currently_available=lambda tool_name: True,
        expand_message_id_aliases=lambda arguments: arguments,
        _update_stage_status=lambda *args, **kwargs: None,
        _reset_consecutive_wait_count=Mock(),
        _end_planner_continuation=Mock(),
        _enter_stop_state=Mock(),
        _monitor_run_id="synthetic-run",
        log_prefix="[test]",
    )
    engine = MaisakaReasoningEngine(runtime)
    engine._record_tool_execution_effects = AsyncMock()  # type: ignore[method-assign]
    engine._append_tool_execution_result = lambda *args, **kwargs: None  # type: ignore[method-assign]
    engine._append_tool_post_history_messages = lambda messages: None  # type: ignore[method-assign]
    return engine, runtime


@pytest.mark.asyncio
async def test_successful_stop_request_finishes_after_full_tool_batch() -> None:
    engine, runtime = _build_tool_engine(
        [
            ToolExecutionResult(
                tool_name="terminal_tool",
                success=True,
                content="已完成",
                stop_after_execution=True,
            ),
            ToolExecutionResult(
                tool_name="second_terminal_tool",
                success=True,
                content="第二个终止工具完成",
                stop_after_execution=True,
            ),
            ToolExecutionResult(tool_name="following_tool", success=True, content="后续工具完成"),
        ]
    )

    tool_state = CycleRuntimeState()
    tool_state.response = _build_chat_response("测试思考", "")
    should_pause, pause_reason, _, monitor_results = await engine._handle_tool_calls(
        [
            ToolCall(call_id="call-1", func_name="terminal_tool"),
            ToolCall(call_id="call-2", func_name="second_terminal_tool"),
            ToolCall(call_id="call-3", func_name="following_tool"),
        ],
        "测试思考",
        CycleDetail(cycle_id=1),
        tool_state,
    )

    assert should_pause is True
    assert pause_reason == STOP_AFTER_EXECUTION_PAUSE_REASON
    assert runtime._tool_registry.invoked_tool_names == [
        "terminal_tool",
        "second_terminal_tool",
        "following_tool",
    ]
    assert [result["stop_after_execution"] for result in monitor_results] == [True, True, False]
    runtime._end_planner_continuation.assert_called_once_with()
    assert runtime._reset_consecutive_wait_count.call_args_list[-1].args == ("tool_stop_after_execution",)
    runtime._enter_stop_state.assert_called_once_with()

    cycle_end = engine._cycle_end_for_pause_tool(pause_reason)
    assert cycle_end.reason == "tool_stop_after_execution"
    assert is_idle_cycle_reason(cycle_end.reason) is True


@pytest.mark.asyncio
async def test_failed_stop_request_keeps_planner_running() -> None:
    engine, runtime = _build_tool_engine(
        [
            ToolExecutionResult(
                tool_name="terminal_tool",
                success=False,
                error_message="执行失败",
                stop_after_execution=True,
            ),
            ToolExecutionResult(tool_name="following_tool", success=True, content="后续工具完成"),
        ]
    )

    tool_state = CycleRuntimeState()
    tool_state.response = _build_chat_response("测试思考", "")
    should_pause, pause_reason, _, _ = await engine._handle_tool_calls(
        [
            ToolCall(call_id="call-1", func_name="terminal_tool"),
            ToolCall(call_id="call-2", func_name="following_tool"),
        ],
        "测试思考",
        CycleDetail(cycle_id=1),
        tool_state,
    )

    assert should_pause is False
    assert pause_reason == ""
    assert runtime._tool_registry.invoked_tool_names == ["terminal_tool", "following_tool"]
    runtime._end_planner_continuation.assert_not_called()
    runtime._enter_stop_state.assert_not_called()


def _build_chat_response(content: Optional[str], reasoning: str) -> ChatResponse:
    """构造仅包含 Planner 思考字段的响应。"""

    result = LLMResponseResult.from_portable_output(
        response=content or "",
        reasoning=reasoning,
    )
    return ChatResponse(
        output_items=result.output_items,
        request_messages=[],
        selected_history_count=0,
        tool_count=0,
        prompt_tokens=0,
        built_message_count=0,
        completion_tokens=0,
        total_tokens=0,
    )


@pytest.mark.parametrize(
    ("content", "reasoning", "expected"),
    [
        (" Planner 工具正文 ", " Provider 原生推理 ", "Planner 工具正文"),
        ("", " Provider 原生推理 ", ""),
        (None, " Provider 原生推理 ", ""),
        ("   ", "   ", ""),
    ],
)
def test_planner_content_does_not_fall_back_to_reasoning(
    content: Optional[str],
    reasoning: str,
    expected: str,
) -> None:
    response = _build_chat_response(content, reasoning)

    result = MaisakaReasoningEngine._get_planner_content(response)

    assert result == expected


def test_native_tool_summary_is_serialized_without_provider_state() -> None:
    summary = NativeToolCallSummary(
        tool_type="web_search",
        call_id="ws_test",
        status="completed",
        action_type="search",
        details=["查询：Responses API"],
        source_count=2,
    )

    block = _serialize_planner_block("完成", [], [summary], 10, 5, 15, 100.0)

    assert block is not None
    assert block["native_tool_calls"] == [
        {
            "tool_type": "web_search",
            "call_id": "ws_test",
            "status": "completed",
            "action_type": "search",
            "details": ["查询：Responses API"],
            "source_count": 2,
        }
    ]
    assert "provider_state" not in block


def test_tool_stop_request_is_serialized_for_monitor() -> None:
    tools = _serialize_tool_results(
        [
            {
                "tool_call_id": "call-test",
                "tool_name": "test_tool",
                "success": True,
                "stop_after_execution": True,
                "started_at": 100.25,
                "ended_at": 102.75,
            }
        ]
    )

    assert tools[0]["stop_after_execution"] is True
    assert tools[0]["started_at"] == 100.25
    assert tools[0]["ended_at"] == 102.75


def test_cache_tokens_and_context_sections_are_serialized_for_monitor() -> None:
    planner_block = _serialize_planner_block("完成", [], [], 100, 5, 105, 100.0, "", 80, 20)
    request_block = _serialize_request_block(
        [{"role": "system", "content": "系统提示词"}],
        1,
        2,
        [ContextSectionUsage("messages", 12, 1)],
    )

    assert planner_block is not None
    assert planner_block["prompt_cache_hit_tokens"] == 80
    assert planner_block["prompt_cache_miss_tokens"] == 20
    assert request_block is not None
    assert request_block["context_sections"] == [{"key": "messages", "chars": 12, "count": 1}]


@pytest.mark.asyncio
async def test_chat_loop_reports_context_sections_and_cache_tokens(monkeypatch) -> None:
    """chat_loop_step 应回报提示词分段用量与缓存命中 token。"""

    class FakeLLMClient:
        async def generate_response_with_context(self, context_factory, options) -> LLMResponseResult:
            del options
            context_factory(None)
            result = LLMResponseResult.from_portable_output(response="好的", model_name="test-model")
            result.prompt_tokens = 1000
            result.completion_tokens = 20
            result.total_tokens = 1020
            result.prompt_cache_hit_tokens = 800
            result.prompt_cache_miss_tokens = 200
            return result

    class PassthroughRuntimeManager:
        async def invoke_hook(self, hook_name: str, **kwargs: object) -> SimpleNamespace:
            del hook_name
            return SimpleNamespace(kwargs=kwargs)

    service = MaisakaChatLoopService(chat_system_prompt="测试系统提示词")
    monkeypatch.setattr(service, "_get_llm_chat_client", lambda request_kind: FakeLLMClient())
    monkeypatch.setattr(
        MaisakaChatLoopService,
        "_get_runtime_manager",
        staticmethod(lambda: PassthroughRuntimeManager()),
    )
    monkeypatch.setattr(
        PromptCLIVisualizer,
        "build_prompt_section_result",
        staticmethod(
            lambda *args, **kwargs: SimpleNamespace(
                panel=None,
                preview_access=SimpleNamespace(preview_web_uri=""),
            )
        ),
    )

    response = await service.chat_loop_step([], tool_definitions=[])

    sections = {section.key: section.chars for section in response.context_sections}
    # 本轮没有历史消息：系统提示词单独成段，末尾的当前时间与提醒归入即时提示
    assert sections[SECTION_SYSTEM_PROMPT] == len("测试系统提示词")
    assert sections[SECTION_INSTANT_NOTICE] > 0
    assert SECTION_MESSAGES not in sections
    assert response.prompt_cache_hit_tokens == 800
    assert response.prompt_cache_miss_tokens == 200


@pytest.mark.asyncio
async def test_chat_loop_keeps_reasoning_separate_from_content(monkeypatch) -> None:
    """Provider 仅返回 reasoning 时，不应将其回填为 Planner 正文。"""

    class FakeLLMClient:
        async def generate_response_with_context(self, context_factory, options) -> LLMResponseResult:
            del context_factory, options
            result = LLMResponseResult.from_portable_output(
                reasoning="Provider 原生推理",
                model_name="test-model",
            )
            logical_turn_id = result.output_items[0].meta.logical_turn_id
            assert logical_turn_id is not None
            result.output_items = (
                *result.output_items,
                ProviderActivityItem(
                    meta=ContextItemMeta.create(
                        logical_turn_id=logical_turn_id,
                    ),
                    provider_type="web_search",
                    call_id="ws_test",
                    status="completed",
                    action_type="search",
                    details=("查询：Responses API",),
                ),
            )
            result.generation_trace = GenerationTrace(
                provider="test-provider",
                endpoint="responses",
                model="test-model",
                response_id="resp_test",
                status="completed",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                prompt_cache_hit_tokens=0,
                prompt_cache_miss_tokens=0,
                output_item_ids=tuple(item.meta.item_id for item in result.output_items),
            )
            result.provider_response = {
                "id": "resp_test",
                "status": "completed",
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_test",
                        "summary": [{"type": "summary_text", "text": "Provider 原生推理"}],
                    },
                    {
                        "type": "web_search_call",
                        "id": "ws_test",
                        "status": "completed",
                        "action": {"type": "search", "queries": ["Responses API"]},
                    },
                ],
            }
            result.generation_attempts = (
                GenerationAttempt(
                    attempt_id="planner-attempt-1",
                    workflow_purpose="planner",
                    workflow_attempt=1,
                    provider_attempt=1,
                    model_attempt=1,
                    status="succeeded",
                    started_at="2026-08-05T00:00:00.000",
                    duration_ms=1.0,
                    provider="test-provider",
                    endpoint="responses",
                    model="test-model",
                    client_type="openai_responses",
                    operation="response",
                    wire_protocol="responses",
                ),
            )
            return result

    class PassthroughRuntimeManager:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        async def invoke_hook(self, hook_name: str, **kwargs: object) -> SimpleNamespace:
            self.calls.append((hook_name, kwargs))
            return SimpleNamespace(kwargs=kwargs)

    runtime_manager = PassthroughRuntimeManager()
    service = MaisakaChatLoopService(chat_system_prompt="测试系统提示词")
    monkeypatch.setattr(service, "_get_llm_chat_client", lambda request_kind: FakeLLMClient())
    monkeypatch.setattr(
        MaisakaChatLoopService,
        "_get_runtime_manager",
        staticmethod(lambda: runtime_manager),
    )
    prompt_preview_kwargs: dict[str, object] = {}

    def build_prompt_section_result(*args: object, **kwargs: object) -> SimpleNamespace:
        del args
        prompt_preview_kwargs.update(kwargs)
        return SimpleNamespace(
            panel=None,
            preview_access=SimpleNamespace(preview_web_uri=""),
        )

    monkeypatch.setattr(
        PromptCLIVisualizer,
        "build_prompt_section_result",
        build_prompt_section_result,
    )

    response = await service.chat_loop_step([])

    after_response_kwargs = next(
        kwargs for hook_name, kwargs in runtime_manager.calls if hook_name == "maisaka.planner.after_response"
    )
    assert len(after_response_kwargs["output_items"]) == 2
    assert response.content is None
    assert all(message.content == "" for message in response.raw_messages)
    assert response.reasoning == "Provider 原生推理"
    assert response.native_tool_calls[0].call_id == "ws_test"
    assert all(not hasattr(message, "native_tool_calls") for message in response.raw_messages)
    preview_output_items = prompt_preview_kwargs["output_items"]
    assert isinstance(preview_output_items, tuple)
    assert len(preview_output_items) == 2
    assert preview_output_items[0].__class__.__name__ == "ReasoningItem"
    assert preview_output_items[1].__class__.__name__ == "ProviderActivityItem"
    generation_attempts = prompt_preview_kwargs["generation_attempts"]
    assert isinstance(generation_attempts, tuple)
    assert generation_attempts[0].attempt_id == "planner-attempt-1"
    assert not hasattr(generation_attempts[0], "wire_response")


_PRIVATE_ANALYSIS_ONLY = (
    "本轮不需要工具调用。\n对方刚发送了私聊消息。现有上下文已经足够理解场面，没有额外资料需要查找。"
)


def _synthetic_user_message(text: str) -> SessionBackedMessage:
    return SessionBackedMessage(
        raw_message=MessageSequence([TextComponent(text)]),
        visible_text=text,
        timestamp=datetime(2026, 11, 22, 12, 0, 0),
        message_id="synthetic-user",
        source_kind="user",
    )


def _mock_completion(content: str, tool_calls: list[ToolCall] | None = None) -> ChatResponse:
    """用本地构造的模型输出模拟 Planner，不访问供应商。"""

    result = LLMResponseResult.from_portable_output(
        response=content,
        reasoning="模拟推理，不会外发",
        tool_calls=tool_calls,
    )
    return ChatResponse(
        output_items=result.output_items,
        request_messages=[],
        selected_history_count=1,
        tool_count=len(tool_calls or []),
        prompt_tokens=10,
        built_message_count=2,
        completion_tokens=8,
        total_tokens=18,
        model_name="mock-planner",
    )


def _build_decision_engine(
    history: list[object],
    *,
    is_group: bool,
) -> tuple[MaisakaReasoningEngine, SimpleNamespace]:
    runtime = SimpleNamespace(
        _chat_history=list(history),
        session_id="synthetic-session",
        chat_stream=SimpleNamespace(is_group_session=is_group),
        log_prefix="[synthetic]",
        _end_planner_continuation=Mock(),
        _reset_consecutive_wait_count=Mock(),
        _enter_stop_state=Mock(),
        _monitor_run_id="synthetic-run",
    )
    engine = MaisakaReasoningEngine(runtime)
    engine._emit_planner_progress = AsyncMock()  # type: ignore[method-assign]
    engine._handle_tool_calls = AsyncMock(return_value=(False, "", [], []))  # type: ignore[method-assign]
    return engine, runtime


async def _decide(
    engine: MaisakaReasoningEngine,
    response: ChatResponse,
    *,
    planner_no_tool_count: int,
) -> tuple[int, bool, CycleRuntimeState]:
    state = CycleRuntimeState()
    state.response = response
    count, should_end = await engine._handle_planner_response_actions(
        response=response,
        cycle_detail=CycleDetail(cycle_id=planner_no_tool_count + 1),
        state=state,
        planner_no_tool_count=planner_no_tool_count,
        planner_extra_lines=state.planner_extra_lines,
    )
    return count, should_end, state


def _hint_count(history: list[object]) -> int:
    return sum(
        1
        for message in history
        if getattr(message, "reference_type", None) == ReferenceMessageType.PLANNER_TOOL_HINT
    )


@pytest.mark.asyncio
async def test_private_analysis_omission_retries_then_invokes_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config.experimental, "planner_unsent_reply_retry", False)
    monkeypatch.setattr(focus_mode_manager, "is_enabled_for_session", lambda session_id, is_group_chat=None: False)
    engine, runtime = _build_decision_engine(
        [_synthetic_user_message("你好，这是一条回复测试，收到请回复一句话。")],
        is_group=False,
    )

    count, should_end, state = await _decide(engine, _mock_completion(_PRIVATE_ANALYSIS_ONLY), planner_no_tool_count=0)

    assert count == 1
    assert should_end is False
    assert state.cycle_end.reason == "planner_no_tool_retry"
    engine._handle_tool_calls.assert_not_awaited()
    runtime._enter_stop_state.assert_not_called()
    hint = runtime._chat_history[-1]
    assert hint.content == PLANNER_UNSENT_REPLY_HINT
    assert _PRIVATE_ANALYSIS_ONLY not in hint.content
    assert _hint_count(runtime._chat_history) == 1
    assert not any(getattr(message, "source_kind", "") == "guided_reply" for message in runtime._chat_history)

    reply_call = ToolCall(
        call_id="call-reply",
        func_name="reply",
        args={"msg_id": "synthetic-user", "reply_reference": "回一句问候"},
    )
    count, should_end, state = await _decide(
        engine,
        _mock_completion("调用 reply 回一句问候。", [reply_call]),
        planner_no_tool_count=count,
    )

    assert should_end is False
    assert count == 0
    assert state.cycle_end.reason == "tool_continue"
    invoked_calls = engine._handle_tool_calls.await_args.args[0]
    assert [tool_call.func_name for tool_call in invoked_calls] == ["reply"]
    assert invoked_calls[0].args == {"msg_id": "synthetic-user", "reply_reference": "回一句问候"}
    assert _PRIVATE_ANALYSIS_ONLY not in str(invoked_calls[0].args)
    passed_thought = engine._handle_tool_calls.await_args.args[1]
    assert passed_thought == "调用 reply 回一句问候。"
    assert _PRIVATE_ANALYSIS_ONLY not in passed_thought


@pytest.mark.asyncio
async def test_private_second_analysis_ends_without_sending_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config.experimental, "planner_unsent_reply_retry", False)
    monkeypatch.setattr(focus_mode_manager, "is_enabled_for_session", lambda session_id, is_group_chat=None: False)
    engine, runtime = _build_decision_engine([_synthetic_user_message("在吗")], is_group=False)

    await _decide(engine, _mock_completion(_PRIVATE_ANALYSIS_ONLY), planner_no_tool_count=0)
    count, should_end, state = await _decide(
        engine,
        _mock_completion(_PRIVATE_ANALYSIS_ONLY),
        planner_no_tool_count=1,
    )

    assert count == 2
    assert should_end is True
    assert state.cycle_end.reason == "planner_no_tool_end"
    assert _hint_count(runtime._chat_history) == 0
    engine._handle_tool_calls.assert_not_awaited()
    runtime._enter_stop_state.assert_called_once_with()


@pytest.mark.asyncio
async def test_query_and_wait_tools_are_not_rewritten_into_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config.experimental, "planner_unsent_reply_retry", False)
    monkeypatch.setattr(focus_mode_manager, "is_enabled_for_session", lambda session_id, is_group_chat=None: False)
    engine, runtime = _build_decision_engine([_synthetic_user_message("帮我回忆一下")], is_group=False)
    query_calls = [
        ToolCall(call_id="call-profile", func_name="query_person_profile", args={"name": "对方"}),
        ToolCall(call_id="call-memory", func_name="query_memory", args={"query": "近况"}),
    ]

    count, should_end, state = await _decide(engine, _mock_completion("先查资料。", query_calls), planner_no_tool_count=0)

    assert should_end is False
    assert count == 0
    assert state.cycle_end.reason == "tool_continue"
    assert [tool_call.func_name for tool_call in engine._handle_tool_calls.await_args.args[0]] == [
        "query_person_profile",
        "query_memory",
    ]
    assert _hint_count(runtime._chat_history) == 0

    engine._handle_tool_calls = AsyncMock(return_value=(True, "wait", [], []))  # type: ignore[method-assign]
    wait_call = ToolCall(call_id="call-wait", func_name="wait", args={"seconds": 20})
    count, should_end, state = await _decide(engine, _mock_completion("先等等。", [wait_call]), planner_no_tool_count=0)

    assert should_end is True
    assert count == 0
    assert state.cycle_end.reason == "tool_pause:wait"
    assert [tool_call.func_name for tool_call in engine._handle_tool_calls.await_args.args[0]] == ["wait"]
    assert _hint_count(runtime._chat_history) == 0


@pytest.mark.asyncio
async def test_group_analysis_without_reply_still_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(global_config.experimental, "planner_unsent_reply_retry", False)
    monkeypatch.setattr(focus_mode_manager, "is_enabled_for_session", lambda session_id, is_group_chat=None: False)
    engine, runtime = _build_decision_engine([_synthetic_user_message("群里随便聊聊")], is_group=True)

    count, should_end, state = await _decide(engine, _mock_completion(_PRIVATE_ANALYSIS_ONLY), planner_no_tool_count=0)

    assert count == 1
    assert should_end is True
    assert state.cycle_end.reason == "planner_no_tool_end"
    assert _hint_count(runtime._chat_history) == 0
    engine._handle_tool_calls.assert_not_awaited()


@pytest.mark.asyncio
async def test_group_explicit_reply_intent_still_follows_experimental_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(focus_mode_manager, "is_enabled_for_session", lambda session_id, is_group_chat=None: False)
    explicit_reply = "决定：调用reply，把这句问候发出去。"
    engine, runtime = _build_decision_engine([_synthetic_user_message("@机器人 你好")], is_group=True)

    monkeypatch.setattr(global_config.experimental, "planner_unsent_reply_retry", False)
    _, should_end, _ = await _decide(engine, _mock_completion(explicit_reply), planner_no_tool_count=0)
    assert should_end is True
    assert _hint_count(runtime._chat_history) == 0

    monkeypatch.setattr(global_config.experimental, "planner_unsent_reply_retry", True)
    engine, runtime = _build_decision_engine([_synthetic_user_message("@机器人 你好")], is_group=True)
    _, should_end, state = await _decide(engine, _mock_completion(explicit_reply), planner_no_tool_count=0)
    assert should_end is False
    assert state.cycle_end.reason == "planner_no_tool_retry"
    assert runtime._chat_history[-1].content == PLANNER_UNSENT_REPLY_HINT
