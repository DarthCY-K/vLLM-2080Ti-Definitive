# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tool_parsers.qwen3xml_tool_parser import StreamingXMLToolCallParser


def _deltas(parser, chunks):
    return [parser.parse_single_streaming_chunks(chunk) for chunk in chunks]


def _tool_calls(deltas):
    return [tool_call for delta in deltas for tool_call in delta.tool_calls]


def test_function_name_is_only_emitted_in_first_delta():
    parser = StreamingXMLToolCallParser()
    deltas = _deltas(
        parser,
        [
            "<tool_call>",
            "<function=search_weather>",
            "<parameter=city>Shanghai</parameter>",
            "</function>",
            "</tool_call>",
        ],
    )

    calls = _tool_calls(deltas)
    assert [call.function.name for call in calls if call.function and call.function.name] == [
        "search_weather"
    ]
