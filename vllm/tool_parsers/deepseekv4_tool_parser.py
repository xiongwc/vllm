# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re

from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import Tool
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser


class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """
    DeepSeek V4 DSML tool parser.

    V4 keeps the V3.2 DSML invoke/parameter grammar, but wraps tool calls in
    ``<｜DSML｜tool_calls>`` instead of ``<｜DSML｜function_calls>``.
    """

    tool_call_start_token: str = "<｜DSML｜tool_calls>"
    tool_call_end_token: str = "</｜DSML｜tool_calls>"

    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)

        self.tool_call_start_token = "<｜DSML｜tool_calls>"
        self.tool_call_end_token = "</｜DSML｜tool_calls>"
        self.tool_call_complete_regex = re.compile(
            re.escape(self.tool_call_start_token)
            + r"(.*?)"
            + re.escape(self.tool_call_end_token),
            re.DOTALL,
        )
