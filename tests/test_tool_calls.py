"""Tests for tool call parsing."""
import pytest

from app.parsing.tool_calls import (
    parse_assistant_json,
    parse_continue_system_tool_plaintext,
    openai_tool_call,
)


def test_parse_assistant_json_valid():
    """Test parsing valid assistant JSON."""
    json_str = '{"assistant_message": "Hello", "tool_calls": [{"name": "read_file", "arguments": {"filepath": "test.py"}}]}'
    content, tool_calls = parse_assistant_json(json_str, [])
    assert content == "Hello"
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read_file"


def test_parse_assistant_json_no_tool_calls():
    """Test parsing assistant JSON without tool calls."""
    json_str = '{"assistant_message": "Hello", "tool_calls": []}'
    content, tool_calls = parse_assistant_json(json_str, [])
    assert content == "Hello"
    assert len(tool_calls) == 0


def test_parse_continue_system_tool_plaintext():
    """Test parsing Continue system tool plaintext format."""
    text = 'TOOL_NAME: read_file\nBEGIN_ARG: filepath\ntest.py\nEND_ARG'
    content, tool_calls = parse_continue_system_tool_plaintext(text, [])
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "read_file"


def test_openai_tool_call():
    """Test creating OpenAI tool call format."""
    tc = openai_tool_call("read_file", {"filepath": "test.py"})
    assert tc["function"]["name"] == "read_file"
    assert tc["function"]["arguments"]["filepath"] == "test.py"
