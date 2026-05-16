"""Tool call validation, argument fixing, and format conversion."""
import json
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.config import log
from app.parsing.json_repair import repair_json_string
from app.tools.continue_defaults import TOOL_ARG_ALIASES, TOOL_NAME_ALIASES


def validate_and_fix_tool_calls(
    tool_calls: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """Validate tool_calls against available tools: fix names, fill missing required args, drop invalid."""
    if not tools or not tool_calls:
        return tool_calls

    tool_map: Dict[str, Dict[str, Any]] = {}
    for td in tools:
        fn = td.get("function") or {}
        name = fn.get("name")
        if name:
            tool_map[name] = td

    if not tool_map:
        return tool_calls

    fixed: List[Dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""

        # ── Name resolution: alias table → case-insensitive → substring ──
        if name not in tool_map:
            matched = None

            # 1. Alias table (covers edit_file→edit_existing_file, etc.)
            alias = TOOL_NAME_ALIASES.get(name)
            if alias and alias in tool_map:
                matched = alias

            # 2. Case-insensitive exact match
            if not matched:
                name_lower = name.lower()
                for tname in tool_map:
                    if tname.lower() == name_lower:
                        matched = tname
                        break

            # 3. Case-insensitive alias lookup
            if not matched:
                name_lower = name.lower()
                for alias_key, alias_val in TOOL_NAME_ALIASES.items():
                    if alias_key.lower() == name_lower and alias_val in tool_map:
                        matched = alias_val
                        break

            # 4. Substring match (fallback)
            if not matched:
                name_lower = name.lower()
                for tname in tool_map:
                    if name_lower in tname.lower() or tname.lower() in name_lower:
                        matched = tname
                        break

            if matched:
                log.info("validate_tool_calls: fixed name %r -> %r", name, matched)
                fn["name"] = matched
                name = matched
            else:
                log.warning("validate_tool_calls: unknown tool %r, skipping", name)
                continue

        # Parse and fix arguments
        raw_args = fn.get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                repaired = repair_json_string(raw_args)
                try:
                    args = json.loads(repaired) if repaired.strip() else {}
                except json.JSONDecodeError:
                    args = {}
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}

        # Fix argument names via alias table (path→filepath, content→contents, etc.)
        arg_aliases = TOOL_ARG_ALIASES.get(name, {})
        if arg_aliases:
            renamed_args: Dict[str, Any] = {}
            for ak, av in args.items():
                canonical = arg_aliases.get(ak)
                if canonical and canonical not in args and canonical not in renamed_args:
                    log.info("validate_tool_calls: fixed arg %r -> %r for %s", ak, canonical, name)
                    renamed_args[canonical] = av
                else:
                    renamed_args[ak] = av
            args = renamed_args

        # Strip wrapping quotes and JSON-unescape string argument values.
        # Models often output BEGIN_ARG values as JSON-escaped strings wrapped in "..."
        for ak in list(args.keys()):
            av = args[ak]
            if not isinstance(av, str):
                continue
            # If wrapped in matching quotes, try JSON decode first (handles \\n, \\", etc.)
            if len(av) >= 2 and av[0] == '"' and av[-1] == '"':
                try:
                    decoded = json.loads(av)
                    if isinstance(decoded, str):
                        args[ak] = decoded
                        continue
                except (json.JSONDecodeError, ValueError):
                    # Fall back to simple strip
                    args[ak] = av[1:-1]
                    continue
            if len(av) >= 2 and av[0] == "'" and av[-1] == "'":
                args[ak] = av[1:-1]
                continue
            # Even without outer quotes, try to unescape \\n etc. if present
            if "\\n" in av or "\\t" in av or '\\"' in av:
                try:
                    decoded = json.loads(f'"{av}"')
                    if isinstance(decoded, str):
                        args[ak] = decoded
                except (json.JSONDecodeError, ValueError):
                    pass

        # Special case: model sends old_string/new_string for edit_existing_file
        # but Continue expects a single 'changes' field.
        if name == "edit_existing_file" and "changes" not in args:
            old_s = args.pop("old_string", None)
            new_s = args.pop("new_string", None)
            if old_s is not None and new_s is not None:
                args["changes"] = f"Replace:\n```\n{old_s}\n```\nWith:\n```\n{new_s}\n```"
                log.info("validate_tool_calls: merged old_string/new_string -> changes for edit_existing_file")
            elif new_s is not None:
                args["changes"] = new_s
            elif old_s is not None:
                args["changes"] = old_s

        # Special case: Continue's ls rejects bare "." — fix to "./"
        if name == "ls" and args.get("dirPath") in (".", ""):
            args["dirPath"] = "./"
            log.info("validate_tool_calls: fixed dirPath '.' -> './' for ls")

        # Convert create_new_file → edit_existing_file to avoid "already exists" errors.
        # Continue's edit tool handles both creation and editing.
        if name == "create_new_file":
            contents = args.pop("contents", "")
            args["changes"] = contents
            name = "edit_existing_file"
            tc["function"]["name"] = name
            log.info("validate_tool_calls: converted create_new_file -> edit_existing_file")

        # Normalize filepath: strip ./ prefix
        # Continue expects bare relative paths like "yandex_parser.py", not "./yandex_parser.py"
        if name in ("read_file", "edit_existing_file", "create_new_file"):
            fp = args.get("filepath", "")
            if fp.startswith("./"):
                args["filepath"] = fp[2:]
                log.info("validate_tool_calls: stripped ./ from filepath %r", fp)

        # Fill missing required args with defaults
        tool_def = tool_map[name]
        params = (tool_def.get("function") or {}).get("parameters") or {}
        required = params.get("required") or []
        props = params.get("properties") or {}

        for req_arg in required:
            if req_arg not in args:
                spec = props.get(req_arg, {})
                if not isinstance(spec, dict):
                    spec = {}
                t = spec.get("type", "string")
                if t == "string":
                    args[req_arg] = spec.get("default", "")
                elif t == "integer":
                    args[req_arg] = int(spec.get("default", 0))
                elif t == "number":
                    args[req_arg] = float(spec.get("default", 0))
                elif t == "boolean":
                    args[req_arg] = bool(spec.get("default", False))
                elif t == "array":
                    args[req_arg] = spec.get("default", [])
                elif t == "object":
                    args[req_arg] = spec.get("default", {})
                log.info("validate_tool_calls: filled missing arg %s for %s", req_arg, name)

        # Remove args not in schema
        if props:
            valid_keys = set(props.keys())
            extra_keys = set(args.keys()) - valid_keys
            for ek in extra_keys:
                ek_lower = ek.lower()
                mapped = False
                for vk in valid_keys:
                    if vk.lower() == ek_lower and vk not in args:
                        args[vk] = args.pop(ek)
                        mapped = True
                        break
                if not mapped:
                    del args[ek]

        fn["arguments"] = json.dumps(args, ensure_ascii=False)
        fixed.append(tc)

    return fixed


def ollama_tool_calls_from_openai(
    tool_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments")
        if isinstance(raw_args, str):
            try:
                args_obj = json.loads(raw_args) if raw_args.strip() else {}
            except json.JSONDecodeError:
                args_obj = {}
        elif isinstance(raw_args, dict):
            args_obj = raw_args
        else:
            args_obj = {}
        out.append({"function": {"name": name, "arguments": args_obj}})
    return out


def ollama_done_metadata() -> Dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return {
        "created_at": now,
        "done": True,
        "done_reason": "stop",
        "total_duration": 0,
        "load_duration": 0,
        "prompt_eval_count": 0,
        "prompt_eval_duration": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }
