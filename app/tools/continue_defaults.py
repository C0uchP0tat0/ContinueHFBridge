"""Default Continue IDE built-in tool definitions.

When Continue doesn't send its tool list in the request (tools=None),
we inject these so the model knows the exact tool names and schemas.

Tool names are from: https://docs.continue.dev/ide-extensions/agent/how-it-works
"""

# ── Alias map: model-guessed name → correct Continue name ─────────────
# GLM often invents close-but-wrong names; this table fixes them.
TOOL_NAME_ALIASES: dict[str, str] = {
    # edit
    "edit_file": "edit_existing_file",
    "editFile": "edit_existing_file",
    "modify_file": "edit_existing_file",
    "update_file": "edit_existing_file",
    "replace_in_file": "edit_existing_file",
    "patch_file": "edit_existing_file",
    # create / write
    "write_file": "create_new_file",
    "create_file": "create_new_file",
    "createFile": "create_new_file",
    "new_file": "create_new_file",
    "writeFile": "create_new_file",
    "save_file": "create_new_file",
    # read
    "open_file": "read_file",
    "view_file": "read_file",
    "readFile": "read_file",
    "cat": "read_file",
    "get_file": "read_file",
    "show_file": "read_file",
    # list directory
    "list_directory": "ls",
    "list_dir": "ls",
    "list_files": "ls",
    "listDir": "ls",
    "listDirectory": "ls",
    "dir": "ls",
    # search
    "search": "grep_search",
    "search_files": "grep_search",
    "find": "grep_search",
    "grep": "grep_search",
    "search_code": "grep_search",
    "find_in_files": "grep_search",
    # terminal
    "run_command": "run_terminal_command",
    "shell": "run_terminal_command",
    "exec": "run_terminal_command",
    "execute": "run_terminal_command",
    "terminal": "run_terminal_command",
    "runCommand": "run_terminal_command",
    "bash": "run_terminal_command",
    # repo map / diff
    "repo_map": "view_repo_map",
    "repoMap": "view_repo_map",
    "diff": "view_diff",
    "show_diff": "view_diff",
}

# ── Argument name aliases per tool ─────────────────────────────────────
# GLM uses "path" but Continue expects "filepath", etc.
TOOL_ARG_ALIASES: dict[str, dict[str, str]] = {
    "read_file": {"path": "filepath", "file": "filepath", "file_path": "filepath", "filename": "filepath"},
    "edit_existing_file": {"path": "filepath", "file": "filepath", "file_path": "filepath", "filename": "filepath", "content": "changes", "new_content": "changes", "code": "changes"},
    "create_new_file": {"path": "filepath", "file": "filepath", "file_path": "filepath", "filename": "filepath", "content": "contents", "body": "contents", "text": "contents"},
    "ls": {"path": "dirPath", "dir": "dirPath", "directory": "dirPath", "dir_path": "dirPath"},
    "grep_search": {"pattern": "query", "search": "query", "term": "query", "path": "dirPath", "dir": "dirPath"},
    "run_terminal_command": {"cmd": "command", "shell_command": "command", "exec": "command"},
}

# Agent Mode tools (read + write)
DEFAULT_CONTINUE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path to the file, e.g. 'driver_setup.py' or 'src/main.py'.",
                    }
                },
                "required": ["filepath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_existing_file",
            "description": "Edit an existing file. Provide the filepath and the changes to make. "
            "The 'changes' argument should contain the new full content of the file or a clear description of what to change.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path to the file to edit, e.g. 'utils.py' or 'src/utils.py'.",
                    },
                    "changes": {
                        "type": "string",
                        "description": "The changes to apply to the file — new content or edit instructions.",
                    },
                },
                "required": ["filepath", "changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_new_file",
            "description": "Create a NEW file that does NOT exist yet. Use ls first to check!",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Relative path for the new file, e.g. 'new_module.py' or 'src/new_module.py'.",
                    },
                    "contents": {
                        "type": "string",
                        "description": "Contents to write to the new file.",
                    },
                },
                "required": ["filepath", "contents"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_terminal_command",
            "description": "Run a shell command in the terminal at the workspace root. Use for installing packages, running scripts, builds, tests, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List the contents of a directory. Returns file and subdirectory names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dirPath": {
                        "type": "string",
                        "description": "Path to the directory to list. Use './' for the workspace root, or a relative path like 'src/' or an absolute path.",
                    }
                },
                "required": ["dirPath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a pattern in files across the project using grep. Returns matching lines with file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search pattern (regex or plain text).",
                    },
                    "dirPath": {
                        "type": "string",
                        "description": "Directory to search in, relative to workspace root. Defaults to '.'.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_repo_map",
            "description": "View the repository structure — a high-level map of all files and directories in the project.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_diff",
            "description": "View all changes (diffs) made so far in the current session.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
