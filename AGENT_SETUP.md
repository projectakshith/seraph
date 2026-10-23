# Set up Seraph in this coding agent

Use these instructions when a user asks to install or use Seraph. Seraph is a local MCP server for searching local Git repositories. MCP works across many coding agents, but each agent has its own server registration mechanism.

## Install

Requirements: Git, Python 3.12, and `uv` (the recommended Python tool installer).

If this project is published on PyPI:

```sh
python -m pip install seraph-mcp
```

If the user provides a Seraph source checkout instead:

```sh
cd /path/to/seraph
uv tool install --from . seraph-mcp
```

Do not install Seraph into the target project or modify that project's dependencies. The user may choose another install method; preserve their existing environment and report any missing prerequisite clearly.

## Register with the current agent

1. Identify the current agent host and use its documented MCP setup mechanism (built-in setup command, supported CLI, settings UI, or documented config file).
2. Register a **local stdio** MCP server named `seraph` with command `seraph-mcp` and no fixed repository environment variable.
3. If `seraph-mcp` is not on the host's PATH, use the absolute executable path returned by `uv tool dir --bin` plus `/seraph-mcp` (macOS/Linux) or the equivalent path for the OS. Alternatively, register `uvx --from seraph-mcp seraph-mcp` when the host supports `uvx` and package downloads.
4. Follow the host's instructions for reloading MCP servers, then confirm the Seraph tools are available. Never claim setup succeeded until the host confirms the server connected.

Do not assume a Codex, Claude, Cursor, VS Code, or other host config format. If the current host has no MCP support, explain that and give the user the relevant alternative instead of writing a guessed config.

## Use it correctly

- Pass `repo` as the absolute path to the local Git checkout **on every tool call**.
- Use the active workspace path for questions about the current project.
- If the user asks about a different local project, use that project's path for that call; do not silently keep searching the active workspace.
- If the intended repository path is unclear, ask which local checkout they mean.
- Seraph does not upload repository contents. Its SQLite index is created under `<repo>/.seraph/` and is ignored by Git.
- Search tools index the requested Git version on first use. `index_repository` is available when the user wants indexing to happen explicitly first.
- Source indexing supports Python, JavaScript, TypeScript, Go, C, C++, Java, C#, Rust, PHP, Ruby, Swift, Kotlin, Bash, Dart, Lua, SQL, Fortran, and Scala. Unsupported file types are skipped by the code index.
- After a package upgrade, the first request for an already indexed Git version backfills files using newly added language support automatically.
- For code-change tasks, start with `prepare_change_context` (default scope: `worktree`). Use `search_worktree` for code currently on disk, including uncommitted and untracked supported source files, and `change_impact` for statically resolved callers/importers/tests. Use `get_repo_map` with `scope="worktree"` to orient on the current files. These tools are evidence aids; follow returned paths and inspect source before editing.
- Seraph bounds context output by characters, not model tokens. Let the agent decide how much of the returned evidence to use, and expand with `path_prefix` or the focused tools when needed.
- The default setup is local BM25 (the `lite` tier) and requires no API key or model.
- Semantic search is the user's choice, in three tiers: `lite` (keywords, nothing to download), `fast` (about 62 MB local model, instant) and `quality` (BGE-Code-v1, about 6 GB, slow first search). On your first Seraph call, call `retrieval_status`. If `user_has_chosen` is false, ask the user which tier they want, then call `set_retrieval_tier`. Never choose for them, and do not ask again once they have.
- `set_retrieval_tier` with `install=true` pip-installs missing packages for `fast`. For `quality` it returns a terminal command (`seraph setup --tier quality --install`) because the download takes minutes; run it in a terminal only after the user agrees. If the server was registered with `uvx`, register it as `uvx --from "seraph-mcp[fast]" seraph-mcp` instead, since that environment has no pip.
- If a search response includes `retrieval_hint` (the query barely overlaps the repository's vocabulary) or `notice` (a chosen tier could not load and search fell back to keywords), tell the user.
- `quality` needs Transformers 4, so do not combine it with `gemma2`. EmbeddingGemma-2 needs Transformers 5; install `seraph-mcp[gemma2]` for that model in its own environment.

## Check setup

Call `index_repository` with the active workspace path, then call `search_code` with a question grounded in that repository and the same `repo` path. Report whether both calls succeeded and where the index was stored.
