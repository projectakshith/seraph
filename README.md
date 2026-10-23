# Seraph

**Code intelligence for coding agents:** search a repository by meaning, symbols, dependencies, and Git history.

## Get started

Install Seraph with one command:

```bash
python -m pip install seraph-mcp
```

Then tell your coding agent:

> Read [AGENT_SETUP.md](AGENT_SETUP.md) and set up Seraph for this repository.

The agent will register Seraph with its MCP support and check that it connects. Seraph starts in **Lite** mode, so it is ready without downloading a model.

## Search tiers

Switch tiers anytime. Fast and Quality add semantic search with local models.

| Tier | Search | Model download |
|---|---|---:|
| **Lite** (default) | BM25 keyword search | None |
| **Fast** | BM25 + `potion-code-16M` embeddings | 62 MB |
| **Quality** | BM25 + BGE-Code-v1 embeddings | About 6 GB |

Use `seraph setup` to choose interactively, or install a tier directly:

```bash
seraph setup --tier fast --install
seraph setup --tier quality --install
```

## Use Seraph with an agent

Register a local stdio MCP server with command `seraph-mcp`. Pass `repo` as the absolute path to the Git checkout on each tool call. Seraph keeps its index in `<repo>/.seraph/`; repository contents are not uploaded.

For an isolated `uvx` install, use command `uvx` with arguments `seraph-mcp`. See [AGENT_SETUP.md](AGENT_SETUP.md) for host-specific setup.

Seraph's MCP tools cover:

- **Search:** `search_code`, `search_at_version`, `search_history`, `search_worktree`.
- **Understand:** `find_symbol`, `find_dependencies`, `get_repo_map`.
- **Changes:** `compare_versions`, `get_working_tree_status`, `prepare_change_context`, `change_impact`.
- **Index and setup:** `index_repository`, `retrieval_status`, `set_retrieval_tier`.

## How it works

![Seraph architecture](https://github.com/user-attachments/assets/c5d717e2-6bf1-4298-a063-b4447efad135)

Seraph parses code into a versioned local index, then combines lexical, semantic, structural, graph, and change-history signals to return relevant code with source context.

## Results

On the CoIR AppsRetrieval test split (3,765 queries over 8,765 Python solutions):

| Embedder | nDCG@10 | MRR@10 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| **BGE-Code-v1** (submitted) | **0.9770** | **0.9702** | 0.9971 | 0.9992 |
| Qwen3-Embedding-0.6B | 0.7487 | 0.7039 | 0.8882 | 0.9835 |

These are retrieval-quality results on this benchmark, not a measure of agent task success. The benchmark and submission details are in the [PRISM 2026 release](https://github.com/wtfPrethiv/seraph/releases/tag/PRISM_GENAI_HACKATHON_Y2026).

For source checkout and evaluation instructions, see [AGENT_SETUP.md](AGENT_SETUP.md) and the [`experiments/`](experiments/) directory.

## Samsung PRISM submission

Team **floppydisk** · SRM Institute of Science and Technology · Theme 1: Agentic Code Intelligence

- [Presentation](docs/SRM_floppydisk_Submission.pptx) · [PDF](docs/SRM_floppydisk_Submission.pdf)
- [Demo video](https://drive.google.com/drive/folders/1TbShN44FSyemdtEXjca11KWpaW50liuU?usp=sharing)
- [AI usage disclosure](docs/floppydisk_AI_Disclosure.docx)

## License

MIT. See [LICENSE](LICENSE).
