# Sources Used

Verified on 2026-05-27 unless noted otherwise.

## Seed Sources

- OpenCode archived repository: https://github.com/opencode-ai/opencode
- OpenCode website: https://www.opencode.live/
- OpenCode documentation: https://opencode.ai/docs/ and https://dev.opencode.ai/
- OpenCode agents documentation: https://opencode.ai/docs/agents/
- OpenCode tools documentation: https://opencode.ai/docs/tools/
- OpenCode configuration documentation: https://opencode.ai/docs/config
- LLM-Based Agents for Tool Learning: A Survey. Data Science and Engineering, 2025. https://link.springer.com/article/10.1007/s41019-025-00296-9
- Advancing Tool-Augmented Large Language Models via Meta-Verification and Reflection Learning / Tool-MVR. arXiv:2506.04625. https://arxiv.org/abs/2506.04625
- Sample, Predict, then Proceed: Self-Verification Sampling for Tool Use of LLMs. arXiv:2506.02918. https://arxiv.org/abs/2506.02918
- A Review of Prominent Paradigms for LLM-Based Agents: Tool Use, Planning, and Feedback Learning. arXiv:2406.05804. https://arxiv.org/abs/2406.05804
- ProtAgents: Protein discovery via large language model multi-agent collaborations combining physics and machine learning. arXiv:2402.04268. https://arxiv.org/abs/2402.04268
- ProteinCrow: A Language Model Agent That Can Design Proteins. OpenReview. https://openreview.net/forum?id=ljXgWDtqCu
- AutoProteinEngine: A Large Language Model Driven Agent Framework for Multimodal AutoML in Protein Engineering. COLING Industry Track 2025. https://aclanthology.org/2025.coling-industry.36.pdf
- Protein Design with Agent Rosetta: A Case Study for Specialized Scientific Agents. arXiv:2603.15952. https://arxiv.org/abs/2603.15952
- Beyond Protein Language Models: An Agentic LLM Framework for Mechanistic Enzyme Design / Genie-CAT. arXiv:2511.19423. https://arxiv.org/abs/2511.19423
- DynaMate: An Autonomous Agent for Protein-Ligand Molecular Dynamics Simulations. arXiv:2512.10034. https://arxiv.org/abs/2512.10034

## General Agent Foundations

- ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629. https://arxiv.org/abs/2210.03629
- Toolformer: Language Models Can Teach Themselves to Use Tools. arXiv:2302.04761. https://arxiv.org/abs/2302.04761
- Reflexion: Language Agents with Verbal Reinforcement Learning. arXiv:2303.11366. https://arxiv.org/abs/2303.11366
- Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arXiv:2305.10601. https://arxiv.org/abs/2305.10601
- Language Agent Tree Search Unifies Reasoning Acting and Planning in Language Models. arXiv:2310.04406. https://arxiv.org/abs/2310.04406
- Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv:2305.16291. https://arxiv.org/abs/2305.16291
- ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arXiv:2307.16789. https://arxiv.org/abs/2307.16789
- AgentTuning: Enabling Generalized Agent Abilities for LLMs. arXiv:2310.12823. https://arxiv.org/abs/2310.12823
- AgentGym: Evolving Large Language Model-based Agents across Diverse Environments. arXiv:2406.04151. https://arxiv.org/abs/2406.04151
- WebArena: A Realistic Web Environment for Building Autonomous Agents. arXiv:2307.13854. https://arxiv.org/abs/2307.13854
- SWE-bench: Can Language Models Resolve Real-World GitHub Issues? arXiv:2310.06770. https://arxiv.org/abs/2310.06770
- SWE-bench official site. https://www.swebench.com/original.html

## Recent Agent Training, Process Supervision, and Verifier Work

- RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning. arXiv:2504.20073. https://arxiv.org/abs/2504.20073
- SPA-RL: Reinforcing LLM Agents via Stepwise Progress Attribution. arXiv:2505.20732. https://arxiv.org/abs/2505.20732
- RRO: LLM Agent Optimization Through Rising Reward Trajectories. arXiv:2505.20737. https://arxiv.org/abs/2505.20737
- SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents. arXiv:2508.02085. https://arxiv.org/abs/2508.02085
- Training Software Engineering Agents and Verifiers with SWE-Gym. Apple Machine Learning Research, 2025. https://machinelearning.apple.com/research/training-software
- Improve Mathematical Reasoning in Language Models by Automated Process Supervision. arXiv:2406.06592. https://arxiv.org/abs/2406.06592
- Verifiable Process Rewards for Agentic Reasoning. arXiv:2605.10325. https://arxiv.org/abs/2605.10325

## Agent Architecture and Tooling

- OpenAI Codex repository. https://github.com/openai/codex
- OpenAI Codex CLI getting started. https://help.openai.com/en/articles/11096431
- Running Codex safely at OpenAI. https://openai.com/index/running-codex-safely/
- Claude Code overview. https://docs.anthropic.com/en/docs/claude-code/overview
- Claude Code subagents. https://docs.anthropic.com/en/docs/claude-code/sub-agents
- Claude Code MCP documentation. https://docs.anthropic.com/en/docs/claude-code/mcp
- LangGraph overview. https://docs.langchain.com/oss/python/langgraph
- LangGraph persistence. https://docs.langchain.com/oss/python/langgraph/persistence
- Model Context Protocol announcement. https://www.anthropic.com/news/model-context-protocol
- Model Context Protocol specification. https://modelcontextprotocol.io/specification/2024-11-05/index
- Model Context Protocol GitHub documentation repository. https://github.com/modelcontextprotocol/modelcontextprotocol

## Local VibeProteinAgent Files Inspected

- `README.md`
- `docs/use_llm_agent.md`
- `docs/architecture/planner_ir.md`
- `src/vibeproteinagent/agents/execution/controller.py`
- `src/vibeproteinagent/agents/planning/workflow_planner.py`
- `src/vibeproteinagent/agents/planning/compiler.py`
- `src/vibeproteinagent/agents/planning/prompts.py`
- `src/vibeproteinagent/agents/evaluation/decision.py`
- `src/vibeproteinagent/agents/evaluation/decision_validator.py`
- `src/vibeproteinagent/agents/session/store.py`
- `src/vibeproteinagent/agents/tools/broker.py`
- `src/vibeproteinagent/agents/tools/schemas.py`
- `src/vibeproteinagent/agents/provider/logging.py`
- `src/vibeproteinagent/agents/config.py`
- `src/vibeproteinagent/workflows/executor.py`
- `src/vibeproteinagent/workspace/schema.sql`

