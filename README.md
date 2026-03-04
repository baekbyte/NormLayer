# NormLayer

[![Tests](https://github.com/baekbyte/NormLayer/actions/workflows/ci.yml/badge.svg)](https://github.com/baekbyte/NormLayer/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/normlayer.svg)](https://pypi.org/project/normlayer/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A framework-agnostic Python SDK that enforces behavioral policies between agents at runtime in multi-agent pipelines.**

## The Problem

When multiple agents work together (one planning, one executing, one reviewing) there is no standard way to enforce *how they behave toward each other*. Existing safety tools focus on agent-to-human behavior. NormLayer focuses on **agent-to-agent behavior**: detecting deception between agents, enforcing role boundaries, catching collusion, and escalating conflicts — all at runtime, across any framework.

## Installation

```bash
# Core SDK (no heavy dependencies)
pip install normlayer

# With semantic scoring for NoDeception policy
pip install normlayer[embeddings]

# With AWS S3 logging + SageMaker audit
pip install normlayer[aws]

# Everything
pip install normlayer[all]
```

## Quick Start

```python
from normlayer import PolicyEngine, policies

engine = PolicyEngine(
    policies=[
        policies.NoDeception(threshold=0.8, handler="escalate"),
        policies.RoleRespect(strict=True, handler="block"),
        policies.EscalateOnConflict(to="supervisor_agent"),
        policies.LoopDetection(max_repetitions=3, handler="warn"),
        policies.NoUnsanctionedAction(
            permissions={"worker": ["execute", "update"]},
            handler="block",
        ),
    ],
)

# Use as a decorator
@engine.enforce
def my_agent(message, context):
    return response

# Or wrap an existing agent inline
safe_agent = engine.wrap(existing_agent)
```

## Built-in Policies

| Policy | Description | Default Handler |
|--------|-------------|-----------------|
| `NoDeception` | Detects when an agent misrepresents information to another agent using embedding similarity | `warn` |
| `RoleRespect` | Flags agents operating outside their defined role/scope | `warn` |
| `EscalateOnConflict` | Triggers escalation to a supervisor when agents disagree past a threshold | `escalate` |
| `LoopDetection` | Detects agents stuck in unproductive repetitive exchanges | `warn` |
| `ResponseProportionality` | Catches disproportionate responses relative to the triggering input | `warn` |
| `CoalitionConsistency` | Checks whether agents apply norms consistently across in-group vs. out-group | `warn` |
| `NormConflictResolution` | Detects contradictory directives given to an agent (e.g., "be brief" + "be thorough") | `warn` |
| `NoUnsanctionedAction` | Enforces action allowlists per agent — blocks unauthorized actions | `block` |

## Framework Adapters

NormLayer works with any multi-agent framework through thin adapters.

### LangGraph

```python
from normlayer.adapters import LangGraphAdapter

adapter = LangGraphAdapter(engine)
safe_graph = adapter.wrap(compiled_graph)
result = safe_graph.invoke({"messages": [HumanMessage(content="Plan the task")]})
```

### CrewAI

```python
from normlayer.adapters import CrewAIAdapter

adapter = CrewAIAdapter(engine)
safe_crew = adapter.wrap(my_crew)
result = safe_crew.kickoff()
```

### AutoGen

```python
from normlayer.adapters import AutoGenAdapter

adapter = AutoGenAdapter(engine)
safe_agent = adapter.wrap(my_agent)
response = await safe_agent.on_messages(messages, cancellation_token)
```

## AWS Integration

### S3 Violation Logging

```python
engine = PolicyEngine(
    policies=[...],
    aws_bucket="my-normlayer-logs",
    aws_region="us-east-1",
)

# Violations are automatically shipped to S3
# Flush buffered violations manually if needed
engine.flush_violations()
```

### SageMaker Batch Audit

```python
from normlayer.logging import SageMakerAuditJob

job = SageMakerAuditJob(
    role_arn="arn:aws:iam::123456789:role/SageMakerRole",
    input_s3_uri="s3://my-normlayer-logs/violations/",
    output_s3_uri="s3://my-normlayer-logs/audit-results/",
)
job.run()
print(job.status())
```

## Handler Actions

Each policy can dispatch one of four handler actions on violation:

| Handler | Behavior |
|---------|----------|
| `block` | Raises `EnforcementError`, stopping the message |
| `warn` | Logs the violation but allows the message through |
| `escalate` | Routes to a designated supervisor agent |
| `log` | Records silently for audit; no visible action |

## Examples

See the `examples/` directory for demo notebooks:

- [`langgraph_demo.ipynb`](examples/langgraph_demo.ipynb) — wrapping a LangGraph compiled graph
- [`crewai_demo.ipynb`](examples/crewai_demo.ipynb) — wrapping a CrewAI crew
- [`autogen_demo.ipynb`](examples/autogen_demo.ipynb) — wrapping an AutoGen agent

## Development

```bash
# Clone and install dev dependencies
git clone https://github.com/baekbyte/NormLayer.git
cd normlayer
pip install -e ".[dev]"

# Run tests
pytest --tb=short -q

# Type checking
mypy normlayer/
```

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.
