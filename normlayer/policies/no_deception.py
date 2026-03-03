"""NoDeception policy — detects unfaithful agent restatements via embedding similarity."""

from __future__ import annotations

from typing import Any

from normlayer.base_policy import AgentMessage, BasePolicy, HandlerType, PolicyResult

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class NoDeception(BasePolicy):
    """Detects when an agent misrepresents another agent's message.

    Uses cosine similarity between sentence embeddings to measure whether an
    agent's restatement or summary of another agent's message is faithful to
    the original. A similarity score below `threshold` is treated as potential
    deception and fires a violation.

    The ``sentence-transformers`` library is imported **lazily** on first use
    so that the package remains installable without the optional ``[embeddings]``
    extra. The model is cached on the instance after the first load.

    Args:
        threshold: Minimum cosine similarity to consider a restatement faithful
            (default ``0.8``). Range is roughly [−1, 1] but in practice [0, 1].
        model_name: SentenceTransformer model identifier
            (default ``"all-MiniLM-L6-v2"`` — fast, ~80 MB, CPU-friendly).
        handler: Action to take on violation (default ``"warn"``).

    Context keys:
        original_message (str): The source message the evaluating agent is
            summarizing. If absent, the policy passes by default — there is
            nothing to compare against.
    """

    name: str = "NoDeception"

    def __init__(
        self,
        threshold: float = 0.8,
        model_name: str = _DEFAULT_MODEL,
        handler: HandlerType = "warn",
    ) -> None:
        super().__init__(handler=handler)
        self.threshold = threshold
        self.model_name = model_name
        self._model: Any = None  # Loaded lazily on first call to score_faithfulness.

    def _get_model(self) -> Any:
        """Return the SentenceTransformer model, loading it on first access.

        Returns:
            A loaded ``SentenceTransformer`` instance.

        Raises:
            ImportError: If ``sentence-transformers`` is not installed.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "NoDeception requires 'sentence-transformers'. "
                    "Install it with: pip install normlayer[embeddings]"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def score_faithfulness(self, original: str, summary: str) -> float:
        """Compute cosine similarity between the original and summary embeddings.

        Args:
            original: The source message from another agent.
            summary: The current agent's restatement or summary of that message.

        Returns:
            Cosine similarity in [−1, 1]. Values above 0.8 indicate faithful
            restatement; values near 0 or negative indicate divergence.
        """
        from sentence_transformers import util

        model = self._get_model()
        embeddings = model.encode([original, summary], convert_to_tensor=True)
        return float(util.cos_sim(embeddings[0], embeddings[1]))

    def evaluate(self, message: AgentMessage, context: dict[str, Any]) -> PolicyResult:
        """Evaluate whether the message faithfully represents its source.

        Requires ``context["original_message"]`` to be set to the source text.
        If the key is absent, the policy passes by default.

        Args:
            message: The AgentMessage containing the agent's restatement.
            context: Must contain ``original_message: str`` for evaluation.

        Returns:
            PolicyResult indicating pass or deception violation.
        """
        original: str | None = context.get("original_message")

        if not original:
            return PolicyResult(
                passed=True,
                violation_score=0.0,
                policy_name=self.name,
                agent_id=message.sender,
                handler=self.handler,
                severity="low",
                details="No original_message in context; faithfulness check skipped.",
            )

        similarity = self.score_faithfulness(original, message.content)
        passed = similarity >= self.threshold
        violation_score = max(0.0, 1.0 - similarity)

        return PolicyResult(
            passed=passed,
            violation_score=violation_score,
            policy_name=self.name,
            agent_id=message.sender,
            handler=self.handler,
            severity="high" if violation_score > 0.5 else "medium",
            details=(
                ""
                if passed
                else (
                    f"Faithfulness score {similarity:.3f} is below threshold "
                    f"{self.threshold}. Possible misrepresentation by "
                    f"'{message.sender}'."
                )
            ),
        )
