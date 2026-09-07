"""Schema tests for ``AgentDef.settings_dir``.

``settings_dir`` names the directory whose Claude Code *project* settings
tier an agent loads skills from. It exists because ``working_dir`` was doing
two unrelated jobs at once: the CLI advertises its cwd as its sole MCP root,
so narrowing cwd onto a target repository to pick up that repository's
skills also narrowed what the agent's MCP servers were permitted to read.

These tests pin the field's shape and, more importantly, that it stays
*independent* of ``working_dir`` -- a schema that coupled them, or that
silently accepted the field on a step type with no LLM session to apply it
to, would reintroduce the confusion the split removes.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from conductor.config.schema import AgentDef, GateOption


class TestSettingsDirAccepted:
    """Provider-backed agents take the field, alone or alongside cwd."""

    def test_accepted_on_a_plain_llm_agent(self) -> None:
        agent = AgentDef(name="judge", prompt="review", settings_dir="/repo")
        assert agent.settings_dir == "/repo"

    def test_defaults_to_none(self) -> None:
        """Omitting it must add no directory -- the tier is opt-in."""
        assert AgentDef(name="judge", prompt="review").settings_dir is None

    def test_independent_of_working_dir(self) -> None:
        """The point of the field: both set, to different directories.

        A wide cwd keeps every path the agent's MCP servers must reach inside
        the single negotiated root, while the narrow settings_dir supplies the
        target repository's conventions.
        """
        agent = AgentDef(
            name="judge", prompt="review", working_dir="/wide", settings_dir="/wide/repo"
        )
        assert (agent.working_dir, agent.settings_dir) == ("/wide", "/wide/repo")

    def test_accepts_a_template(self) -> None:
        """The directory a step reviews is normally an upstream step's output,
        so the raw value is a Jinja template the engine renders."""
        agent = AgentDef(
            name="judge", prompt="review", settings_dir="{{ setup.output.worktree_path }}"
        )
        assert agent.settings_dir == "{{ setup.output.worktree_path }}"


class TestSettingsDirRejectedOnNonProviderSteps:
    """Every step type with no LLM session rejects it.

    Accepting it silently is the failure mode that matters here: an author
    would see a green ``conductor validate`` and conclude the target
    repository's conventions were loaded when nothing had been.
    """

    @pytest.mark.parametrize(
        ("kwargs",),
        [
            ({"type": "wait", "duration": "1s"},),
            ({"type": "set", "value": "x"},),
            ({"type": "terminate", "status": "success", "reason": "done"},),
            ({"type": "script", "command": "echo hi"},),
            ({"type": "workflow", "workflow": "child.yaml"},),
            (
                {
                    "type": "human_gate",
                    "prompt": "ok?",
                    "options": [GateOption(label="OK", value="ok", route="$end")],
                },
            ),
        ],
    )
    def test_rejected(self, kwargs: dict) -> None:
        with pytest.raises(ValidationError, match="cannot have 'settings_dir'"):
            AgentDef(name="bad", settings_dir="/repo", **kwargs)

    def test_error_names_the_step_type(self) -> None:
        """So the message says which step to fix, not merely that one is wrong."""
        with pytest.raises(ValidationError, match="wait agents cannot have 'settings_dir'"):
            AgentDef(name="bad", type="wait", duration="1s", settings_dir="/repo")
