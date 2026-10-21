"""Unit tests for the discover, probe, and render helper functions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from discover import (  # noqa: E402
    extract_cloud_model_metadata,
    extract_family_usage_from_search,
    extract_usage,
    merge_usage,
    normalize_model_name,
)
from probe import classify, sanitize_excerpt  # noqa: E402
from render_readme import STATUS_LABELS, STATUS_ORDER, format_usage  # noqa: E402


class TestExtractUsage:
    def test_low(self):
        assert extract_usage("gemma4:cloud — Low usage from $0.10") == "low"

    def test_medium(self):
        assert extract_usage("Medium usage from $0.15") == "medium"

    def test_high(self):
        assert extract_usage("High usage $1.40 in") == "high"

    def test_extra_high(self):
        assert extract_usage("Extra High  usage and rising") == "extra high"

    def test_missing(self):
        assert extract_usage("no usage info here") is None


class TestNormalizeModelName:
    def test_library_prefix(self):
        assert normalize_model_name("/library/gemma4") == "gemma4"

    def test_cloud_tag(self):
        assert normalize_model_name("/library/glm-5.3:cloud") == "glm-5.3:cloud"

    def test_rejects_namespace(self):
        assert normalize_model_name("user/model:cloud") is None

    def test_rejects_empty(self):
        assert normalize_model_name("/library/") is None


class TestSearchFamilyUsage:
    SEARCH_HTML = """
    <html><body>
    <a href="/library/gemma4" class="group">
      <h2><span>gemma4</span></h2>
      <p>Frontier models</p>
      <span>Medium usage</span>
      <span>From $0.30 in</span>
    </a>
    <a href="/library/deepseek-v4-pro">
      <span>deepseek-v4-pro</span>
      <span>Extra High usage</span>
    </a>
    <a href="/library/no-usage-family">
      <span>no-usage-family</span>
      <span>no text</span>
    </a>
    </body></html>
    """

    def test_extracts_family_usage(self):
        usage = extract_family_usage_from_search(self.SEARCH_HTML)
        assert usage == {"gemma4": "medium", "deepseek-v4-pro": "extra high"}

    def test_ignores_tag_links(self):
        usage = extract_family_usage_from_search('<a href="/library/gemma4:cloud">High usage</a>')
        assert usage == {}

    def test_merge_usage_overwrites(self):
        target = {"a": "high"}
        merge_usage(target, "a", "low")
        merge_usage(target, "b", "low")
        merge_usage(target, "c", None)
        assert target == {"a": "low", "b": "low"}


class TestFamilyPageMetadata:
    def test_extracts_cloud_tags_and_usage(self):
        html = """
        <html><body>
        <a href="/library/gemma4:31b-cloud">31b-cloud 4.1GB · High usage</a>
        <a href="/library/gemma4">gemma4</a>
        </body></html>
        """
        metadata = extract_cloud_model_metadata(html)
        assert "gemma4:31b-cloud" in metadata
        assert metadata["gemma4:31b-cloud"]["usage"] == "high"


class TestClassify:
    def test_success_is_free(self):
        assert classify(200, None) == "free"

    def test_subscription_error(self):
        assert classify(402, "This model is not in the Free plan") == "requires_subscription"

    def test_decision_model_is_unsupported(self):
        assert classify(400, "clef:cloud is a decision model and only supports /v1/systemone") == "unsupported"

    def test_only_supports_is_unsupported(self):
        assert classify(400, "model only supports /v1/embeddings") == "unsupported"

    def test_other_400_stays_error(self):
        assert classify(400, "invalid request body") == "error"

    def test_rollout_403_stays_error(self):
        assert classify(403, "This model is currently being rolled out and is not yet available to you") == "error"

    def test_network_error_is_error(self):
        assert classify(None, "connection reset") == "error"


class TestSanitizeExcerpt:
    def test_redacts_api_key(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_API_KEY", "secret-key-123")
        assert sanitize_excerpt("auth failed for secret-key-123") == "auth failed for [redacted]"

    def test_redacts_bearer(self):
        assert sanitize_excerpt("Bearer abc123 denied") == "Bearer [redacted] abc123 denied"

    def test_strips_urls(self):
        assert sanitize_excerpt("see https://ollama.com/help for details") == "see for details"

    def test_truncates(self):
        assert len(sanitize_excerpt("x" * 300)) == 180

    def test_none_passthrough(self):
        assert sanitize_excerpt(None) is None


class TestRenderHelpers:
    def test_status_labels_complete(self):
        assert set(STATUS_LABELS) == {"free", "requires_subscription", "unsupported", "error"}
        assert set(STATUS_ORDER) == set(STATUS_LABELS)

    def test_format_usage_meters(self):
        assert format_usage("low") == "▰▱▱▱"
        assert format_usage("medium") == "▰▰▱▱"
        assert format_usage("high") == "▰▰▰▱"
        assert format_usage("extra high") == "▰▰▰▰"
        assert format_usage(None) == "—"
        assert format_usage("unknown") == "—"
