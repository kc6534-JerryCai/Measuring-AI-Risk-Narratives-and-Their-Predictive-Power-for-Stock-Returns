import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import URLError

from annotate_deepseek_risk import (
    DEFAULT_INPUT_ROOT,
    DeepSeekClient,
    Journal,
    Outcome,
    RateLimiter,
    Settings,
    TextUnit,
    annotate_units,
    boundary_units,
    flatten,
    load_units,
    select_units_for_mode,
    validate_annotation,
)


def unit(*, headline=True, target="AI could replace workers.", context="", key="1"):
    kind = "headline" if headline else "clause"
    return TextUnit(
        unit_key=f"test|{kind}|{key}",
        unit_type=kind,
        platform="test",
        document_id="doc-1",
        source_id="source-1",
        sentence_id="" if headline else "sentence-1",
        clause_id="" if headline else "clause-1",
        published_at="",
        time_basis="",
        source_segment="title" if headline else "body",
        target_text=target,
        context_text=context,
    )


def irrelevant(*, review=False):
    return {
        "topic_relevance": 0,
        "topic_relevance_reason": "仅讨论效率，未涉及五类风险。",
        "topic_relevance_evidence": [],
        "risk_type": None,
        "risk_type_reason": None,
        "risk_type_evidence": [],
        "risk_direction": None,
        "direction_status": "not_applicable",
        "risk_direction_reason": None,
        "risk_direction_evidence": [],
        "needs_review": review,
        "review_reason": "指代不清，证据不足。" if review else "",
    }


def relevant(*, direction=1, status="risk_affirming", review=False):
    return {
        "topic_relevance": 1,
        "topic_relevance_reason": "文本警示 AI 可能替代工作。",
        "topic_relevance_evidence": [
            {"source": "target", "quote": "AI could replace workers"}
        ],
        "risk_type": ["job_displacement"],
        "risk_type_reason": "涉及潜在就业替代。",
        "risk_type_evidence": [
            {"source": "target", "quote": "replace workers"}
        ],
        "risk_direction": direction,
        "direction_status": status,
        "risk_direction_reason": "以条件性措辞警示潜在伤害。",
        "risk_direction_evidence": (
            [{"source": "target", "quote": "could replace workers"}]
            if direction != 0 or status == "mixed"
            else []
        ),
        "needs_review": review,
        "review_reason": "同一目标包含相反方向。" if review else "",
    }


def generic(*, direction=1, status="risk_affirming", review=False):
    return {
        "topic_relevance": 0.2,
        "topic_relevance_reason": "明确讨论笼统 AI 风险，但没有具体类别证据。",
        "topic_relevance_evidence": [
            {"source": "target", "quote": "warned about AI risks"}
        ],
        "risk_type": [],
        "risk_type_reason": "文本只表达笼统 AI 风险，无法归入五类中的具体类别。",
        "risk_type_evidence": [],
        "risk_direction": direction,
        "direction_status": status,
        "risk_direction_reason": "warned 明确警示笼统风险。",
        "risk_direction_evidence": (
            [{"source": "target", "quote": "warned about AI risks"}]
            if direction != 0 or status == "mixed"
            else []
        ),
        "needs_review": review,
        "review_reason": "同一目标包含相反方向。" if review else "",
    }


class ValidationTests(unittest.TestCase):
    def test_topic_zero_requires_null_dependent_dimensions(self):
        result = validate_annotation(irrelevant(), unit(target="AI improves efficiency."))
        self.assertIsNone(result["risk_type"])
        self.assertIsNone(result["risk_direction"])
        self.assertEqual(result["direction_status"], "not_applicable")

    def test_insufficient_context_is_zero_with_review(self):
        result = validate_annotation(irrelevant(review=True), unit(target="It comes."))
        self.assertEqual(result["topic_relevance"], 0)
        self.assertTrue(result["needs_review"])

    def test_conditional_harm_is_risk_affirming(self):
        result = validate_annotation(relevant(), unit())
        self.assertEqual(result["risk_direction"], 1)
        self.assertEqual(result["direction_status"], "risk_affirming")

    def test_generic_ai_risk_uses_point_two_and_keeps_direction(self):
        target = "A senator warned about AI risks."
        result = validate_annotation(generic(), unit(target=target))
        self.assertEqual(result["topic_relevance"], 0.2)
        self.assertEqual(result["risk_type"], [])
        self.assertEqual(result["risk_direction"], 1)
        self.assertEqual(result["direction_status"], "risk_affirming")
        self.assertFalse(result["needs_review"])

    def test_generic_ai_risk_requires_empty_type_list(self):
        value = generic()
        value["risk_type"] = None
        with self.assertRaisesRegex(ValueError, r"risk_type=\[\]"):
            validate_annotation(value, unit(target="A senator warned about AI risks."))

    def test_generic_ai_risk_still_requires_direction(self):
        value = generic()
        value["risk_direction"] = None
        value["direction_status"] = "not_applicable"
        with self.assertRaisesRegex(ValueError, "requires integer risk_direction"):
            validate_annotation(value, unit(target="A senator warned about AI risks."))

    def test_unapproved_fractional_topic_score_is_rejected(self):
        value = generic()
        value["topic_relevance"] = 0.5
        with self.assertRaisesRegex(ValueError, "numeric 0, 0.2, or 1"):
            validate_annotation(value, unit(target="A senator warned about AI risks."))

    def test_direction_status_must_match_score(self):
        value = relevant(direction=-1, status="risk_affirming")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            validate_annotation(value, unit())

    def test_mixed_requires_review(self):
        value = relevant(direction=0, status="mixed", review=False)
        with self.assertRaisesRegex(ValueError, "mixed direction requires needs_review"):
            validate_annotation(value, unit())

    def test_exact_quote_is_enforced(self):
        value = relevant()
        value["risk_type_evidence"][0]["quote"] = "replaces workers"
        with self.assertRaisesRegex(ValueError, "not an exact input substring"):
            validate_annotation(value, unit())

    def test_context_alone_cannot_establish_relevance(self):
        target = "which could do so."
        context = "An AI system could replace workers, which could do so."
        value = relevant()
        value["topic_relevance_evidence"] = [
            {"source": "context", "quote": "AI system could replace workers"}
        ]
        value["risk_type_evidence"] = [
            {"source": "context", "quote": "replace workers"}
        ]
        value["risk_direction_evidence"] = [
            {"source": "target", "quote": "could do so"}
        ]
        with self.assertRaisesRegex(ValueError, "target evidence"):
            validate_annotation(
                value,
                unit(headline=False, target=target, context=context),
            )

    def test_topic_one_cannot_admit_category_is_unknown(self):
        value = relevant()
        value["risk_type_reason"] = "文本仅笼统提及风险，无法确定具体类别。"
        with self.assertRaisesRegex(ValueError, "admits that no category"):
            validate_annotation(value, unit())

    def test_topic_one_cannot_hide_category_uncertainty_in_review_reason(self):
        value = relevant(review=True)
        value["review_reason"] = "证据较弱，无法可靠确定具体风险类别。"
        with self.assertRaisesRegex(ValueError, "admits that no category"):
            validate_annotation(value, unit())

    def test_headline_cannot_cite_context(self):
        value = relevant()
        value["topic_relevance_evidence"] = [
            {"source": "context", "quote": "replace workers"}
        ]
        with self.assertRaisesRegex(ValueError, "headline evidence"):
            validate_annotation(
                value,
                unit(context="AI could replace workers."),
            )

    def test_topic_zero_rejects_non_null_risk_type(self):
        value = irrelevant()
        value["risk_type"] = []
        with self.assertRaisesRegex(ValueError, "risk_type=null"):
            validate_annotation(value, unit(target="AI improves efficiency."))

    def test_boolean_is_not_accepted_as_integer(self):
        value = irrelevant()
        value["topic_relevance"] = False
        with self.assertRaisesRegex(ValueError, "numeric 0, 0.2, or 1"):
            validate_annotation(value, unit(target="AI improves efficiency."))


class InputAndBoundaryTests(unittest.TestCase):
    def test_input_loader_keeps_headlines_separate_from_title_clauses(self):
        headlines, clauses, manifest = load_units(Path(DEFAULT_INPUT_ROOT))
        self.assertTrue(headlines)
        self.assertTrue(clauses)
        self.assertTrue(all(row.unit_type == "headline" for row in headlines))
        self.assertTrue(all(row.source_segment != "title" for row in clauses))
        self.assertGreater(manifest["x"]["excluded_title_clause_rows"], 0)
        self.assertGreater(manifest["guardian"]["excluded_title_clause_rows"], 0)

    def test_context_boundaries_are_clause_targets(self):
        by_id = {row.source_id: row for row in boundary_units()}
        self.assertEqual(by_id["context_reference"].unit_type, "clause")
        self.assertIn("AI tool", by_id["context_reference"].context_text)
        self.assertEqual(by_id["context_no_transfer"].target_text, "the company apologized.")

    def test_full_clauses_mode_excludes_every_headline_and_title_clause(self):
        headlines, clauses, _ = load_units(Path(DEFAULT_INPUT_ROOT))
        selected, boundaries = select_units_for_mode(
            "full-clauses",
            headlines,
            clauses,
            sample_size=200,
            seed=20261002,
            no_boundary_tests=False,
        )
        self.assertFalse(boundaries)
        self.assertEqual(len(selected), 57034)
        self.assertTrue(all(row.unit_type == "clause" for row in selected))
        self.assertTrue(all(row.source_segment != "title" for row in selected))
        self.assertEqual(sum(row.platform == "x" for row in selected), 13284)
        self.assertEqual(sum(row.platform == "guardian" for row in selected), 43750)

    def test_full_clauses_mode_can_select_only_x(self):
        headlines, clauses, _ = load_units(Path(DEFAULT_INPUT_ROOT))
        selected, boundaries = select_units_for_mode(
            "full-clauses",
            headlines,
            clauses,
            sample_size=200,
            seed=20261002,
            no_boundary_tests=False,
            platform="x",
        )
        self.assertFalse(boundaries)
        self.assertEqual(len(selected), 13284)
        self.assertTrue(all(row.platform == "x" for row in selected))
        self.assertTrue(all(row.unit_type == "clause" for row in selected))
        self.assertTrue(all(row.source_segment != "title" for row in selected))

    def test_full_headlines_mode_can_select_only_x(self):
        headlines, clauses, _ = load_units(Path(DEFAULT_INPUT_ROOT))
        selected, boundaries = select_units_for_mode(
            "full-headlines",
            headlines,
            clauses,
            sample_size=200,
            seed=20261002,
            no_boundary_tests=False,
            platform="x",
        )
        self.assertFalse(boundaries)
        self.assertEqual(len(selected), 1882)
        self.assertTrue(all(row.platform == "x" for row in selected))
        self.assertTrue(all(row.unit_type == "headline" for row in selected))


class ResilienceTests(unittest.TestCase):
    def settings(self, attempts=2):
        return Settings(
            base_url="https://api.deepseek.example",
            model="test-model",
            temperature=0.0,
            max_tokens=1000,
            timeout=5.0,
            max_attempts=attempts,
        )

    def test_models_preflight_retries_transport_failure(self):
        client = DeepSeekClient("not-a-real-key", self.settings(), RateLimiter(0))
        with mock.patch.object(
            client,
            "request",
            side_effect=[URLError("temporary"), {"data": [{"id": "model-b"}, {"id": "model-a"}]}],
        ), mock.patch("annotate_deepseek_risk.time.sleep"):
            self.assertEqual(client.list_models(), ["model-a", "model-b"])

    def test_validation_failure_retries_then_writes_and_reuses_cache(self):
        annotation = relevant()
        invalid_response = {
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]
        }
        valid_response = {
            "id": "response-1",
            "model": "test-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(annotation, ensure_ascii=False)},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        client = DeepSeekClient("not-a-real-key", self.settings(), RateLimiter(0))
        with tempfile.TemporaryDirectory() as folder, mock.patch.object(
            client, "request", side_effect=[invalid_response, valid_response]
        ), mock.patch("annotate_deepseek_risk.time.sleep"):
            outcome = client.annotate(unit(), Path(folder))
            self.assertEqual(outcome.attempts, 2)
            self.assertEqual(outcome.source, "api")
            self.assertEqual(
                outcome.metadata["validation_retry_version"],
                "exact_quote_retry_v2_2026-10-02",
            )
            with mock.patch.object(client, "request", side_effect=AssertionError("API reused")):
                cached = client.annotate(unit(), Path(folder))
            self.assertEqual(cached.source, "cache")
            self.assertEqual(cached.annotation["risk_direction"], 1)

    def test_csv_nulls_are_explicit_and_consistent(self):
        row = irrelevant()
        flat = flatten(row)
        self.assertEqual(flat["risk_type"], "null")
        self.assertEqual(flat["risk_type_reason"], "null")
        self.assertEqual(flat["risk_direction"], "null")
        self.assertEqual(flat["risk_direction_reason"], "null")

    def test_csv_generic_risk_keeps_empty_type_list(self):
        flat = flatten(generic())
        self.assertEqual(flat["topic_relevance"], 0.2)
        self.assertEqual(flat["risk_type"], "[]")
        self.assertEqual(flat["risk_direction"], 1)

    def test_max_new_requests_defers_remaining_units(self):
        client = mock.Mock()
        client.annotate.side_effect = lambda _unit, _cache: Outcome(
            annotation=relevant(),
            metadata={"actual_model": "test-model"},
            source="api",
            attempts=1,
        )
        units = [
            unit(target=f"AI could replace workers {index}.", key=str(index))
            for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results, errors, stats = annotate_units(
                units,
                client,
                self.settings(),
                root / "cache",
                Journal(root / "progress.jsonl"),
                Journal(root / "errors.jsonl"),
                workers=1,
                offline=False,
                max_new_requests=2,
            )
        self.assertEqual(len(results), 2)
        self.assertFalse(errors)
        self.assertEqual(client.annotate.call_count, 2)
        self.assertEqual(stats["deferred_unique_requests"], 1)
        self.assertEqual(stats["deferred_units"], 1)


if __name__ == "__main__":
    unittest.main()
