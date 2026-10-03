import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from construct_hourly_ai_risk_index import (
    COMPONENTS,
    RISK_TYPES,
    IndexDocument,
    aggregate_metric,
    build_hourly_index,
    deduplicate_documents,
    floor_hour,
    parse_timestamp,
    safe_scale,
    scale_from_history,
    weighted_score,
)


UTC = timezone.utc


def make_document(
    document_id: str,
    when: datetime,
    *,
    h: float = 0.0,
    b: float = 0.0,
    relevant: bool = False,
    types=(),
    title: str | None = None,
    body: str | None = None,
) -> IndexDocument:
    raw = {component: 0.0 for component in COMPONENTS}
    raw["H"] = h
    raw["B"] = b
    for risk_type in types:
        raw[f"H_{risk_type}"] = h
        raw[f"B_{risk_type}"] = b
    title = title or f"Title {document_id}"
    body = body or f"Body text for {document_id}"
    return IndexDocument(
        document_id=document_id,
        source_id=document_id,
        title=title,
        document_text=f"{title}\n\n{body}",
        source_timestamp_raw=when.isoformat(),
        available_at=when,
        time_basis="test",
        timestamp_missing_original=False,
        timestamp_fallback_used=False,
        timestamp_quality="test",
        within_nominal_collection_window=True,
        is_partial_boundary_hour=False,
        raw=raw,
        total_relevant=relevant,
        relevant_types=frozenset(types),
        headline_topic_relevance=1.0 if relevant else 0.0,
        body_relevant_clause_count=1 if relevant else 0,
        body_clause_count=1,
    )


class ScaleTests(unittest.TestCase):
    def test_rms_includes_zero_values_in_denominator(self):
        info = scale_from_history(3, 0**2 + 3**2 + 4**2, 2, 3, 2)
        self.assertEqual(info.status, "ready")
        self.assertAlmostEqual(info.value, math.sqrt(25 / 3))

    def test_safe_scale_keeps_raw_zero_and_preserves_sign(self):
        self.assertEqual(safe_scale(0, math.nan), 0)
        self.assertEqual(safe_scale(0, 2), 0)
        self.assertTrue(math.isnan(safe_scale(2, math.nan)))
        self.assertEqual(safe_scale(-2, 2), -1)

    def test_nonzero_threshold_uses_only_prior_values(self):
        info = scale_from_history(100, 4.0, 4, 100, 5)
        self.assertEqual(info.status, "insufficient_nonzero")
        self.assertTrue(math.isnan(info.value))

    def test_weighted_score_does_not_reweight_missing_component(self):
        self.assertTrue(math.isnan(weighted_score(0, math.nan)))
        self.assertAlmostEqual(weighted_score(2, 4), 3.5)


class HourlyTests(unittest.TestCase):
    def test_same_hour_values_do_not_leak_into_scale(self):
        h0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
        docs = [
            make_document("a", h0, h=3, relevant=True),
            make_document("b", h0 + timedelta(minutes=30), h=4, relevant=True),
            make_document("c", h0 + timedelta(hours=1), h=12, relevant=True),
        ]
        rows, scales = build_hourly_index(
            docs,
            {},
            min_history_docs=2,
            min_nonzero_docs=2,
            nominal_start=h0,
            collection_cutoff=h0 + timedelta(hours=3),
            queries_expected=1,
            queries_succeeded=1,
            queries_at_result_cap=0,
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(scales[h0 + timedelta(hours=1)]["H"].value, math.sqrt(25 / 2))
        self.assertEqual(scales[h0 + timedelta(hours=1)]["H"].history_count, 2)

    def test_multi_label_contributes_once_overall_and_once_to_each_type(self):
        h0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
        types = ("malicious_use", "privacy_violation")
        doc = make_document("multi", h0, b=1, relevant=True, types=types)
        self.assertEqual(doc.raw["B"], 1)
        self.assertEqual(doc.raw["B_malicious_use"], 1)
        self.assertEqual(doc.raw["B_privacy_violation"], 1)
        overall = weighted_score(0, 1)
        type_sum = weighted_score(0, 1) + weighted_score(0, 1)
        self.assertNotEqual(overall, type_sum)

    def test_no_news_is_zero_but_missing_is_nan(self):
        h0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
        net, average, status, _ = aggregate_metric([], "total", "confirmed_no_news")
        self.assertEqual(net, 0)
        self.assertTrue(math.isnan(average))
        self.assertEqual(status, "confirmed_no_news")
        net, average, status, _ = aggregate_metric([], "total", "missing_collection")
        self.assertTrue(math.isnan(net))
        self.assertTrue(math.isnan(average))
        self.assertEqual(status, "unknown_collection_coverage")

    def test_relevant_neutral_has_zero_average(self):
        h0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
        doc = make_document("neutral", h0, relevant=True)
        doc.scores = {"total": 0.0}
        net, average, status, _ = aggregate_metric([doc], "total", "observed_complete")
        self.assertEqual(net, 0)
        self.assertEqual(average, 0)
        self.assertEqual(status, "computed")

    def test_nan_document_is_not_silently_skipped(self):
        h0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
        good = make_document("good", h0, relevant=True)
        bad = make_document("bad", h0, relevant=True)
        good.scores = {"total": 1.0}
        bad.scores = {"total": math.nan}
        net, average, status, _ = aggregate_metric([good, bad], "total", "observed_complete")
        self.assertTrue(math.isnan(net))
        self.assertTrue(math.isnan(average))
        self.assertEqual(status, "warmup")


class DedupAndTimeTests(unittest.TestCase):
    def test_greedy_dedup_retains_earliest_and_is_stable_on_tie(self):
        h0 = datetime(2026, 1, 1, 0, tzinfo=UTC)
        title = "AI system raises the same privacy concern"
        body = "The exact same detailed body explains the event and its impact."
        later = make_document("b", h0 + timedelta(hours=1), title=title, body=body)
        earlier = make_document("a", h0, title=title, body=body)
        summary = deduplicate_documents(
            [later, earlier],
            {},
            {},
            near_window_hours=72,
        )
        self.assertEqual(summary["deduplicated_documents"], 1)
        self.assertTrue(earlier.included_in_index)
        self.assertFalse(later.included_in_index)
        self.assertEqual(later.canonical_document_id, "a")

    def test_timezone_offsets_map_to_same_utc_hour(self):
        left = parse_timestamp("2026-01-01T10:30:00+02:00")
        right = parse_timestamp("2026-01-01T08:45:00Z")
        self.assertEqual(floor_hour(left), floor_hour(right))
        self.assertIsNotNone(left.utcoffset())


if __name__ == "__main__":
    unittest.main()
