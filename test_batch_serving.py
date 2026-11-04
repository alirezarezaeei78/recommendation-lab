import copy
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
import numpy as np
from app import handler_for
from batch_serving import BatchServing, batch_csv, fixed_order_scores
from bpr import BPRRecommender
from recommender import Row, synthetic_rows


class BatchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = BPRRecommender(synthetic_rows(), epochs=3)
        cls.engine = BatchServing(cls.model)
        cls.queries = [{'query_id': f'q-{i}', 'user_id': u} for i, u in enumerate(cls.model.users)]
        cls.queries += [{'query_id': 'cold', 'user_id': 'NEW'}, {'query_id': 'anonymous'}]

    def test_partition_invariance_and_legacy_ranks(self):
        expected = self.engine.recommend(self.queries, batch_size=1)
        for size in (7, 32, 128):
            self.assertEqual(expected, self.engine.recommend(self.queries, batch_size=size))
        for query in expected['results']:
            old = self.model.recommend(method='bpr', user_id=query['user_id']
                if query['user_id'] in self.model.user_index else None)
            self.assertEqual([r['item'] for r in old['ranking']], [r['item'] for r in query['ranking']])
            self.assertEqual(old['candidate_count'], query['candidate_count'])

    def test_all_scores_independent_scalar_oracle_and_input_immutable(self):
        before_p = self.model.user_factors.copy()
        before_q = self.model.item_factors.copy()
        got = fixed_order_scores(before_p, before_q)
        for u in range(len(before_p)):
            for i in range(len(before_q)):
                oracle = math.fsum(float(a) * float(b) for a, b in zip(before_p[u], before_q[i]))
                self.assertAlmostEqual(got[u, i], oracle, delta=1e-12)
        self.engine.recommend(self.queries)
        np.testing.assert_array_equal(before_p, self.model.user_factors)
        np.testing.assert_array_equal(before_q, self.model.item_factors)

    def test_exact_and_near_ties_use_declared_order_without_rounding(self):
        engine = copy.deepcopy(self.engine)
        engine.items = ('A', 'B', 'C')
        engine.index = {v: i for i, v in enumerate(engine.items)}
        engine.seen = {'U': frozenset()}
        engine.user_index = {'U': 0}
        engine.trained_users = {'U'}
        engine.user_factors = np.array([[1.0]])
        engine.item_factors = np.array([[1.0], [1.0], [1.0 + 1e-10]])
        engine.popularity = np.array([2.0, 2.0, 1.0])
        got = engine.recommend([{'query_id': 'tie', 'user_id': 'U'}])['results'][0]['ranking']
        self.assertEqual([r['item'] for r in got], ['C', 'A', 'B'])
        self.assertGreater(got[0]['score'], got[1]['score'])
        engine.popularity[1] = 3
        got = engine.recommend([{'query_id': 'tie', 'user_id': 'U'}])['results'][0]['ranking']
        self.assertEqual([r['item'] for r in got], ['C', 'B', 'A'])

    def test_filtering_excludes_every_seen_and_extra_exclusion(self):
        query = {'query_id': 'q', 'user_id': 'D00', 'exclude': ['S01', 'S01']}
        got = self.engine.recommend([query])['results'][0]
        allowed = set(self.model.items) - self.model.seen['D00'] - {'S01'}
        self.assertEqual(got['candidate_count'], len(allowed))
        self.assertTrue({r['item'] for r in got['ranking']} <= allowed)
        self.assertEqual(len(got['ranking']), len({r['item'] for r in got['ranking']}))

    def test_unknown_and_no_positive_profiles_have_explicit_fallback(self):
        model = BPRRecommender([Row('U', 'A', 2, 1), Row('V', 'B', 5, 1)], epochs=2)
        engine = BatchServing(model)
        got = engine.recommend([{'query_id': 'known', 'user_id': 'U'},
            {'query_id': 'unknown', 'user_id': 'NEW'}])
        self.assertTrue(all(q['fallback'] for q in got['results']))
        self.assertTrue(all(q['effective_method'] == 'popularity' for q in got['results']))
        self.assertNotEqual(got['results'][0]['fallback_reason'], got['results'][1]['fallback_reason'])

    def test_empty_candidate_status_survives_csv_and_json(self):
        got = self.engine.recommend([{'query_id': 'empty', 'exclude': list(self.engine.items)}])
        query = got['results'][0]
        self.assertEqual(query['status'], 'no_candidates')
        self.assertEqual(query['ranking'], [])
        rows = list(csv.DictReader(io.StringIO(batch_csv(got))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['row_type'], 'query_status')
        self.assertEqual(rows[0]['query_id'], 'empty')
        self.assertEqual(json.loads(json.dumps(got)), got)

    def test_csv_preserves_status_context_order_and_exact_floats(self):
        got = self.engine.recommend(self.queries, n=4)
        rows = list(csv.DictReader(io.StringIO(batch_csv(got))))
        statuses = [r for r in rows if r['row_type'] == 'query_status']
        self.assertEqual([r['query_id'] for r in statuses], [q['query_id'] for q in self.queries])
        for query in got['results']:
            exported = [r for r in rows if r['row_type'] == 'ranking' and r['query_id'] == query['query_id']]
            self.assertEqual(len(exported), len(query['ranking']))
            for a, b in zip(exported, query['ranking']):
                self.assertEqual(a['item'], b['item'])
                self.assertEqual(float(a['score']), b['score'])
                self.assertEqual(int(a['rank']), b['rank'])
                self.assertEqual(a['model_fingerprint'], got['model_fingerprint'])

    def test_validation_rejects_entire_bad_batch_before_scoring(self):
        bad = [[], {}, [None], [{'query_id': 'x'}, {'query_id': 'x'}],
               [{'query_id': 'x', 'user_id': []}], [{'query_id': 'x', 'exclude': ['UNKNOWN']}],
               [{'query_id': 'x', 'liked': ['S01']}], [{'query_id': 'x', 'exclude': 'S01'}],
               [{'query_id': '=SUM(A1)'}], [{'query_id': 'x'}] * 1001]
        for queries in bad:
            with self.subTest(queries=str(queries)[:50]), self.assertRaises(ValueError):
                self.engine.recommend(queries)
        for option in ({'n': True}, {'batch_size': True}, {'batch_size': 129}, {'n': 0}, {'method': 'unknown'}):
            with self.assertRaises(ValueError):
                self.engine.recommend([{'query_id': 'x'}], **option)

    def test_popularity_matches_old_and_snapshot_isolated_from_mutation(self):
        expected = self.engine.recommend(self.queries, method='popularity')
        for query in expected['results']:
            seen = self.model.seen.get(query['user_id'], set())
            old = self.model.recommend(seen=list(seen), method='popularity')
            self.assertEqual(old['ranking'], [dict(r, contributors=[]) for r in query['ranking']])
            self.assertFalse(query['fallback'])
        model = copy.deepcopy(self.model)
        snap = BatchServing(model)
        baseline = snap.recommend(self.queries)
        model.user_factors[:] = 0
        model.item_factors[:] = 0
        model.popularity[:] = 0
        self.assertEqual(baseline, snap.recommend(self.queries))

    def test_bad_factors_and_nonfinite_scores_rejected(self):
        for users, items in [(np.ones(2), np.ones((2, 2))),
                             (np.ones((2, 3)), np.ones((2, 2))),
                             (np.array([[np.nan]]), np.ones((2, 1)))]:
            with self.assertRaises(ValueError):
                fixed_order_scores(users, items)


class BatchAPIAndCLI(unittest.TestCase):
    def test_http_batch_route_and_bad_query(self):
        model = BPRRecommender(synthetic_rows(), epochs=2)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(model, True))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}/api/batch'
        try:
            args = {'queries': [{'query_id': 'first', 'user_id': 'D00'}, {'query_id': 'new', 'user_id': 'NEW'}], 'n': 3}
            request = urllib.request.Request(url, json.dumps(args).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=5) as response:
                got = json.load(response)
            expected = BatchServing(model).recommend(**args)
            self.assertEqual(got.pop('csv'), batch_csv(expected))
            self.assertEqual(got, expected)
            args['queries'][0]['exclude'] = ['UNKNOWN']
            with self.assertRaises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(urllib.request.Request(url, json.dumps(args).encode()), timeout=5)
            self.assertEqual(error.exception.code, 400)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_actual_cli_files_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'batch'
            cmd = [sys.executable, 'batch_cli.py', '--output', str(prefix), '--batch-size', '7', '--n', '3']
            got = subprocess.run(cmd, text=True, capture_output=True, timeout=45)
            self.assertEqual(got.returncode, 0, got.stderr)
            data = json.loads(prefix.with_suffix('.json').read_text(encoding='utf-8'))
            self.assertEqual(data['input_provenance'], 'authored_synthetic_demo')
            self.assertEqual(prefix.with_suffix('.csv').read_text(encoding='utf-8'), batch_csv(data))
            self.assertEqual(data['query_count'], 4)
            before = prefix.with_suffix('.json').read_bytes()
            repeat = subprocess.run(cmd, text=True, capture_output=True, timeout=45)
            self.assertEqual(repeat.returncode, 2)
            self.assertEqual(prefix.with_suffix('.json').read_bytes(), before)
            invalid = subprocess.run([sys.executable, 'batch_cli.py', '--output', str(Path(directory)/'bad'),
                '--batch-size', '0'], text=True, capture_output=True, timeout=45)
            self.assertEqual(invalid.returncode, 2)
            self.assertFalse((Path(directory)/'bad.json').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
