"""
Unit tests for the baseline RAG pipeline.
Run with: python test_baseline.py
Or run a specific test: python test_baseline.py TestChunking
"""

import unittest


# CHUNKING TESTS
class TestChunking(unittest.TestCase):

    def setUp(self):
        from chunking import chunk_text
        self.chunk_text = chunk_text
        # 600 words sample text
        self.sample = " ".join([f"word{i}" for i in range(600)])

    def test_basic_chunking(self):
        chunks = self.chunk_text(self.sample, chunk_size=256, overlap=32)
        self.assertGreater(len(chunks), 1)
        print(f"  ✓ basic chunking: {len(chunks)} chunks")

    def test_chunk_size(self):
        chunks = self.chunk_text(self.sample, chunk_size=256, overlap=32)
        for chunk in chunks[:-1]:  # last chunk can be smaller
            self.assertLessEqual(len(chunk.split()), 256)
        print(f"  ✓ chunk size respected")

    def test_overlap(self):
        chunks = self.chunk_text(self.sample, chunk_size=256, overlap=32)
        # last words of chunk 0 should appear at start of chunk 1
        last_words = chunks[0].split()[-32:]
        first_words = chunks[1].split()[:32]
        self.assertEqual(last_words, first_words)
        print(f"  ✓ overlap correct")

    def test_short_text(self):
        # text shorter than chunk_size → should return single chunk
        short = "hello world this is a short text"
        chunks = self.chunk_text(short, chunk_size=256, overlap=32)
        self.assertEqual(len(chunks), 1)
        print(f"  ✓ short text handled")

    def test_empty_text(self):
        chunks = self.chunk_text("", chunk_size=256, overlap=32)
        self.assertEqual(len(chunks), 0)
        print(f"  ✓ empty text handled")


# RETRIEVER TESTS
class TestRetriever(unittest.TestCase):

    def setUp(self):
        from retriever import Retriever
        self.retriever = Retriever()
        self.sample_chunks = [
            "The Eiffel Tower was built in 1889 in Paris France.",
            "The moon landing happened on July 20 1969.",
            "Python is a programming language created by Guido van Rossum.",
            "The Great Wall of China was built over many centuries.",
            "Albert Einstein developed the theory of relativity."
        ]
        self.retriever.build_index(self.sample_chunks)

    def test_index_built(self):
        self.assertIsNotNone(self.retriever.index)
        self.assertEqual(self.retriever.index.ntotal, len(self.sample_chunks))
        print(f"  ✓ index built with {self.retriever.index.ntotal} vectors")

    def test_retrieve_returns_k(self):
        results = self.retriever.retrieve("Eiffel Tower", k=3)
        self.assertEqual(len(results), 3)
        print(f"  ✓ retrieve returns correct k")

    def test_retrieve_relevance(self):
        # most relevant chunk for this query should be about Eiffel Tower
        results = self.retriever.retrieve("When was the Eiffel Tower built?", k=1)
        self.assertIn("Eiffel", results[0])
        print(f"  ✓ retrieval relevance: '{results[0][:50]}...'")

    def test_retrieve_without_index(self):
        from retriever import Retriever
        r = Retriever()
        with self.assertRaises(RuntimeError):
            r.retrieve("test query")
        print(f"  ✓ error raised when index not built")

    def test_save_load_index(self):
        import os
        import shutil
        path = "/tmp/test_index"
        os.makedirs(path, exist_ok=True)

        # save
        self.retriever.save_index(path)
        self.assertTrue(os.path.exists(f"{path}/faiss.index"))

        # load into new retriever
        from retriever import Retriever
        new_retriever = Retriever()
        new_retriever.load_index(path)
        self.assertEqual(
            new_retriever.index.ntotal,
            self.retriever.index.ntotal
        )
        print(f"  ✓ save/load index works")

        # cleanup
        shutil.rmtree(path)


# EVALUATE TESTS
class TestEvaluate(unittest.TestCase):

    def setUp(self):
        from evaluate import exact_match, rouge_l, evaluate
        self.exact_match = exact_match
        self.rouge_l = rouge_l
        self.evaluate = evaluate

    def test_exact_match_correct(self):
        score = self.exact_match("1889", ["1889", "in 1889"])
        self.assertEqual(score, 1)
        print(f"  ✓ exact match correct answer")

    def test_exact_match_wrong(self):
        score = self.exact_match("around 1889", ["1889"])
        self.assertEqual(score, 0)
        print(f"  ✓ exact match wrong answer")

    def test_exact_match_case_insensitive(self):
        score = self.exact_match("PARIS", ["paris"])
        self.assertEqual(score, 1)
        print(f"  ✓ exact match case insensitive")

    def test_rouge_l_perfect(self):
        score = self.rouge_l("1889", ["1889"])
        self.assertAlmostEqual(score, 1.0)
        print(f"  ✓ rouge_l perfect score: {score:.4f}")

    def test_rouge_l_partial(self):
        score = self.rouge_l("built in 1889", ["1889"])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)
        print(f"  ✓ rouge_l partial score: {score:.4f}")

    def test_rouge_l_zero(self):
        score = self.rouge_l("completely wrong answer", ["1889"])
        self.assertAlmostEqual(score, 0.0, places=1)
        print(f"  ✓ rouge_l zero score: {score:.4f}")

    def test_evaluate_aggregate(self):
        results = [
            {
                "question": "q1",
                "prediction": "1889",
                "gold_answers": ["1889"],
                "chunks": []
            },
            {
                "question": "q2",
                "prediction": "wrong",
                "gold_answers": ["1969"],
                "chunks": []
            }
        ]
        metrics = self.evaluate(results)
        self.assertEqual(metrics["exact_match"], 0.5)
        self.assertEqual(metrics["n_samples"], 2)
        print(f"  ✓ aggregate metrics: {metrics}")

    def test_evaluate_empty(self):
        with self.assertRaises(ValueError):
            self.evaluate([])
        print(f"  ✓ empty results raises error")


# GENERATOR TESTS (lightweight — no full model load)
class TestGenerator(unittest.TestCase):

    def setUp(self):
        from generator import Generator
        self.generator = Generator()

    def test_prompt_format(self):
        question = "When was the Eiffel Tower built?"
        chunks = ["The Eiffel Tower was built in 1889."]
        prompt = self.generator.build_prompt(question, chunks)
        self.assertIn("[INST]", prompt)
        self.assertIn(question, prompt)
        self.assertIn(chunks[0], prompt)
        print(f"  ✓ prompt format correct")

    def test_prompt_multiple_chunks(self):
        question = "test question"
        chunks = ["chunk 1", "chunk 2", "chunk 3"]
        prompt = self.generator.build_prompt(question, chunks)
        self.assertIn("Passage 1", prompt)
        self.assertIn("Passage 2", prompt)
        self.assertIn("Passage 3", prompt)
        print(f"  ✓ multiple chunks in prompt")

    def test_generate_returns_string(self):
        question = "When was the Eiffel Tower built?"
        chunks = ["The Eiffel Tower was built in 1889 in Paris."]
        answer = self.generator.generate(question, chunks)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)
        print(f"  ✓ generate returns string: '{answer}'")


# MAIN
if __name__ == "__main__":
    print("\n" + "="*50)
    print("BASELINE TESTS")
    print("="*50)

    # run all tests with verbose output
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # add test classes in order
    suite.addTests(loader.loadTestsFromTestCase(TestChunking))
    suite.addTests(loader.loadTestsFromTestCase(TestRetriever))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluate))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerator))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)