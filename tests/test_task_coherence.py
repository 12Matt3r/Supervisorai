import unittest
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from task_coherence.coherence_analyzer import CoherenceAnalyzer

class TestCoherenceAnalyzer(unittest.TestCase):
    """Test suite for the CoherenceAnalyzer class."""

    def setUp(self):
        """Set up a new CoherenceAnalyzer for each test."""
        self.analyzer = CoherenceAnalyzer()

    def test_coherent_output(self):
        """Test an output that is coherent with its goals."""
        output = "This is a summary of the quarterly financial report, focusing on profits and losses."
        goals = ["summarize the financial report", "discuss profits"]
        analysis = self.analyzer.analyze(output, goals)
        self.assertTrue(analysis["is_coherent"])
        self.assertLess(analysis["drift_score"], 0.5)

    def test_drift_pattern(self):
        """Test an output that contains a clear drift pattern."""
        output = "I was going to summarize the report, but let's focus on a different approach and talk about marketing strategies."
        goals = ["summarize the financial report"]
        analysis = self.analyzer.analyze(output, goals)
        self.assertFalse(analysis["is_coherent"])
        self.assertGreater(analysis["drift_score"], 0.3)
        self.assertTrue(any("topic shift" in f for f in analysis["findings"]))

    def test_distraction_keyword(self):
        """Test an output that contains a common distraction."""
        output = "Instead of the report, let me tell me a joke."
        goals = ["summarize the financial report"]
        analysis = self.analyzer.analyze(output, goals)
        self.assertFalse(analysis["is_coherent"])
        self.assertGreater(analysis["drift_score"], 0.2)
        self.assertTrue(any("distraction" in f for f in analysis["findings"]))

    def test_low_keyword_alignment(self):
        """Test an output with very few keywords related to the goal."""
        output = "The sky is blue and the clouds are white today."
        goals = ["summarize the quarterly financial report about profits"]
        analysis = self.analyzer.analyze(output, goals)
        self.assertFalse(analysis["is_coherent"])
        self.assertGreater(analysis["drift_score"], 0.4)
        self.assertTrue(any("keyword alignment" in f for f in analysis["findings"]))

if __name__ == '__main__':
    unittest.main()
