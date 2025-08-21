import unittest
import sys
import os

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from idea_validation.validator import Validator
from idea_validation.data_models import Idea

class TestIdeaValidator(unittest.TestCase):
    """Test suite for the Validator class."""

    def setUp(self):
        """Set up a new Validator for each test."""
        self.validator = Validator()

    def test_good_idea(self):
        """Test a good, low-risk idea."""
        idea = Idea(
            description="A simple to-do list application using Python and Flask.",
            required_skills=["python"],
            market_niche="productivity tools",
            estimated_time_hours=20
        )
        report = self.validator.validate(idea)
        self.assertEqual(len(report.findings), 0)
        self.assertEqual(report.overall_score, 1.0)
        self.assertIn("highly viable", report.summary)

    def test_saturated_market_idea(self):
        """Test an idea in a known saturated market."""
        idea = Idea(
            description="A new photo sharing social media app.",
            required_skills=["python", "javascript"],
            market_niche="social media",
            estimated_time_hours=80
        )
        report = self.validator.validate(idea)
        self.assertTrue(any(f.category == "Market Viability" for f in report.findings))
        self.assertLess(report.overall_score, 1.0)

    def test_technically_complex_idea(self):
        """Test an idea with high technical complexity."""
        idea = Idea(
            description="A new real-time multiplayer game using a custom blockchain.",
            required_skills=["python"],
            market_niche="gaming",
            estimated_time_hours=200
        )
        report = self.validator.validate(idea)
        self.assertTrue(any(f.category == "Technical Feasibility" for f in report.findings))
        self.assertLess(report.overall_score, 1.0)

    def test_missing_skills_idea(self):
        """Test an idea that requires unavailable skills."""
        idea = Idea(
            description="A mobile app using Swift.",
            required_skills=["swift", "ios development"],
            market_niche="mobile apps",
            estimated_time_hours=50
        )
        report = self.validator.validate(idea)
        self.assertTrue(any(f.category == "Resource Mismatch" for f in report.findings))
        self.assertLess(report.overall_score, 1.0)

if __name__ == '__main__':
    unittest.main()
