import unittest

from calc import clamp, mean


class MeanTests(unittest.TestCase):
    def test_mean_of_a_list(self):
        self.assertEqual(mean([2, 4, 6]), 4)

    def test_mean_of_zeros(self):
        self.assertEqual(mean([0, 0, 0]), 0)


class ClampTests(unittest.TestCase):
    def test_clamp_inside_the_range(self):
        self.assertEqual(clamp(5, 0, 10), 5)

    def test_clamp_above_the_range(self):
        self.assertEqual(clamp(15, 0, 10), 10)


if __name__ == "__main__":
    unittest.main()
