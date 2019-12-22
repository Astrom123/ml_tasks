import unittest
from humor_classifier import HumorClassifier


class TestHumorClf(unittest.TestCase):

    def setUp(self):
        self.humor_clf = HumorClassifier()
        joke = "Колобок повесился"
        self.is_funny = self.humor_clf.is_humorous(joke)

    def test_unclean_jokes(self):
        jokes = ["Колобок повесился!",
                 "{Колобок повесился...}",
                 "-Колобок: повесился???",
                 """Колобок повесился???
                 ..."""]

        is_funny = [self.humor_clf.is_humorous(joke) for joke in jokes]
        for res in is_funny:
            self.assertEqual(res, self.is_funny)

    def test_empty_joke(self):
        joke = ""

        is_funny = self.humor_clf.is_humorous(joke)
        self.assertIsNotNone(is_funny)


if __name__ == '__main__':
    unittest.main()
