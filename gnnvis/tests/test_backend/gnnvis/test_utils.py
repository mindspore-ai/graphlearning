import os
import unittest
from backend import Utils
from example_data import mini_dict, mini_ndarray


class TestUtils(unittest.TestCase):
    save_path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_path = os.path.join(TestUtils.save_path, "TestUtils")
        self.init()

    def init(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        Utils.export_dict_to_json(mini_dict, self.save_path, "Utils_output_dict.json")
        self.json_file_path = os.path.join(self.save_path, "Utils_output_dict.json")

        Utils.export_numpy_to_csv(mini_ndarray, self.save_path,
                                  "Utils_output_ndarray.csv", ["t1", "t2", "t3"])
        self.csv_file_path = os.path.join(self.save_path, "Utils_output_ndarray.csv")

    def test_json_file(self):
        self.assertTrue(os.path.exists(self.json_file_path))

    def test_csv_file(self):
        self.assertTrue(os.path.exists(self.csv_file_path))
