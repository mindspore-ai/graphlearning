import os
import sys
import unittest

sys.path.append('../..')

from gnnvis import (TestUtils,
                    TestGnnVisNodeClassification,
                    TestGnnVisLinkPrediction,
                    TestGnnVisGraphClassification)
from custom_index import TestCustomIndex

TEST_TEMP_FILE_PATH = "test_temp_file"


def check_path():
    if not os.path.exists(TEST_TEMP_FILE_PATH):
        os.makedirs(TEST_TEMP_FILE_PATH)


if __name__ == '__main__':
    check_path()

    test_case_list = [
        TestUtils,
        TestGnnVisNodeClassification,
        TestGnnVisLinkPrediction,
        TestGnnVisGraphClassification,
        TestCustomIndex
    ]

    for test_case_proto in test_case_list:
        test_case_proto.save_path = TEST_TEMP_FILE_PATH

    suite = unittest.TestSuite()
    tests = [unittest.makeSuite(test_case_proto) for test_case_proto in test_case_list]

    suite.addTests(tests)

    unittest.TextTestRunner(verbosity=2).run(suite)
