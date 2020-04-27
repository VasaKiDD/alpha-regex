import glob
import numpy as np
import re
import time
import math


class MalwareEvaluator:
    def __init__(self, good_example_path, bad_example_path):
        self.goods = glob.glob(good_example_path)
        self.bads = glob.glob(bad_example_path)
        self.good_requests = []
        self.bad_requests = []

    def reset(self, good_examples, bad_examples):
        good_limit = min(good_examples, len(self.goods))
        bad_limit = min(bad_examples, len(self.bads))
        good_batch = np.random.choice(np.arange(len(self.goods)), good_limit, replace=False)
        bad_batch = np.random.choice(np.arange(len(self.bads)), bad_limit, replace=False)

        self.good_requests = []
        for i in good_batch:
            file = open(self.goods[i], "br")
            self.good_requests.append(file.read())
            file.close()

        self.bad_requests = []
        for i in bad_batch:
            file = open(self.bads[i], "br")
            self.bad_requests.append(file.read())
            file.close()

    def evaluate(self, regex, _print=False):

        try:
            rule = re.compile(regex)
        except:
            print(regex)
            import pdb

            pdb.set_trace()

        b = time.time()
        false_positives = 0
        for string in self.good_requests:
            res = rule.search(string)
            if res:
                false_positives += 1.0
        true_negatives = len(self.good_requests) - false_positives

        true_positives = 0
        for string in self.bad_requests:
            res = rule.search(string)
            if res:
                true_positives += 1.0
        f = time.time()
        false_negatives = len(self.bad_requests) - true_positives

        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / len(self.bad_requests)

        # F1_score = 2 * ((precision * recall) / (precision + recall))

        MCC1 = (true_positives * true_negatives) - (false_positives * false_negatives)
        MCC2 = math.sqrt(
            (true_positives + false_positives)
            * (true_positives + false_negatives)
            * (true_negatives + false_positives)
            * (true_negatives + false_negatives)
        )
        MCC = MCC1 / MCC2 if MCC2 > 0 else 0.0

        # print(
        #     "Regex :",
        #     regex,
        #     "F1 score :",
        #     F1_score,
        #     "Precision :",
        #     precision,
        #     "Recall :",
        #     recall,
        #     "true_positives :",
        #     true_positives,
        #     "false positives :",
        #     false_positives,
        #     "time :",
        #     f - b,
        # )

        if _print:
            print(
                "Regex :",
                regex,
                "MCC :",
                MCC,
                "true_positives :",
                true_positives,
                "false positives :",
                false_positives,
                "time :",
                f - b,
            )

        return MCC


class RequestEvaluator:
    def __init__(self, good_file, bad_file):
        good_file = open(good_file, "rb")
        goods = good_file.readlines()
        goods = [line.strip() for line in goods]
        good_file.close()

        bad_file = open(bad_file, "rb")
        bads = bad_file.readlines()
        bads = [line.strip() for line in bads]

        self.good_requests = goods
        self.bad_requests = bads

        print("Good requests :", len(goods))
        print("Bad requests :", len(bads))

    def _evaluate(self, regex, _print=False):

        try:
            rule = re.compile(regex)
        except:
            print(regex)
            import pdb

            pdb.set_trace()

        b = time.time()
        false_positives = 0
        for string in self.good_requests:
            res = rule.search(string)
            if res:
                false_positives += 1.0
        true_negatives = len(self.good_requests) - false_positives

        true_positives = 0
        for string in self.bad_requests:
            res = rule.search(string)
            if res:
                true_positives += 1.0
        f = time.time()
        false_negatives = len(self.bad_requests) - true_positives

        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / len(self.bad_requests)

        # F1_score = 2 * ((precision * recall) / (precision + recall))

        MCC1 = (true_positives * true_negatives) - (false_positives * false_negatives)
        MCC2 = math.sqrt(
            (true_positives + false_positives)
            * (true_positives + false_negatives)
            * (true_negatives + false_positives)
            * (true_negatives + false_negatives)
        )
        MCC = MCC1 / MCC2 if MCC2 > 0 else 0.0

        # print(
        #     "Regex :",
        #     regex,
        #     "F1 score :",
        #     F1_score,
        #     "Precision :",
        #     precision,
        #     "Recall :",
        #     recall,
        #     "true_positives :",
        #     true_positives,
        #     "false positives :",
        #     false_positives,
        #     "time :",
        #     f - b,
        # )

        if _print:
            print(
                "Regex :",
                regex,
                "MCC :",
                MCC,
                "true_positives :",
                true_positives,
                "false positives :",
                false_positives,
                "time :",
                f - b,
            )

        return MCC

    def evaluate(self, regex, _print=False):

        try:
            rule = re.compile(regex)
        except Exception as e:
            print(regex, e)
            import pdb

            pdb.set_trace()

        b = time.time()
        false_positives = 0
        for string in self.good_requests:
            res = rule.search(string)
            if res:
                false_positives += 1.0
        true_negatives = len(self.good_requests) - false_positives

        true_positives = 0
        for string in self.bad_requests:
            res = rule.search(string)
            if res:
                true_positives += 1.0
        f = time.time()
        false_negatives = len(self.bad_requests) - true_positives

        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / len(self.bad_requests)

        # F1_score = 2 * ((precision * recall) / (precision + recall))

        MCC1 = (true_positives * true_negatives) - (false_positives * false_negatives)
        MCC2 = math.sqrt(
            (true_positives + false_positives)
            * (true_positives + false_negatives)
            * (true_negatives + false_positives)
            * (true_negatives + false_negatives)
        )
        MCC = MCC1 / MCC2 if MCC2 > 0 else 0.0

        # print(
        #     "Regex :",
        #     regex,
        #     "F1 score :",
        #     F1_score,
        #     "Precision :",
        #     precision,
        #     "Recall :",
        #     recall,
        #     "true_positives :",
        #     true_positives,
        #     "false positives :",
        #     false_positives,
        #     "time :",
        #     f - b,
        # )

        if _print:
            print(
                "Regex :",
                regex,
                "MCC :",
                MCC,
                "true_positives :",
                true_positives,
                "false positives :",
                false_positives,
                "time :",
                f - b,
            )

        return MCC
