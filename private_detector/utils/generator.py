"""
Generator class to generate images for training the private detector
"""
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


class Generator:
    """
    Generator for image data

    Parameters
    ----------
    classes_files : List[str]
        Files with paths to images for each class
    seed : int
        Random seed to use
    sequential : bool
        Batch classes sequentially or not
    """
    def __init__(self,
                 classes_files: List[str],
                 seed: int = None,
                 sequential: bool = False):
        self.sequential = sequential

        self.classes = defaultdict(list)
        self.counts = {}

        np.random.seed(seed)

        self.labels = {}
        self.keys = {}

        if isinstance(classes_files, str):
            classes_files = [classes_files]

        for classes_file in classes_files:
            with open(classes_file, 'r') as f:
                js = json.load(f)

            for class_name, obj in js.items():
                path = obj['path']
                label = int(obj['label'])

                with open(path, 'r') as f:
                    filenames = f.read().splitlines()

                self.classes[class_name] += list(filenames)

                if class_name not in self.keys:
                    self.keys[label] = class_name
                    self.labels[class_name] = label
                else:
                    saved_label = self.labels[class_name]
                    if label != saved_label:
                        raise(
                            Exception(
                                f'Class label did not match : {class_name}, {label}, {saved_label}'
                            )
                        )

        new_classes = {}

        for class_name, filenames in self.classes.items():
            new_classes[class_name] = np.array(filenames)
            self.counts[class_name] = [1.] * len(filenames)

        self.classes = new_classes

        new_keys = []
        for i in range(len(self.keys)):
            class_name = self.keys[i]
            new_keys.append(class_name)

        self.keys = new_keys

        if sequential:
            self.flat_pairs = []
            for class_name, filenames in self.classes.items():
                label = self.labels[class_name]

                self.flat_pairs += [(filename, label) for filename in filenames]

            np.random.shuffle(self.flat_pairs)
            self.flat_index = 0

    def num_images(self) -> int:
        """
        Return total number of images in the dataset

        Returns
        -------
        num_images : int
            Total number of images in the dataset
        """
        num_images = 0
        for _, class_images in self.classes.items():
            num_images += len(class_images)

        return num_images

    def class_labels(self) -> Dict[str, int]:
        """
        Return names of classes and their corresponding label

        Returns
        -------
        class_labels : Dict[str, int]
            Names of classes and their corresponding label
        """
        class_labels = self.keys

        return class_labels

    def num_classes(self) -> int:
        """
        Return number of classes

        Returns
        -------
        num_classes : int
            Number of classes in dataset
        """
        num_classes = len(self.labels)

        return num_classes

    def num_files(self, class_name: str) -> int:
        """
        Return number of images in a given class

        Parameters
        ----------
        class_name : str
            Name of class to get files for

        Returns
        -------
        num_files : int
            Number of images in a given class
        """
        num_files = len(self.classes[class_name])

        return num_files

    def get_sequential(self,
                       num: int = None,
                       want_full: bool = False) -> Tuple[List[str], List[int]]:
        """
        Get image paths/labels, but parse sequentially per class

        Parameters
        ----------
        num : int
            Number of records to get if want_full is False
        want_full : bool
            Get full dataset or not

        Returns
        -------
        paths : List[str]
            List of paths to images
        labels : List[str]
            List of labels corresponding to each path
        """
        paths = []
        labels = []

        if want_full:
            pairs = self.flat_pairs
        else:
            pairs = []
            while len(pairs) < num:
                pairs += self.flat_pairs[self.flat_index: self.flat_index + num]
                self.flat_index += num
                if self.flat_index >= len(self.flat_pairs):
                    self.flat_index = 0

        for p, l in pairs:
            paths.append(p)
            labels.append(l)

        return paths, labels

    def get(self,
            num: int = None,
            want_full: bool = False) -> Tuple[List[str], List[int]]:
        """
        Get image paths/labels, but don't necessarily parse sequentially

        Parameters
        ----------
        num : int
            Number of records to get if want_full is False
        want_full : bool
            Get full dataset or not

        Returns
        -------
        paths : List[str]
            List of paths to images
        labels : List[str]
            List of labels corresponding to each path
        """
        if self.sequential:
            return self.get_sequential(num, want_full)

        pairs = []

        if want_full:
            for class_name, filenames in self.classes.items():
                label = self.labels[class_name]
                indexes = np.arange(len(filenames))

                for fn, idx in zip(filenames, indexes):
                    pairs.append((fn, label))
        else:
            class_names = np.random.choice(self.keys, num, replace=True)
            class_name_nums = defaultdict(int)
            for class_name in class_names:
                class_name_nums[class_name] += 1

            pairs = []
            for class_name, class_name_num in class_name_nums.items():
                label = self.labels[class_name]
                filenames = self.classes[class_name]
                counts = self.counts[class_name]
                probs = counts / np.sum(counts)

                num = min(len(filenames), class_name_num)

                indexes = np.arange(len(filenames))
                indexes = np.random.choice(indexes, num, replace=False, p=probs)

                filenames = filenames[indexes]

                for fn, idx in zip(filenames, indexes):
                    pairs.append((fn, label))

        np.random.shuffle(pairs)

        paths = []
        labels = []

        for p, l in pairs:
            paths.append(p)
            labels.append(l)

        return paths, labels
