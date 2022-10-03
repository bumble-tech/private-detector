"""
Config module for efficientnetv2, taken mostly from
https://github.com/google/automl/tree/master/efficientnetv2
and modified to not have as much noise
"""
import re
from collections import namedtuple
from typing import List

from .hparams import Config


class EfficientNetV2Config(Config):
    """
    Wrapper class for efficientnetv2 config

    Parameters
    ----------
    num_classes : int
        Number of classes to use for model

    Notes
    -----
    efficientnet-b4 config used here
    """
    def __init__(self, num_classes: int):
        BLOCK = [
            'r3_k3_s1_e1_i24_o24_c1',
            'r5_k3_s2_e4_i24_o48_c1',
            'r5_k3_s2_e4_i48_o80_c1',
            'r7_k3_s2_e4_i80_o160_se0.25',
            'r14_k3_s1_e6_i160_o176_se0.25',
            'r18_k3_s2_e6_i176_o304_se0.25',
            'r5_k3_s1_e6_i304_o512_se0.25',
        ]
        WIDTH = 1.0
        DEPTH = 1.0
        TRAIN_SIZE = 384
        EVAL_SIZE = 480
        DROPOUT = 0.3
        RANDAUG = 15
        MIX = 0.2
        AUG = 'randaug'

        super().__init__(
            model={
                'model_name': 'efficientnetv2-m',
                'blocks_args': self.decode(BLOCK),
                'width_coefficient': WIDTH,
                'depth_coefficient': DEPTH,
                'dropout_rate': DROPOUT,
                'num_classes': num_classes
            },
            train={
                'isize': TRAIN_SIZE,
                'stages': 4,
                'sched': True
            },
            eval={
                'isize': EVAL_SIZE
            },
            data={
                'augname': AUG,
                'ram': RANDAUG,
                'mixup_alpha': MIX,
                'cutmix_alpha': MIX
            }

        )

    def decode(self, string_list: List[str]) -> List[namedtuple]:
        """
        Decodes a list of string notations to specify blocks inside the network

        Parameters
        ----------
        string_list : List[str]
            Each string is a notation of blocks

        Returns
        -------
        blocks_args : List[namedtuple]
            A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)

        blocks_args = []

        for block_string in string_list:
            blocks_args.append(
                self._decode_block_string(block_string)
            )
        return blocks_args

    def _decode_block_string(self, block_string) -> Config:
        """
        Gets a block through a string notation of arguments

        Parameters
        ----------
        block_string : str
            Block string to decode

        Return
        ------
        decoded_config : Config
            Decoded config based on string input
        """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}

        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        decoded_config = Config(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=int(options['s']),
            conv_type=int(options['c']) if 'c' in options else 0,
        )

        return decoded_config
