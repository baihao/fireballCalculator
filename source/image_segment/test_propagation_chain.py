#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多参考点生成单元测试（不加载 SAM）。"""

import unittest

import numpy as np

from prompt_generation import PromptPointGenerator


class MultiReferencePromptTest(unittest.TestCase):
    def test_background_intersection(self):
        gen = PromptPointGenerator()
        h, w = 20, 20
        m0 = np.zeros((h, w), dtype=np.uint8)
        m0[5:15, 5:15] = 1
        m1 = np.zeros((h, w), dtype=np.uint8)
        m1[6:14, 6:14] = 1
        bg = gen.compute_background_mask_intersection([m0, m1], (h, w))
        self.assertTrue(bg[0, 0])
        self.assertFalse(bg[10, 10])

    def test_multi_reference_uses_cached_positive_without_resample(self):
        gen = PromptPointGenerator()
        h, w = 32, 32
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :] = (40, 40, 40)
        mask = np.zeros((h, w), dtype=bool)
        mask[8:24, 8:24] = True
        img[mask] = (200, 100, 50)

        pos_a = [(10, 10), (12, 12)]
        pos_b = [(14, 14)]
        entries = [
            {'idx': 0, 'image_rgb': img, 'mask': mask, 'positive_points': pos_a},
            {'idx': 1, 'image_rgb': img, 'mask': mask, 'positive_points': pos_b},
        ]
        points, labels, dbg = gen.generate_points_multi_reference(
            reference_entries=entries,
            primary_entry=entries[0],
            target_image=img,
            num_positive=2,
            num_negative=2,
            return_debug_info=True,
        )
        self.assertGreaterEqual(len(points), 1)
        self.assertEqual(len(dbg['reference_positive']), 3)


if __name__ == '__main__':
    unittest.main()
