#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""传播链选取与多参考点生成单元测试（不加载 SAM）。"""

import unittest

import numpy as np

from prompt_generation import PromptPointGenerator


class PropagationChainRefsTest(unittest.TestCase):
    """与 IterativeMaskPropagationSegmenter._get_propagation_chain_refs 同逻辑。"""

    PROPAGATION_CHAIN_LENGTH = 5

    def _chain(self, ref_idx, target_idx, processed, n_images=20):
        processed_indices = set(processed)
        all_masks = [object() if i in processed_indices else None for i in range(n_images)]

        k = self.PROPAGATION_CHAIN_LENGTH
        if ref_idx not in processed_indices or all_masks[ref_idx] is None:
            return []
        if target_idx == ref_idx:
            return [ref_idx]
        step = -1 if target_idx > ref_idx else 1
        chain = []
        cur = ref_idx
        while len(chain) < k:
            if cur < 0 or cur >= n_images:
                break
            if cur not in processed_indices or all_masks[cur] is None:
                if cur != ref_idx:
                    break
            else:
                chain.append(cur)
            nxt = cur + step
            if nxt < 0 or nxt >= n_images:
                break
            if nxt not in processed_indices or all_masks[nxt] is None:
                break
            cur = nxt
        return sorted(chain)

    def test_forward_propagation_chain(self):
        processed = list(range(0, 8))
        self.assertEqual(self._chain(7, 8, processed), [3, 4, 5, 6, 7])

    def test_backward_propagation_chain(self):
        processed = list(range(0, 14))
        self.assertEqual(self._chain(9, 8, processed), [9, 10, 11, 12, 13])

    def test_short_chain(self):
        processed = [3, 4]
        self.assertEqual(self._chain(4, 5, processed), [3, 4])

    def test_gap_stops_chain(self):
        processed = [3, 5, 6, 7]
        self.assertEqual(self._chain(7, 8, processed), [5, 6, 7])


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
