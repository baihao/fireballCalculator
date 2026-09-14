#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""稳健峰值基准与截断自检单元测试。"""

import unittest

from data_filter import apply_data_filter, filter_smoke_interference_data


class RobustDataFilterTest(unittest.TestCase):
    def test_inlier_aware_smooth_not_contaminated_by_spike(self):
        """离群点剔除后，邻域滑动平均不得再被尖峰抬高。"""
        n = 40
        t = [float(i) for i in range(n)]
        D = [70.0] * 12 + [10.0 + 0.2 * i for i in range(n - 12)]
        from data_filter import compute_inlier_mask, _calculate_sliding_average_inlier_aware
        import numpy as np

        inlier, _ = compute_inlier_mask(t, D, window_size=10, warmup_ms=50.0)
        smooth = _calculate_sliding_average_inlier_aware(np.array(D), inlier, 10)
        self.assertTrue(inlier[17])
        self.assertLess(smooth[17], 15.0, msg=f"smooth[17]={smooth[17]}")

    def test_early_spike_rejected_by_sanity(self):
        """开头尖峰不应截断到仅保留少量点。"""
        n = 200
        t = [float(i) for i in range(n)]
        D = [70.0] * 12 + [10.0 + 0.25 * i for i in range(n - 12)]
        ft, fd, stats = apply_data_filter(t, D, drop_threshold=0.02, window_size=10)
        self.assertGreater(stats.get('outliers_removed', 0), 0)
        self.assertEqual(len(ft), n - stats['outliers_removed'])
        self.assertEqual(len(fd), len(ft))

    def test_real_plateau_cutoff_still_works(self):
        """正常膨胀后平台再下降仍应截断（保留率足够）。"""
        t = [float(i) for i in range(300)]
        D = []
        for i in range(300):
            if i < 150:
                D.append(10.0 + i * 0.2)
            else:
                D.append(40.0 - (i - 150) * 0.05)
        cutoffs, _ = filter_smoke_interference_data(
            t, D, drop_threshold=0.02, window_size=10, warmup_ms=30.0
        )
        self.assertTrue(len(cutoffs) > 0)
        ft, _, _ = apply_data_filter(t, D, drop_threshold=0.02, window_size=10, warmup_ms=30.0)
        self.assertLess(len(ft), len(t))
        self.assertGreater(len(ft), 50)


if __name__ == "__main__":
    unittest.main()
