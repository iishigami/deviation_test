import unittest
import pandas as pd
import numpy as np
import os
from plotter import Plotter

class TestPlotter(unittest.TestCase):

    def setUp(self):
        self.data = pd.read_json('deviation.json')
        self.plotter = Plotter('deviation.json')

    def test_evaluate_model_performance(self):
        metrics = self.plotter.evaluate_model_performance()
        gt_corners = self.data['gt_corners'].values
        rb_corners = self.data['rb_corners'].values

        expected_mae = np.mean(np.abs(gt_corners - rb_corners))
        expected_mse = np.mean((gt_corners - rb_corners) ** 2)
        expected_rmse = np.sqrt(expected_mse)

        self.assertAlmostEqual(metrics['MAE'], expected_mae, places=2)
        self.assertAlmostEqual(metrics['MSE'], expected_mse, places=2)
        self.assertAlmostEqual(metrics['RMSE'], expected_rmse, places=2)

    def test_draw_plots(self):
        plot_paths = self.plotter.draw_plots()
        self.assertGreater(len(plot_paths), 0)
        for path in plot_paths:
            self.assertTrue(os.path.exists(path))

if __name__ == '__main__':
    unittest.main()
