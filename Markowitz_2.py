"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys
import math


"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]


"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:


    def __init__(self, price, exclude, lookback=50, gamma=0,
                 target_vol=0.10, topk=6, cap=0.20):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback      
        self.gamma = gamma            
        self.target_vol = target_vol
        self.topk = topk
        self.cap = cap

    def calculate_weights(self):

        assets = self.price.columns[self.price.columns != self.exclude]
        dates = self.price.index

        self.portfolio_weights = pd.DataFrame(
            0.0, index=dates, columns=self.price.columns
        )

        short_w = 21
        mid_w = 63
        long_w = 252
        ma_window = 200
        vol_shrink = 0.5
        eps = 1e-8
        start_idx = max(long_w, self.lookback, ma_window)

        for i in range(start_idx, len(dates)):
            date = dates[i]

            
            R_short = self.returns[assets].iloc[i - short_w:i].mean()
            R_mid = self.returns[assets].iloc[i - mid_w:i].mean()
            R_long = self.returns[assets].iloc[max(0, i - long_w):i].mean()

            momentum = (
                0.6 * R_short +
                0.3 * R_mid +
                0.1 * R_long
            ).fillna(0.0)

            pos_mom = momentum.clip(lower=0.0)


            if pos_mom.sum() == 0:
                ranked = momentum.rank(method='first', ascending=False)
            else:
                ranked = pos_mom.rank(method='first', ascending=False)

            selected = ranked.nsmallest(self.topk).index

     
            vol_all = self.returns[assets].iloc[i - mid_w:i].std(ddof=0)
            med = vol_all[vol_all > 0].median()
            vol_all = vol_all.replace(0, med).fillna(med)

            inv_vol = 1.0 / vol_all
            mean_inv = inv_vol.mean()
            inv_vol = vol_shrink * inv_vol + (1 - vol_shrink) * mean_inv

           
            inv_sel = inv_vol.loc[selected]
            mom_sel = momentum.loc[selected].clip(lower=0.0)

            raw = mom_sel * inv_sel
            if raw.sum() <= 0 or not np.isfinite(raw.sum()):
                raw = inv_sel.copy()

            w_sel = raw / raw.sum()

           
            w_sel = np.minimum(w_sel, self.cap)
            if w_sel.sum() <= 0 or not np.isfinite(w_sel.sum()):
                w_sel = pd.Series(1 / len(selected), index=selected)
            else:
                w_sel = w_sel / w_sel.sum()

            w_full = pd.Series(0.0, index=self.price.columns)
            w_full.loc[w_sel.index] = w_sel.values


            spy_hist = self.price[self.exclude].iloc[:i + 1]
            ma200 = spy_hist.rolling(200).mean().iloc[-1]

            regime_scale = 1.0
            if pd.notna(ma200) and spy_hist.iloc[-1] < ma200:
                regime_scale = 0.5

           
            R_window = self.returns[assets].iloc[i - mid_w:i]
            Sigma = R_window.cov(ddof=0).values

            wA = np.array([w_full[a] for a in assets])
            port_var_daily = float(wA @ Sigma @ wA)
            port_vol_ann = math.sqrt(max(port_var_daily, eps)) * math.sqrt(252)

            if port_vol_ann <= 0:
                vol_scale = 1.0
            else:
                vol_scale = min(1.0, self.target_vol / port_vol_ann)

            scale = regime_scale * vol_scale
            w_scaled = w_full * scale

            self.portfolio_weights.iloc[i] = w_scaled.values        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns



if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)


