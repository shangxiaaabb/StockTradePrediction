<!--
 * @Author: Jie Huang huangjie20011001@163.com
 * @Date: 2024-06-14 21:37:23
-->
# File Structure

```
├─data
│  └─0308
└─model
```

The `data` directory contains the trading data for 18 stocks from `2020-05-13` to `2021-03-08`. The subdirectory `0308` likely holds data specific to March 8th, and the `model` directory may contain related models or analyses.

```data``` represents the following for each stock:
- `daily_volume`: Daily trading volume.
- `bin_volume`: Trading volume within each trading interval.
- `bin_num`: Interval number set according to trading time.
- `volatility`: Standard deviation of the current inventory price, calculated as $\sqrt{\frac{1}{n}\sum_{i=1}^{n}{(P_{i}-\overline{P})^{2}}}$.
- `quote_imbalance`: The exponentially weighted moving average of the quote imbalance within the current time interval, calculated as $\mathrm{EWMA}\left(\frac{b1_v-a1_v}{b1_v+a1_v}\right)$.

> 1. Where `n` is the number of prices within the current interval, $P_i$ is the $i$th price, and $P̄$ is the average price within the interval.
> 2. `b1v` is the volume at the buyer's best price, `a1v` is the volume at the seller's best price, and `EWMA` stands for Exponentially Weighted Moving Average. The formula for `EWMA` can be expressed as: $EWMA(X_t) = (1 - lambda) * X_t + lambda * EWMA(X_{t-1})$, where `lambda` is the decay factor, typically ranging in value from 0 to 1. In this study, $X_t$ represents $(b1v - a1v) / (b1v + a1v)$.

> 1. Each trading day in the Chinese stock market lasts for 4 hours. Excluding the last three minutes for the closing auction, the trading time for each trading day is 3 hours and 57 minutes. Each trading day is divided into 24 intervals, with each interval lasting 9.875 minutes.
> 2. After performing a first-order difference operation on the cumulative trading volume and cumulative turnover to obtain the trading volume every 3 seconds, the data is aggregated by date and interval number to obtain: daily trading volume and trading volume within each trading interval.
