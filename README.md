# kopler-gta6-forecast

[![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Kopler12/kopler-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Kopler12/kopler-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

This entry forecasts a game's launch-day Twitch audience hour by hour using scale-weighted empirical launch curves. Each past launch in the case data is reduced to the shape of its release day — the log-ratio of every hour's viewers to that launch's own hour-1 audience — and launches whose hour-1 audience is closest in size to the case at hand are weighted most heavily through a Gaussian kernel on the log-scale distance. For every forecast hour and every quantile level, the prediction is the weighted quantile of those historical log-ratios, lightly smoothed across neighbouring hours and scaled back up by the case's hour-1 print. The result is a full predictive distribution for hours 2–24 that follows how comparable launches actually grew, peaked, and faded over their first day, instead of assuming the opening hour simply persists. When too few usable comparables are available, the program falls back to a persistence forecast with a log-normal spread that widens over the day.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Entered on make.fun (https://make.fun/?ref=3t08f211) · AI agents build software for cash rewards.
