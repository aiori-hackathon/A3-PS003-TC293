# Presentation outline and demo script (about 8 minutes)

Speakers: **S** = Shaurya, **N** = Nayab. Say "simulated" every time you mention intersections.

| # | Slide | Time | Who | Key message |
|---|---|---|---|---|
| 1 | Title | 0:20 | S | PS-003, team, mentor, AIORI. |
| 2 | Problem | 0:45 | S | Raw data to one cloud costs bandwidth and builds a record of movement. |
| 3 | Idea | 0:45 | N | Move the model to the data. Clients send only weights; the server averages them (FedAvg). |
| 4 | Architecture | 0:60 | S | Diagram from the README: 4 simulated clients, server, rounds. Server receives bytes only. |
| 5 | Data and honesty | 0:60 | N | UCI I-94 hourly data, ONE sensor, so clients are simulated slices. Time split, scaler on train only. |
| 6 | The leak we caught | 0:45 | S | Filled gaps contain the future value; we excluded those targets (4,655 train / 38 test) and regenerated every result. |
| 7 | Experiment design | 0:45 | N | IID vs Non-IID (weekday morning/afternoon/night/weekend), centralized and local-only baselines, 3 seeds, same budget. |
| 8 | Results table | 1:00 | S | Show the table. IID FedAvg 148.2 matches centralized 158.6; Non-IID 304.8 is about 2x worse but 46% below the best baseline; local-only Non-IID collapses. |
| 9 | Communication | 0:60 | N | 208,000 bytes per round, 6.24 MB total = 1.93x the raw file. No bandwidth saving on this small file; the idea pays off for continuous raw streams (labeled assumption). |
| 10 | Limits and future | 0:45 | S | Simulated clients, not full privacy (update leakage), one location. Future: secure aggregation, real multi-site data, LSTM. |
| 11 | Live demo | 1:00 | N | See script below. |

## Live demo script

1. Open the live link (**5-10 minutes before, free hosting sleeps**). Fallback: `python deploy\app.py` on a laptop.
2. Read the notice box aloud: simulated, historical data, not full privacy.
3. Pick a weekday, click **Show forecast**: black = real traffic, colored lines = the two models. Point to the MAE/RMSE for the selection.
4. Switch to **7 days**: forecasts follow the daily rhythm; point out where they miss (rush-hour peaks).
5. Scroll to the results table and the communication facts (the honest finding).
6. Pick a date outside the range to show the clear error message (optional).

## Expected judge questions (answers use measured facts only)

* **Why simulate the clients?** Our dataset has one sensor. We say so; real multi-intersection data is future work.
* **Is it private?** Not fully. Raw data stays local, but updates can leak information; secure aggregation and differential privacy are future work.
* **Does it save bandwidth?** Not on this dataset: 6.24 MB over 30 rounds versus a 3.24 MB raw file (1.93x). The saving argument applies to continuous raw streams such as video; our table for that is an assumption.
* **Why is Non-IID worse?** Each client drifts toward its own regime during local training, so the average is a compromise: MAE 304.8 versus 148.2.
* **Is federated better than centralized?** We do not claim that. IID FedAvg matches it within the noise (148.2 vs 158.6; the centralized error varies 10-15 between epochs).
* **How did you prevent leakage?** Time-based split; scaler fitted on training data only; examples built after the split; and we found and fixed the interpolation leak.
* **Biggest mistake?** The interpolation leak. Found when reviewing the first results; fixed, tested, results regenerated. MAE moved from 154.1 to 158.6.
* **How is it reproducible?** Fixed seeds, one config file, saved results, automated tests, and an identical preprocessing summary on both laptops.
* **Why an MLP?** Small and explainable; a message is only 26 KB. LSTM is future work.
* **Why 30 rounds?** It was fixed in advance to match the centralized budget (60 epochs). Non-IID had not fully flattened; a longer run would be a separate, labeled experiment.
* **What would real deployment need?** Real edge devices and networking, secure aggregation, monitoring for drift, and data from several real intersections (including Indian roads).

## Implemented / simulated / future

* **Implemented and tested:** preprocessing, IID/Non-IID partitions, MLP, FedAvg, byte counting, baselines, analysis, demo app.
* **Simulated:** the intersections, the network, the edge hardware.
* **Future:** secure aggregation, differential privacy, real multi-site data, LSTM, weather features, deployment on devices.
