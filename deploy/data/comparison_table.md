| Method | MAE (veh/h) | RMSE (veh/h) | Bytes sent over training |
|---|---|---|---|
| Always predict the training average | 1728.3 | 1974.2 | 0 |
| Repeat the last hour | 588.3 | 815.5 | 0 |
| Same hour yesterday | 569.2 | 1028.2 | 0 |
| Centralized MLP (all data in one place) | 158.6 ± 0.8 | 232.6 ± 2.0 | raw data must be uploaded |
| Local-only, IID clients (mean of 4 models) | 167.3 ± 4.8 | 248.9 ± 2.9 | 0 |
| Local-only, Non-IID clients (mean of 4 models) | 837.9 ± 66.0 | 1231.2 ± 74.1 | 0 |
| FedAvg, IID clients | 148.2 ± 1.9 | 229.0 ± 1.0 | 6,240,000 |
| FedAvg, Non-IID clients | 304.8 ± 10.8 | 439.9 ± 22.4 | 6,240,000 |
