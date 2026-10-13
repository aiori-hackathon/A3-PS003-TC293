# FEDERATED TRAFFIC-FLOW PREDICTION-A3-PS003-TC293
Predict future traffic flow with **Federated Learning (FL)**so raw traffic data stays at each (simulated) intersection, and only small model updates travel to a central server.


## 1. Problem Context
 
Cameras and sensors at road intersections collect traffic data all day. In the usual ** centralised ** setup, all of it is sent to one cloud server where a forecasting model is trained. This causes two problems:
 
- **Bandwidth congestion.** Data travels over cellular links, which have limited capacity and cost money. Streaming raw video and sensor logs continuously from many intersections strains them.
- **Privacy vulnerability.** The server ends up holding one large record of where vehicles were and when. If that store is breached or misused, it exposes travel behaviour.
## 2. Proposed Solution
 
Instead of moving data to the model, **move the model to the data**:
 
1. Each intersection trains a small forecasting model on its own data.
2. Each intersection sends only the learned **weights** (not raw records) to a server.
3. The server merges the weights using **Federated Averaging (FedAvg)** into one global model.
4. The global model is sent back, and the cycle repeats.
The aims are to reduce bandwidth use and keep raw logs local, while still forecasting traffic accurately.
 
**Project contribution.** FedAvg is an established algorithm and is not claimed as novel. The contribution of this project is a reproducible, end-to-end simulation for traffic forecasting that measures prediction error, communication cost, and IID vs. non-IID behaviour.
