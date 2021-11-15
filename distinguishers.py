import numpy as np
from utils import recombine_fwht, Net, FastTensorDataLoader
from sklearn.ensemble import RandomForestClassifier
import torch

# This file contains various distinguishers
# All the function takes as input the leakage and the shares
# They return the pmf function.


def gt_sasca(leakage, shares, d, b):
    mean_shares = np.zeros((d, 2 ** b))
    std_shares = np.zeros(d)
    for i in range(d):
        for x in range(2 ** b):
            indexes = np.where(shares[:, i] == x)[0]
            if len(indexes) == 0:
                mean_shares[i, x] = 0
            else:
                mean_shares[i, x] = np.mean(leakage[indexes, i])
        std_shares[i] = np.std(leakage[:, i] - mean_shares[i, shares[:, i]])

    def pmf(leakage):
        n, ndim = leakage.shape
        prs_shares = np.zeros((n, d, 2 ** b))
        for i in range(d):
            for x in range(2 ** b):
                prs_shares[:, i, x] = np.exp(
                    -0.5 * ((leakage[:, i] - mean_shares[i, x]) / std_shares[i]) ** 2
                )

        prs_shares[np.where(prs_shares < 1e-100)] = 1e-100

        prs = recombine_fwht(prs_shares.T).T
        prs = (prs.T / np.sum(prs, axis=1)).T
        return prs

    return pmf


def rf(leakage, shares, d, b):
    data = 0
    for i in range(d):
        data ^= shares[:, i]
    clf = RandomForestClassifier(max_depth=8, random_state=0, n_estimators=100)
    clf.fit(leakage, data)
    return clf.predict_proba


def mlp(leakage, shares, d, b):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n, ndim = leakage.shape
    batch_size = max(1024, n // 1024)

    net = Net(ndim, 1000, b)
    secret = np.bitwise_xor.reduce(shares, axis=1)

    inputs = torch.tensor(leakage, dtype=torch.float).to(device)
    labels_torch = torch.tensor(secret, dtype=torch.long).to(device)

    # 20% of data for validation
    qt_train = inputs.shape[0] - inputs.shape[0] // 5
    inputs_train = inputs[:qt_train]
    labels_train = labels_torch[:qt_train]

    inputs_val = inputs[qt_train:]
    labels_val = labels_torch[qt_train:]

    losses_val = []

    # Fonciton de coût
    criterion = torch.nn.NLLLoss()

    # Defines the optimizer (and the corresponding learning rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Pour passer le modèle sur la GPU/CPU
    model = net.to(device)
    epochs = 100
    patience = 10

    log2inv = 1 / np.log(2)

    # training loops
    for epoch in range(epochs):

        # Updating the model
        model.train()
        for i, batch in enumerate(
            FastTensorDataLoader(
                inputs_train, labels_train, batch_size=batch_size, shuffle=True
            )
        ):

            traces, labels = batch
            optimizer.zero_grad()
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # model validation:
        loss_avg = []
        model.eval()
        for i, batch in enumerate(
            FastTensorDataLoader(
                inputs_val, labels_val, batch_size=len(inputs_val), shuffle=True
            )
        ):
            traces, labels = batch
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss_val = loss.item() / np.log(2)
            loss_avg.append(loss_val)
        losses_val.append(np.mean(loss_avg))

        losses_train_np = np.array(losses_val)

        # use validation to avoid overfitting
        if epoch > patience and epoch > 20:
            if (
                np.all(losses_train_np[-patience] <= losses_train_np[-patience:])
                and losses_train_np[-patience] > 0
            ):
                break

    def get_prs_k_l(leakage_test, model=model):
        inputs = torch.tensor(leakage_test, dtype=torch.float).to(device)
        prs_l_k = np.exp(model(inputs).detach().cpu().numpy())
        prs_l_k[np.where(prs_l_k < 1e-100)] = 1e-100
        prs_k_l = (prs_l_k.T / np.sum(prs_l_k, axis=1)).T
        return prs_k_l

    return get_prs_k_l
