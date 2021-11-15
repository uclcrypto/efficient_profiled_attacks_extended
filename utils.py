import numpy as np

HW = 0
r = np.arange(256).astype(np.uint8)
while np.max(r) > 0:
    HW += r & 0x1
    r = r >> 1


def get_HW(matrix):
    return HW[matrix]


def fwht(a):
    h = 1
    x = np.zeros(a[0].shape).astype(np.float64)
    y = np.zeros(a[0].shape).astype(np.float64)
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x[:] = a[j]
                y[:] = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def recombine_fwht(pr):
    """
    pr is of size (Nk x D x  Nt):
        Nk size of the field
        D number of shares
        Nt  number of traces
    """
    pr = pr.astype(np.float64)
    pr_fft = pr.copy()
    fwht(pr_fft)
    pr = np.prod(pr_fft, axis=1)
    fwht(pr)
    return pr


#######################
### PyTorch
#######################
import torch


class Net(torch.nn.Module):
    """
    Classe définissant l'architecture du modèle utilisé.
    """

    def __init__(self, dim_inp, n_hidden, n_bits):
        """
        Where we initialize the different layers.
        """
        super(Net, self).__init__()
        self.bn0 = torch.nn.BatchNorm1d(dim_inp)
        self.fc1 = torch.nn.Linear(dim_inp, n_hidden)
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)
        self.fc2 = torch.nn.Linear(n_hidden, n_hidden // 8)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden // 8)
        self.fc3 = torch.nn.Linear(n_hidden // 8, 1 << n_bits)

    def forward(self, x):
        """
        The forward pass, the backward pass is automatically
        defined in Pytorch.
        """
        x = self.bn0(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.relu(x)

        x = self.fc3(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x


from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class FastTensorDataLoader:
    """
    A DataLoader-like object for a set of tensors that can be much faster than
    TensorDataset + DataLoader because dataloader grabs individual indices of
    the dataset and calls cat (slow).
    Source:
    https://discuss.pytorch.org/t/dataloader-much-slower-than-manual-batching/
    """

    def __init__(self, *tensors, batch_size=32, shuffle=False):
        """
        Initialize a FastTensorDataLoader.

        :param *tensors: tensors to store. Must have the same length @ dim 0.
        :param batch_size: batch size to load.
        :param shuffle: if True, shuffle the data *in-place* whenever an
            iterator is created out of this object.

        :returns: A FastTensorDataLoader.
        """
        assert all(t.shape[0] == tensors[0].shape[0] for t in tensors)
        self.tensors = tensors

        self.dataset_len = self.tensors[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.tensors = [t[r] for t in self.tensors]
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = tuple(t[self.i : self.i + self.batch_size] for t in self.tensors)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches


def mlp_torch(leakage_train, shares_train, bits, D):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    # print("start_mlp_torch",device)
    net = Net(len(leakage_train[0, :]), 1000, bits)
    secret_train = np.bitwise_xor.reduce(shares_train, axis=1)

    inputs = torch.tensor(leakage_train, dtype=torch.float).to(device)
    labels_torch = torch.tensor(secret_train, dtype=torch.long).to(device)

    # 1/5ème des données simulées servent pour la validation
    qt_train = inputs.shape[0] - inputs.shape[0] // 5
    inputs_train = inputs[:qt_train]
    labels_train = labels_torch[:qt_train]

    inputs_val = inputs[qt_train:]
    labels_val = labels_torch[qt_train:]

    batch_size = max(1024, len(leakage_train) // 1024)

    losses_train = []
    losses_val = []

    # Fonciton de coût
    criterion = torch.nn.NLLLoss()

    # Defines the optimizer (and the corresponding learning rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    # Pour passer le modèle sur la GPU/CPU
    model = net.to(device)
    epochs = 100
    asympto = False
    if asympto:
        patience = 5
    else:
        patience = epochs
        patience = 10

    log2inv = 1 / np.log(2)
    # Boucle d'entraînement
    for epoch in tqdm(range(epochs)):
        # Passe sur les données d'entraînement
        loss_avg = []
        start = time.time()
        model.train()
        start_2 = time.time()
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
            loss_train = loss.item() * log2inv
            loss_avg.append(loss_train)
            # print(time.time()-start_2)
            # start_2 = time.time()

        losses_train.append(np.mean(loss_avg))
        # Passe sur les données de validation
        loss_avg = []
        model.eval()
        for i, batch in enumerate(
            FastTensorDataLoader(inputs_val, labels_val, batch_size=20000, shuffle=True)
        ):
            traces, labels = batch
            outputs = model(traces)
            loss = criterion(outputs, labels)
            loss_val = loss.item() / np.log(2)
            loss_avg.append(loss_val)
        losses_val.append(np.mean(loss_avg))

        if epoch % 1 == 0 and verbose:
            print(
                "\rEpoch: %d, loss_train: %.7f \tloss_eval: %.7f %.4f %d"
                % (
                    epoch,
                    bits - losses_train[-1],
                    bits - losses_val[-1],
                    time.time() - start,
                    os.getpid(),
                )
            )

        losses_train_np = np.array(losses_val)
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
