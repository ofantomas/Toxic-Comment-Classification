import torch
import tqdm


class BLSTM(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embeddings, device, dropout=0.5):
        super(BLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                                  dropout=dropout, bidirectional=True).to(self.device)
        self.linear = torch.nn.Linear(hidden_size * 2, num_classes).to(self.device)
    
    def init_hidden(self, batch_size):
        h0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        out = self.embedding(x).float().to(self.device)
        out, _ = self.lstm(out, (h0, c0))
        out = self.linear(out[:, -1, :])
        return torch.sigmoid(out)
    
class nnPredictor():
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embeddings, dropout=0.50, 
                 learning_rate=0.05, criterion=torch.nn.BCELoss, optimizer=torch.optim.Adam, load_from=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() is True else 'cpu')
        self.learning_rate = learning_rate
        self.model = BLSTM(input_size, hidden_size, num_layers, num_classes, embeddings, self.device, dropout)
        self.criterion = criterion()
        self.optimizer = optimizer
        
        
        if load_from is not None:
            self.model.load_state_dict(torch.load(os.getcwd() + "\\" + load_from))
        
    def train(self, dataloader, num_epochs=3, verbose_step=5000):
        self.optimizer = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        
        acc_loss = 0
        verbose_step = verbose_step
        total_steps = len(dataloader)

        for epoch in range(num_epochs):
            for i, batch in enumerate(dataloader.batch_generator()):
                text, labels = batch
                labels = labels.to(self.device)
                
                outputs = self.model(text)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                acc_loss += loss.item()
                if ((i + 1) * dataloader.batch_size) % verbose_step == 0:
                    print('Epoch [{:1d} / {:1d}], Step [{:6d} / {:6d}], Average loss: {:.4f}'
                          .format(epoch + 1, num_epochs, (i + 1) * dataloader.batch_size, 
                                  total_steps, acc_loss * dataloader.batch_size / verbose_step))
                    acc_loss = 0
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def predict(self, dataloader):
        with torch.no_grad():
            res = []
            for batch in dataloader.batch_generator():
                output = self.model(batch).to(torch.device('cpu'))
                res.append(output)
        
        last = res[-1]
        res = torch.stack(res[:-1])
        res = res.reshape(-1, 6)
        res = torch.cat((res, last), 0)
        
        return res
