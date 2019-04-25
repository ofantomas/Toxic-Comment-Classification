import numpy as np
import torch
import tqdm


class BLSTM(torch.nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embeddings, device, dropout=0.0):
        super(BLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = torch.nn.Embedding.from_pretrained(embeddings)
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                                  dropout=dropout, bidirectional=True).to(self.device)
        self.linear = torch.nn.Linear(hidden_size * 4, num_classes).to(self.device)
    
    def init_hidden(self, batch_size):
        h0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        c0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
        return (h0, c0)

    def forward(self, x):
        h0, c0 = self.init_hidden(x.size(0))
        out = self.embedding(x).float().to(self.device)
        out, _ = self.lstm(out, (h0, c0))
        avg_pool = torch.mean(out, 1)
        max_pool, _ = torch.max(out, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        out = self.linear(conc)
        return out
    
class nnPredictor():
    def __init__(self, input_size, hidden_size, num_layers, num_classes, embeddings, dropout=0.5, 
                 learning_rate=1e-3, criterion=torch.nn.BCEWithLogitsLoss, optimizer=torch.optim.Adam, load_from=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() is True else 'cpu')
        self.learning_rate = learning_rate
        self.model = BLSTM(input_size, hidden_size, num_layers, num_classes, embeddings, self.device, dropout)
        self.criterion = criterion(reduction='mean')
        self.optimizer = optimizer
        
        
        if load_from is not None:
            self.model.load_state_dict(torch.load(os.getcwd() + "\\" + load_from))
        
    def train(self, data_train, data_test=None, num_epochs=3, batch_size=64, scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau):
        self.optim = self.optimizer(self.model.parameters(), lr=self.learning_rate)
        scheduler = scheduler(self.optim, 'min', patience=2)
        
        train_loss = []
        test_loss = []
         
        for epoch in range(num_epochs):
            with tqdm.tqdm(enumerate(data_train.batch_generator()), total=int(len(data_train) / batch_size)) as iterator:
                for batch_num, batch in iterator:
                    self.optim.zero_grad()
                    text, labels = batch
                    labels = labels.to(self.device)
                
                    outputs = self.model(text)
                    loss = self.criterion(outputs, labels)

                    loss.backward()
                    self.optim.step()
                    
                    train_loss.append(float(loss))            
                    iterator.set_description('Train loss: %.5f' % train_loss[-1])
            
            if data_test is not None:
                self.model.eval()
                val_loss = self.test(data_test, batch_size)
                scheduler.step(val_loss)
                test_loss.append(val_loss)
                self.model.train()
        
        return train_loss, test_loss

    def test(self, data_test, batch_size=64):
        #add other reductions + add functionality to remove predict
        scores = []
    
        with torch.no_grad():
            with tqdm.tqdm(enumerate(data_test.batch_generator()), total=int(len(data_test) / batch_size)) as iterator:
                for batch_num, batch in iterator:
                    text, labels = batch
                    labels = labels.to(self.device)
                
                    outputs = self.model(text)
                    scores.append(self.criterion(outputs, labels))
                    
        return torch.mean(torch.stack(scores))
        
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def predict(self, dataloader):
        with torch.no_grad():
            res = []
            for batch in dataloader.batch_generator():
                output = torch.sigmoid(self.model(batch)).to(torch.device('cpu'))
                res.append(output)
        
        last = res[-1]
        res = torch.stack(res[:-1])
        res = res.reshape(-1, 6)
        res = torch.cat((res, last), 0)
        
        return res
