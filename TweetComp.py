from DataPreparation import get_dl_training_set
from DataPreparation import get_dl_testing_set
from ModelPreparation import build_model
from ModelPreparation import build_encoder
from sklearn.metrics import accuracy_score
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
import torch
import numpy as np
import torch.nn as nn

class trainer(object):

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self, model, trainLoader, loss_func, optimizer_cls, optimizer_tr, scheduler):
        model.train()
        total_loss = 0
        instance = 0

        for step, batch in enumerate(trainLoader):
            instance += batch['label'].size()[0]

            emb = batch['embedding'].to(self.device)
            label = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            output = model(emb, mask).double()
            
            loss = loss_func(output, label[:, None])
            total_loss += loss.item()
            
            optimizer_tr.zero_grad()
            optimizer_cls.zero_grad()
            loss.backward()
            
            optimizer_tr.step()
            optimizer_cls.step()
            
            scheduler.step()
        
        return round(total_loss / instance, 5)

    def evaluate(self, model, valLoader, loss_func):
        model.eval()
        eval_loss = 0
        instance = 0
        pred = []
        ground_truth = []

        for batch in valLoader:
            instance += batch['label'].size()[0]

            emb = batch['embedding'].to(self.device)
            label = batch['label'].to(self.device)
            mask = batch['mask'].to(self.device)

            with torch.no_grad():
                output = model(emb, mask).double()
                
                loss = loss_func(output, label[:, None])
                eval_loss += loss.item()
                pred.extend(output.cpu().detach().numpy().tolist())
                ground_truth.extend(label.cpu().detach().numpy().tolist())

        pred = np.array(pred) >= 0.5
        accuracy = accuracy_score(ground_truth, pred)

        return round(eval_loss / instance, 5), round(accuracy, 4)

    def fit(self, models, batch_size, lr = 1e-5, tr_decay = 0.95, epochs = 3, saving = False):
        '''
        Executing training process

        Args:
            models(list): The name of transformer models
            batch_size(int): The number of samples in each batch
            lr(float): Learning rate of the model
            lr_decay(float): Decay rate for the transformer layers
            epochs(int): Number of training epochs
            saving(bool): Whether saves the model

        Returns:
        '''
        history = {}
        for model_name in models:
            # Preparing all required blocks
            model_history = []
            best_eval_loss = float('inf')
            encoder = build_encoder(model_name)
            model = build_model(model_name)
            model = model.to(self.device)
            trainLoader, valLoader = get_dl_training_set(encoder, batch_size)
            tr_lr = []
            for index in range(len(model.tr.encoder.layer)):
                holder = {'params' : model.tr.encoder.layer[-(index + 1)].parameters(),
                          'lr' : lr * (tr_decay ** index)}
                tr_lr.append(holder)
            criterion = nn.BCEWithLogitsLoss(reduction = 'sum')
            optimizer_cls = AdamW(model.cls.parameters(), lr)
            optimizer_tr = AdamW(tr_lr)
            scheduler = get_linear_schedule_with_warmup(optimizer = optimizer_tr,
                                                        num_warmup_steps = 150,
                                                        num_training_steps = len(trainLoader) * epochs)

            # Start training
            print(f'------{model_name}------')
            for epoch in range(epochs):
                print(f'Epoch {epoch + 1} training...')
                training_loss = self.train(model, trainLoader, criterion, optimizer_cls, optimizer_tr, scheduler)
                print(f'Evaluating...')
                eval_loss, accuracy = self.evaluate(model, valLoader, criterion)
                print(f'Training loss: {training_loss} | eval_loss: {eval_loss}')
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    if saving:
                        torch.save(model.state_dict(), model_name + '.pth')
                record = {'training_loss' : training_loss,
                          'eval_loss' : eval_loss,
                          'accuracy' : accuracy}
                model_history.append(record)
            history[model_name] = model_history
        
        return history

class inferencer(object):
    
    def __init__(self, model):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = build_encoder(model)
        self.loader = get_dl_testing_set(self.encoder)
        self.model = build_model(model)
        self.model.load_state_dict(torch.load(model + '.pth'))

    def inference(self):
        ids = []
        prediction = []
        self.model = self.model.to(self.device)
        for batch in self.loader:
            emb = batch['embedding'].to(self.device)
            mask = batch['mask'].to(self.device)
            ids.extend(batch['id'].tolist())
            
            output = self.model(emb, mask).squeeze().detach().cpu()
            output = np.array(output) >= 0.5
            prediction.extend(output.astype(int).tolist())
        return ids, prediction 