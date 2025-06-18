import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Загрузка актуального файла взаимодействий 
df = pd.read_csv("C:/Users/moyap/user_item_interactions_filtered.csv")

# Преобразуем ID в индексы 
user_ids  = df['user_id'].unique()
track_ids = df['track_id'].unique()
user2idx  = {u: i for i, u in enumerate(user_ids)}
track2idx = {t: i for i, t in enumerate(track_ids)}
df['user_idx']  = df['user_id'].map(user2idx)
df['track_idx'] = df['track_id'].map(track2idx)

num_users = len(user2idx)
num_items = len(track2idx)

#  Генерация негативных примеров
def generate_negatives(df, num_items, num_neg=4):
    neg = []
    user_pos = df.groupby('user_idx')['track_idx'].apply(set).to_dict()
    for u, positives in user_pos.items():
        for pos in positives:
            for _ in range(num_neg):
                j = np.random.randint(num_items)
                while j in positives:
                    j = np.random.randint(num_items)
                neg.append((u, j, 0))
    return pd.DataFrame(neg, columns=['user_idx', 'track_idx', 'interaction'])

neg_df = generate_negatives(df, num_items=num_items, num_neg=4)

#  Собираем полный датасет
data = pd.concat([
    df[['user_idx','track_idx','interaction']],
    neg_df
], ignore_index=True)

# Train/Test split
train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
# Определение Dataset для NCF
class NCFSampledDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        # Берём массивы индексов и меток
        self.users  = df['user_idx'].values
        self.items  = df['track_idx'].values
        self.labels = df['interaction'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'user':  torch.tensor(self.users[idx],  dtype=torch.long),
            'item':  torch.tensor(self.items[idx],  dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }
print(14+1)
# Создаём DataLoader  
batch_size = 256
train_loader = DataLoader(
    NCFSampledDataset(train_df),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0           # на Windows лучше 0, чтобы не было ошибок spawn
)
test_loader  = DataLoader(
    NCFSampledDataset(test_df),
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# Определение модели NCF
class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, u, i):
        # u, i — тензоры shape [batch]
        embs = torch.cat([
            self.user_emb(u),
            self.item_emb(i)
        ], dim=1)           # [batch, emb_dim*2]
        return self.mlp(embs).squeeze()  # [batch]

# Инициализация тренировки
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = NCF(num_users, num_items).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs    = 5

# Функции тренировки и валидации
def train_epoch(loader):
    model.train()
    total_loss = 0.0
    for batch in loader:
        u = batch['user'].to(device)
        i = batch['item'].to(device)
        y = batch['label'].to(device)
        optimizer.zero_grad()
        pred = model(u, i)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * u.size(0)
    return total_loss / len(loader.dataset)

def eval_epoch(loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            u = batch['user'].to(device)
            i = batch['item'].to(device)
            y = batch['label'].to(device)
            pred = model(u, i)
            total_loss += criterion(pred, y).item() * u.size(0)
    return total_loss / len(loader.dataset)

# Основной цикл обучения
for epoch in range(1, epochs+1):
    train_loss = train_epoch(train_loader)
    test_loss  = eval_epoch(test_loader)
    print(f"Epoch {epoch}/{epochs} — Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

# Сохранение модели
torch.save(model.state_dict(), 'ncf_model_new.pth')
print("Модель сохранена в ncf_model.pth")

# Оценка 

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

y_true = []
y_scores = []

model.eval()
with torch.no_grad():
    for batch in test_loader:
        u = batch['user'].to(device)
        i = batch['item'].to(device)
        y = batch['label'].to(device)

        preds = model(u, i)
        y_true.extend(y.cpu().tolist())
        y_scores.extend(preds.cpu().tolist())

# AUC
auc = roc_auc_score(y_true, y_scores)

# Метки по порогу 0.5
y_pred = [1 if s >= 0.5 else 0 for s in y_scores]

acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

print("\n=== Binary classification metrics on test set ===")
print(f"AUC:       {auc:.4f}")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")
