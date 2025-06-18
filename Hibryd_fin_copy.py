import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict

# === 1. Подгружаем данные ===
interactions = pd.read_csv("C:/Users/moyap/user_item_interactions_filtered.csv")
features = pd.read_csv("C:/Users/moyap/content_features.csv").dropna(subset=['track_id'])

# Обеспечиваем наличие нужных столбцов
assert 'title' in features.columns, "В content_features.csv должен быть столбец 'title'"
assert 'artist' in features.columns, "В content_features.csv должен быть столбец 'artist'"

# Кодировка пользователей и треков для совместимости с NCF
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
interactions['user_num'] = user_encoder.fit_transform(interactions['user_id'])
interactions['item_num'] = item_encoder.fit_transform(interactions['track_id'])
# features['item_num'] = item_encoder.transform(features['track_id'])

features = features[features['track_id'].isin(item_encoder.classes_)].copy()
features['item_num'] = item_encoder.transform(features['track_id'])




# === 2. Модель NCF ===
class NCF(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, u, i):
        # u, i — тензоры shape [batch]
        embs = torch.cat([
            self.user_emb(u),
            self.item_emb(i)
        ], dim=1)           # [batch, emb_dim*2]
        return self.mlp(embs).squeeze()  # [batch]

num_users = interactions['user_num'].nunique()
num_items = interactions['item_num'].nunique()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NCF(num_users, num_items).to(device)
model.load_state_dict(torch.load("C:/Users/moyap/ncf_model_new.pth", map_location=device))
model.eval()

# === 3. Content-based часть (KNN по аудиофичам) ===
audio_cols = [col for col in features.columns if col not in ['track_id', 'title', 'artist', 'item_num'] and features[col].dtype in [np.float64, np.int64]]
scaler = StandardScaler()
features[audio_cols] = scaler.fit_transform(features[audio_cols])
track_feature_dict = features.set_index('track_id')[audio_cols].to_dict(orient='index')
track_matrix = features[audio_cols].values
track_ids_knn = features['track_id'].values

knn = NearestNeighbors(n_neighbors=20, metric='cosine')
knn.fit(track_matrix)

def get_knn_scores(user_history, n=10):
    scores = defaultdict(float)
    for track_id in user_history:
        if track_id not in track_feature_dict:
            continue
        vec = np.array([list(track_feature_dict[track_id].values())])
        distances, indices = knn.kneighbors(vec, n_neighbors=n)
        for dist, idx in zip(distances[0], indices[0]):
            tid = track_ids_knn[idx]
            scores[tid] += 1 - dist  # ближе → выше
    return dict(scores)

# === 4. Гибридная рекомендация ===
def hybrid_recommend(user_id, alpha=0.7, top_n=10):
    # История пользователя
    user_hist = interactions[interactions['user_id'] == user_id]['track_id'].tolist()
    if not user_hist:
        print("Нет истории пользователя.")
        return None
    user_num = user_encoder.transform([user_id])[0]
    # Кандидаты для рекомендации
    candidate_ids = features['track_id'].values
    candidate_item_nums = features['item_num'].values

    # NCF предсказания
    user_arr = torch.tensor([user_num]*len(candidate_item_nums), dtype=torch.long, device=device)
    item_arr = torch.tensor(candidate_item_nums, dtype=torch.long, device=device)
    with torch.no_grad():
        ncf_scores = model(user_arr, item_arr).cpu().numpy()
    # Content-based предсказания
    knn_scores_dict = get_knn_scores(user_hist, n=20)
    knn_scores = np.array([knn_scores_dict.get(tid, 0) for tid in candidate_ids])
    # Нормализация
    ncf_scores = (ncf_scores - ncf_scores.min()) / (ncf_scores.max() - ncf_scores.min() + 1e-6)
    if knn_scores.max() > 0:
        knn_scores = knn_scores / (knn_scores.max() + 1e-6)
    # Гибридный скор
    hybrid_scores = alpha * ncf_scores + (1 - alpha) * knn_scores

    # Формируем итоговый DataFrame
    res_df = features[['track_id', 'title', 'artist']].copy()
    res_df['score'] = hybrid_scores
    # Исключаем уже прослушанные
    res_df = res_df[~res_df['track_id'].isin(user_hist)]
    res_df = res_df.sort_values('score', ascending=False).head(top_n)
    res_df = res_df.reset_index(drop=True)
    return res_df

# === 5. Метрики качества для топ-10 ===
def evaluate_hybrid(interactions, features, model, user_encoder, item_encoder, alpha=0.7, top_n=10, n_users=100):
    precisions, recalls, ndcgs = [], [], []
    np.random.seed(42)
    test_users = np.random.choice(interactions['user_id'].unique(), size=n_users, replace=False)
    for user_id in test_users:
        user_hist = interactions[interactions['user_id'] == user_id]['track_id'].tolist()
        # Для простоты — считаем, что последняя прослушка — "будущая" (для теста)
        if len(user_hist) < 2:
            continue
        test_item = user_hist[-1]
        train_hist = user_hist[:-1]
        recs = hybrid_recommend(user_id, alpha, top_n)
        if recs is None: continue
        recommended_ids = recs['track_id'].tolist()
        hit = int(test_item in recommended_ids)
        precision = hit / top_n
        recall = hit / 1  # только 1 релевантный объект
        # NDCG
        if test_item in recommended_ids:
            rank = recommended_ids.index(test_item)
            ndcg = 1 / np.log2(rank + 2)
        else:
            ndcg = 0
        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)
    return np.mean(precisions), np.mean(recalls), np.mean(ndcgs)

# === Пример вывода ===
user_id = interactions['user_id'].iloc[10000]
recs_df = hybrid_recommend(user_id, alpha=0.7, top_n=10)
print(f"\nГибридные рекомендации для пользователя {user_id}:")
print(recs_df)

# Оценка метрик
precision, recall, ndcg = evaluate_hybrid(
    interactions, features, model, user_encoder, item_encoder, alpha=0.7, top_n=10, n_users=50
)
print(f"\nМетрики качества (по 50 случайным пользователям):")
print(f"Precision@10: {precision:.4f}")
print(f"Recall@10: {recall:.4f}")
print(f"NDCG@10: {ndcg:.4f}")

# Сохранить рекомендации в файл
recs_df.to_csv("hybrid_recommendations.csv", index=False)
