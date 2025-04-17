import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

# Load and clean data
df = pd.read_csv("amazon_reviews.csv")[['reviewText', 'overall']].dropna()
df = df[df['overall'].isin([1, 2, 3, 4, 5])]
df['label'] = df['overall'].astype(int) - 1

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Custom dataset class
class ReviewDataset(Dataset):
    def __init__(self, reviews, labels):
        self.encodings = tokenizer(reviews, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

# Prepare dataset
dataset = ReviewDataset(df['reviewText'].tolist(), df['label'].tolist())
train_size = int(0.8 * len(dataset))
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5).to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# Training loop
model.train()
for epoch in range(3):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_postfix(loss=loss.item())

# Evaluation
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == batch['labels']).sum().item()
        total += batch['labels'].size(0)

print(f"Accuracy: {correct / total:.4f}")
