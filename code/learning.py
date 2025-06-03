import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import CoffeeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


X = torch.tensor([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
    [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
    [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]
], dtype=torch.float32).to(device)
Y = torch.tensor([
    1, 1, 1, 1,
    1, 0, 0, 0,
    1, 1, 1, 1,
    1, 0, 0, 0
], dtype=torch.float32).to(device)

model = CoffeeModel().to(device)

dataset = TensorDataset(X, Y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 500
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

with torch.no_grad():
    outputs = model(X)
    predicted = (outputs > 0.5).float()
    accuracy = (predicted == Y).float().mean()
    print(f'\nAccuracy: {accuracy.item()*100:.2f}%')


torch.save(model.state_dict(), 'code/coffee_model.pth')
print("Model saved as 'coffee_model.pth'")




