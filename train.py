import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data import MyDataset
from RDL_unet import UNet
import pandas as pd
from torchvision.utils import save_image

# configuration parameter
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'
save_path_label = 'GD/label'
save_path_out = 'GD/out'
epoch_num = 300  # Set the number of epochs for training

# Initialize models and devices
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Load dataset
data_loader = DataLoader(MyDataset(data_path), batch_size=1, shuffle=True)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_function = nn.BCELoss()

# training process
epoch_losses = []  # Used to store the total loss for each epoch
epoch_averages = []  # Used to store the average loss for each epoch

for epoch in range(1, epoch_num + 1):
    epoch_loss = 0.0
    for i, (image, segment_image) in enumerate(data_loader):
        image, segment_image = image.to(device), segment_image.to(device)
        optimizer.zero_grad()
        output = model(image)
        loss = loss_function(output, segment_image)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        if i % 5 == 0:
            print(f'Epoch {epoch} - Iteration {i}: Loss = {loss.item()}')

        if i % 50 == 0:
            _segment_image = segment_image[0]
            _out_image = output[0]
            img = torch.stack([_segment_image, _out_image], dim=0)
            save_image(img, f'{save_path}/{epoch}_{i}.png')
            save_image(_segment_image, f'{save_path_label}/{epoch}_{i}.png')
            save_image(_out_image, f'{save_path_out}/{epoch}_{i}.png')

    average_loss = epoch_loss / len(data_loader)
    epoch_losses.append(epoch_loss)
    epoch_averages.append(average_loss)
    print(f'Epoch {epoch} - Average Loss: {average_loss}')

    torch.save(model.state_dict(), weight_path)

loss_data = {
    'Epoch': range(1, epoch_num + 1),
    'Total Loss': epoch_losses,
    'Average Loss': epoch_averages
}
df = pd.DataFrame(loss_data)
df.to_csv('loss_data.csv', index=False)
