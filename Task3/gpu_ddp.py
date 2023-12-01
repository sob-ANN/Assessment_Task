import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, fp16=False):
    if fp16:
        model = model.half()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, eps=1e-8)
        amp = torch.cuda.amp.GradScaler()

    model.to(device)

    for epoch in range(num_epochs):
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if fp16:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                amp.scale(loss).backward()
                amp.step(optimizer)
                amp.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                valid_loss += loss.item()

        train_loss /= len(train_loader)
        valid_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Val Loss: {valid_loss:.4f}')

def main():
    # Model definition
    model = ...  
    optimizer = ...  
    criterion = ...  
    
    #Data loaders
    train_loader = ...  
    val_loader = ...  

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Single GPU training
    if torch.cuda.device_count() == 1:
        train_model(model, train_loader, val_loader, optimizer, criterion, device)

    # Distributed Data Parallel (DDP) training
    if torch.cuda.device_count() > 1:
        model = DistributedDataParallel(model)
        train_model(model, train_loader, val_loader, optimizer, criterion, device)

    # Fully Sharded Data Parallel (FSDP) training
    if FSDP.is_fsdp_available() and torch.cuda.device_count() > 1:
        model = FSDP(model)
        train_model(model, train_loader, val_loader, optimizer, criterion, device)

if __name__ == '__main__':
    main()
