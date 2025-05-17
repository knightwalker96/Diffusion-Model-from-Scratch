import torch
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from unet import Unet
from noise_scheduler import LinearNoiseScheduler
import torchvision
import torchvision.transforms as transforms

def train(args):
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    
    scheduler = LinearNoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device = device_str)

    transform = transforms.Compose([
        transforms.ToTensor(),           
        transforms.Lambda(lambda x: (2 * x) - 1) 
    ])

    mnist = torchvision.datasets.MNIST(
        root='./data',          
        train=True,             
        download=True,          
        transform=transform      
    )

    mnist_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=4)   
    
    # Instantiate the model
    model = Unet(in_channels=1).to(device)
    model.train()
    
    # Create output directories
    if not os.path.exists('default'):
        os.makedirs('default', exist_ok=True)
    
    # Initialize optimizer and criterion
    num_epochs = 50
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.MSELoss()

    # Checkpoint resuming logic
    start_epoch = 0
    checkpoint_path = os.path.join('default', 'ddpm_ckpt.pth')
    if args.resume and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        losses = []
        model.train()
        for batch in tqdm(mnist_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, _ = batch
            optimizer.zero_grad()
            images = images.float().to(device)

            # Sample random noise
            noise = torch.randn_like(images).to(device)
                
            # Sample timestep
            t = torch.randint(0, 1000, (images.shape[0],), device=device)  # Remove .float()
                
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(images, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
    
        # Print average loss for the epoch
        avg_loss = np.mean(losses)
        print(f'Finished epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
    
    print('Done Training ...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a DDPM model on MNIST")
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint")
    args = parser.parse_args()
    train(args)