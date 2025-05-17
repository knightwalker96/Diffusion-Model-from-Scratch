import torch
import torchvision
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from unet import Unet
from noise_scheduler import LinearNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample(model, scheduler):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    xt = torch.randn((100, 1, 28, 28)).to(device)
    for i in tqdm(reversed(range(1000))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        xt, x0_pred = scheduler.sample_prev_step(xt, noise_pred, torch.as_tensor(i).to(device))
        
        ims = torch.clamp(x0_pred, -1., 1.).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=10, normalize= False)
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join('default', 'samples')):
            os.makedirs(os.path.join('default', 'samples'), exist_ok = True)
        img.save(os.path.join('default', 'samples', 'x0_{}.png'.format(i)))
        img.close()


def infer():

    model = Unet(in_channels = 1).to(device)
    checkpoint = torch.load(os.path.join('default', 'ddpm_ckpt.pth'), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'    
    
    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device = device_str)
    
    with torch.no_grad():
        sample(model, scheduler)


if __name__ == '__main__':
    infer()