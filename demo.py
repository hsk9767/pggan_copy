import os
import wget
import os.path
import argparse
import logging


from model import *
from utils import *
from torchvision import transforms
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='DATA', help='directory containing the data')
parser.add_argument('--weight', type=str, default='./weight/Gs.pth', help='directory containing the weight')
parser.add_argument('--outd', default='demo_result', help='directory to save results')
parser.add_argument('--outf', default='Images', help='folder to save synthetic images')
parser.add_argument('--outl', default='Losses', help='folder to save Losses')
parser.add_argument('--outm', default='Models', help='folder to save models')

parser.add_argument('--workers', type=int, default=8, help='number of data loading workers')
parser.add_argument('--batchSizes', type=list, default=[4, 4, 4, 4, 4, 4, 3, 2], help='list of batch sizes during the training')
parser.add_argument('--nch', type=int, default=8, help='base number of channel for networks')
parser.add_argument('--BN', action='store_true', help='use BatchNorm in G and D')
parser.add_argument('--WS', action='store_true', help='use WeightScale in G and D')
parser.add_argument('--PN', action='store_true', help='use PixelNorm in G')
parser.add_argument('--CM', action='store_true', help='use cutmix in D')
parser.add_argument('--MAX_RES', type=int, default=7, help='log2(im_size) - 2')
parser.add_argument('--savenum', type=int, default=36, help='number of examples images to save')


opt = parser.parse_args()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
logger = logging.getLogger('name')
logger.setLevel(logging.INFO)

# creating output folders
if not os.path.exists(opt.outd):
    os.makedirs(opt.outd)
    os.makedirs(os.path.join(opt.outd, opt.outf))

# Load the weight
G = Generator(max_res=opt.MAX_RES, nch=opt.nch, nc=3).to(DEVICE)
G.load_state_dict(torch.load(opt.weight).state_dict())
G.eval()
logging.info('Model Weight Loaded')
#
# Generate
z_save = hypersphere(torch.randn(opt.savenum, opt.nch * 32, 1, 1, device=DEVICE))
with torch.no_grad():
    fake_image = G(z_save, 7.00)
    save_image(fake_image, os.path.join(opt.outd, opt.outf, f'demo_img.jpg'),
               nrow=6, pad_value=0, normalize=True, range=(-1, 1))
logger.info('Image Generated and Saved')
