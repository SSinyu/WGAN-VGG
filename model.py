import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import vgg19
from RED_CNN_util import build_dataset

class train_dcm_data_loader(Dataset):
    def __init__(self, input_lst, target_lst, crop_size=None, crop_n=None):
        self.input_lst = input_lst
        self.target_lst = target_lst
        self.crop_size = crop_size
        self.crop_n = crop_n

    def __getitem__(self, idx):
        input_img = self.input_lst[idx]
        target_img = self.target_lst[idx]

        if self.crop_n:
            assert input_img.shape == target_img.shape
            crop_input = []
            crop_target = []
            h, w = input_img.shape
            new_h, new_w = self.crop_size, self.crop_size
            for _ in range(self.crop_n):
                top = np.random.randint(0, h-new_h)
                left = np.random.randint(0, w-new_w)
                input_img_ = input_img[top:top+new_h, left:left+new_w]
                target_img_ = target_img[top:top+new_h, left:left+new_w]
                crop_input.append(input_img_)
                crop_target.append(target_img_)
            crop_input = np.array(crop_input)
            crop_target = np.array(crop_target)

            sample = (crop_input, crop_target)
            return sample
        else:
            sample = (input_img, target_img)
            return sample

    def __len__(self):
        return len(self.input_lst)



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out



class Generator_CNN(nn.Module):
    def __init__(self):
        super(Generator_CNN, self).__init__()
        self.g_conv_n32s1_f = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n1s1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, img):
        out = self.relu(self.g_conv_n32s1_f(img))
        for _ in range(6):
            out = self.relu(self.g_conv_n32s1(out))
        out = self.relu(self.g_conv_n1s1(out))
        return out



class Discriminator_CNN(nn.Module):
    def __init__(self, input_size=55):
        super(Discriminator_CNN, self).__init__()

        def after_conv_size_c(input_size, kernel_size_list, stride_list):
            cal = (input_size - kernel_size_list[0]) // stride_list[0] + 1
            for i in range(1, len(kernel_size_list)):
                cal = (cal - kernel_size_list[i]) // stride_list[i] + 1
            return cal

        def discriminator_block(in_filters, out_filters, stride):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, padding=0)]
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        for in_filters, out_filters, stride in [(1,64,1), (64,64,2), (64,128,1), (128,128,2), (128,256,1), (256,256,2)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride))

        self.fc_size = after_conv_size_c(input_size, [3,3,3,3,3,3], [1,2,1,2,1,2])
        self.cnn = nn.Sequential(*layers)
        self.leaky = nn.LeakyReLU()
        self.fc1 = nn.Linear(256*self.fc_size*self.fc_size, 1024)
        self.fc2 = nn.Linear(1024,1)

    def forward(self, img):
        out = self.cnn(img)
        out = out.view(-1, 256*self.fc_size*self.fc_size)
        out = self.fc1(out)
        out = self.leaky(out)
        out = self.fc2(out)
        return out


def calc_gradeint_penalty(discriminator, real_data, fake_data, lambda_):
    alpha = torch.rand(BATCH_SIZE*CROP_NUMBER, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda(gpu) if torch.cuda.is_available() else alpha

    interpolates = alpha * real_data + ((1-alpha) * fake_data)

    if torch.cuda.is_available():
        interpolates = interpolates.cuda(gpu)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates, grad_outputs=torch.ones(disc_interpolates.size()).cuda(gpu) if torch.cuda.is_available() else torch.ones(disc_interpolates.size()), create_graph=True, retain_graph=Ture, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_
    return gradient_penalty

    
    
#### training ####
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
BATCH_SIZE = 4
CROP_NUMBER = 100  # The number of patches to extract from a single image. --> total batch img is BATCH_SIZE * CROP_NUMBER
PATCH_SIZE = 55
NUM_WORKERS = 20
LAMBDA_ = 10
d_min = -1024.0
d_max = 3072.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_path = '/home/datascience/Denoising/AAPM-Mayo-CT-Challenge/'
test_patients = [data for data in os.listdir(data_path) if 'zip' not in data and 'data' not in data]

patient = 'L506'
# model weight save path
if not (os.path.isdir('/home/datascience/PycharmProjects/sinyu/REDCNN/result/{}test'.format(patient))):
    os.makedirs(os.path.join('/home/datascience/PycharmProjects/sinyu/REDCNN/result/{}test'.format(patient)))

# train/test data processing
input_dir, target_dir, test_input_dir, test_target_dir = build_dataset(patient, "3mm", norm_range=(d_min, d_max))
assert len(os.listdir(os.path.join(data_path, "{}/full_3mm").format(patient))) == len(test_input_dir)



train_dcm = train_dcm_data_loader(input_dir, target_dir, crop_size=PATCH_SIZE, crop_n=CROP_NUMBER)
train_loader = DataLoader(train_dcm, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

criterion_perceptual = nn.L1Loss()

generator = Generator_CNN()
discriminator = Discriminator_CNN(input_size=55)
feature_extractor = FeatureExtractor()

if torch.cuda.device_count() > 1:
    print("Use {} GPUs".format(torch.cuda.device_count()), "="*9)
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    feature_extractor = nn.DataParallel(feature_extractor)

generator.to(device)
discriminator.to(device)
feature_extractor.to(device)

criterion_GAN = nn.MSELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5,0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5,0.9))


total_step = len(train_loader)
for epoch in range(10):
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.reshape(-1, 55, 55).to(device)
        targets = targets.reshape(-1, 55, 55).to(device)

        input_img = torch.tensor(inputs, requires_grad=True).unsqueeze(1).to(device)
        input_img = input_img.type(torch.FloatTensor)
        target_img = torch.tensor(targets).unsqueeze(1).to(device)
        target_img = target_img.type(torch.FloatTensor)

        # Train D
        discriminator.zero_grad()

        # Train D on real
        d_real_decision = discriminator(target_img)
        d_real_error = -torch.mean(d_real_decision)
        d_real_error.backward()

        # Train D on fake
        d_fake_data = generator(input_img).detach()
        d_fake_decision = discriminator(d_fake_data)
        d_fake_error = torch.mean(d_fake_decision)
        d_fake_error.backward()
        optimizer_D.step()

        # Train with gradient penalty
        #gradient_penalty = calc_gradeint_penalty(discriminator, input_img, target_img, 10)
        #gradient_penalty.backward()

        # Weight Clipping
        for p in discriminator.parameters():
            p.data.clamp_(-0.01, 0.01)

        # Train G
        generator.zero_grad()

        g_fake_data = generator(input_img)
        dg_fake_decision = discriminator(g_fake_data)
        g_error = -torch.mean(dg_fake_decision)

        # perceptual Loss
        fake_data_dup = d_fake_data.repeat(1,3,1,1)
        real_data_dup = target_img.repeat(1,3,1,1)
        fake_features = feature_extractor(fake_data_dup)
        real_features = Variable(feature_extractor(real_data_dup), requires_grad=False)
        perceptual_error = criterion_perceptual(fake_features, real_features)

        g_perceptual_error = g_error + (0.1 * perceptual_error)
        g_perceptual_error.backward()
        optimizer_G.step()

        if i % 10 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, 10, i, len(train_loader), loss_D.item(), loss_G.item()))

        if (epoch+1) % 10 == 0:
            torch.save(generator.state_dict(), "WGAN_VGG_{}ep.ckpt".format(epoch+1))
            torch.save(discriminator.state_dict(), "WGAN_VGG_{}ep.ckpt".format(epoch + 1))
