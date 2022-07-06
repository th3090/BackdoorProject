import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import torch.optim as optim
import random

from models.vgg import *
from utils.images import *
from attack.poison_frogs import feature_extractor

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)

img_size = (224, 224)
nc = 3
nz = 100  # Size of z latent vector ( size of generator input)
nfc = 4096  # Feature extractor에서 나오는 dimension의 수
ngf = 224  # Size of feature maps in generator
ndf = 224  # Size of feature maps in discriminator
num_epochs = 501
lr = 0.0001
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizers

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # nn.Linear(nz, nfc),
            # nn.ReLU(True),
            # nn.Dropout(),

            nn.ConvTranspose2d(nz, ngf * 16, 7, 1, 0, bias=False),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 224 x 224
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.main(input)
        out = out.squeeze()
        return out


def main():
    # image path
    path = './dataset/test/deep_poison'
    benign = './dataset/test/deep_poison/0_benign.jpg'
    poison = './dataset/test/deep_poison/0_poison.jpg'

    # image load
    benign_image = img_read(benign)
    poison_image = img_read(poison)

    # image to instance
    # benign_instance = img_to_tensor(benign_image, img_size)
    # poison_instance = img_to_tensor(poison_image, img_size)

    # benign_instance = benign_instance.to(device)
    # poison_instance = poison_instance.to(device)

    # Hyperparameter for Feature Extraction loss & Perturbation loss
    # loss_generator = loss_gan + alpha * loss_fe + beta * loss_pert
    alpha = 10
    beta = 3

    # Generator declare
    netG = Generator().to(device)
    netG.apply(weights_init)

    # Discriminator declare
    netD = Discriminator().to(device)
    netD.apply(weights_init)

    # Feature Extractor declare
    with torch.no_grad():
        model = vgg16(pretrained=True)
        model = feature_extractor('vggnet', model)
        model.to(device)
        model.eval()

    # Loss function define
    criterion_gan = nn.BCELoss()
    criterion_fe = nn.MSELoss()
    criterion_pert = nn.L1Loss()

    #  Initial noise
    fixed_noise = torch.randn(1, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    real_label = torch.tensor(real_label, dtype=torch.float)
    real_label = real_label.to(device)
    fake_label = 0.
    fake_label = torch.tensor(fake_label, dtype=torch.float)
    fake_label = fake_label.to(device)

    # Setup Adam optimizers for both G and D
    optimizer_d = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_g = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

    # list for visualization
    img_list = []
    g_losses = []
    d_losses = []
    test1 = []
    test2 = []
    test3 = []
    test4 = []
    test5 = []

    print("Starting Training Loop...")

    # # variable for model
    # poison_feature = model(poison_instance.view(1, *poison_instance.shape))
    # poison_feature.requires_grad = False

    for epoch in range(num_epochs):

        benign_instance = img_to_tensor(benign_image, img_size)
        poison_instance = img_to_tensor(poison_image, img_size)

        benign_instance = benign_instance.to(device)
        poison_instance = poison_instance.to(device)

        # variable for model
        poison_feature = model(poison_instance.view(1, *poison_instance.shape))
        benign_feature = model(benign_instance.view(1, *benign_instance.shape))
        # poison_feature.requires_grad = False

        # 1. Train Generator
        netG.train()
        optimizer_g.zero_grad()

        # 2. Perturbation loss
        # with torch.no_grad():
        perturbation = netG(fixed_noise)
        perturbation_squeeze = perturbation.squeeze()

        perturbation_feature = model(perturbation_squeeze.view(1, *perturbation_squeeze.shape))

        loss_pert = criterion_pert(benign_instance, perturbation_squeeze)

        # 3. Feature extract loss
        with torch.no_grad():
            poisoned_instance = perturbation_squeeze + benign_instance
            poisoned_feature = model(poisoned_instance.view(1, *poisoned_instance.shape))
        # poisoned_feature = model(poisoned_instance)

        # loss_pert = criterion_pert(benign_instance, poisoned_instance)
        loss_fe = criterion_fe(poisoned_feature, poison_feature)
        loss_test = criterion_fe(poisoned_feature, benign_feature)
        loss_test1 = criterion_fe(poison_feature, benign_feature)
        loss_test2 = criterion_fe(perturbation_feature, benign_feature)
        loss_test3 = criterion_fe(perturbation_feature, poison_feature)

        # 4. GAN loss
        loss_gan = criterion_gan(netD(poisoned_instance.view(1, *poisoned_instance.shape)), real_label)

        # 5. Generator's total Loss
        loss_generator = loss_gan + alpha * loss_fe + beta * loss_pert

        loss_generator.backward()
        optimizer_g.step()

        # del loss_fe
        # del loss_gan
        # del loss_pert
        # del perturbation
        #
        # torch.cuda.empty_cache()

        # ==============================================================

        # 1. Train Discriminator
        # if epoch % 50 == 0:
        netD.train()
        optimizer_d.zero_grad()

        # 2. Calculate the gan loss of real instance
        loss_real = criterion_gan(netD(benign_instance.view(1, *benign_instance.shape)), real_label)

        # 3. Calculate the gan loss of fake instance
        loss_fake = criterion_gan(netD(poisoned_instance.view(1, *poisoned_instance.shape)), fake_label)

        # 4. Discriminator's total Loss
        loss_discriminator = (loss_real + loss_fake) / 2

        # del poison_instance
        # del benign_instance
        #
        # torch.cuda.empty_cache()

        loss_discriminator.backward()

        optimizer_d.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                fake = poisoned_instance
                # img_list.append(fake)
                pert = perturbation_squeeze
                attack_image_save(pert, path, f"Deep_poisoned_perturbation_iter{epoch}.jpg")
                attack_image_save(fake, path, f"Deep_poisoned_instance_iter{epoch}.jpg")

            print(
                "[Epoch %d/%d] \n"
                "[D loss: %f] \n"
                "[G loss: %f] \n"
                % (epoch, num_epochs-1, loss_discriminator.item(), loss_generator.item())
            )

            print(f"loss_gan : {loss_gan}")
            print(f"loss_real : {loss_real}")
            print(f"loss_fake : {loss_fake}")
            print(f"Feature diff _ poison vs benign : {loss_test1}")
            print(f"Feature diff _ poisoned vs poison : {loss_fe}")
            print(f"Feature diff _ poisoned vs benign : {loss_test}")
            print(f"Feature diff _ perturbation vs poison : {loss_test3}")
            print(f"Feature diff _ perturbation vs benign : {loss_test2}")

        g_losses.append(loss_generator.item())
        d_losses.append(loss_discriminator.item()/10)

        test1.append(loss_test1.item())
        test2.append(loss_fe.item())
        test3.append(loss_test.item())
        test4.append(loss_test3.item())
        test5.append(loss_test2.item())

        # del poisoned_instance
        # torch.cuda.empty_cache()

    plt.figure(figsize=(10, 5))
    plt.title("Feature extractor l2 loss During Training")
    plt.plot(test1, label="P vs B")
    plt.plot(test2, label="Pd vs P")
    plt.plot(test3, label="Pd vs B")
    plt.plot(test4, label="Pe vs P")
    plt.plot(test5, label="Pe vs B")
    plt.xlabel("iterations")
    plt.ylabel("Feature diff")
    plt.legend()
    plt.show()

    # plt.figure(figsize=(10, 5))
    # plt.title("Generator and Discriminator Loss During Training")
    # plt.plot(g_losses, label="G")
    # plt.plot(d_losses, label="D")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    main()
