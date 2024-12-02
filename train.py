import torch
import pytorch_lightning as pl
from torch.nn import functional as F

from generator import ResNet, UNet
from discriminator import PatchGAN, MultiScale

class RecycleGAN(pl.LightningModule):
    def __init__(self, l_adv, l_cycle, l_iden, l_temp, learning_rate):
        super(RecycleGAN, self).__init__()
        # Generators
        self.AtoB = ResNet()
        self.BtoA = ResNet()
        self.nextA = UNet()
        self.nextB = UNet()

        # Discriminators
        self.discriminatorA = MultiScale()
        self.discriminatorB = MultiScale()

        # Hyperparameters for loss
        self.l_adv = l_adv
        self.l_cycle = l_cycle
        self.l_iden = l_iden
        self.l_temp = l_temp

        # Learning rate
        self.learning_rate = learning_rate

        # Loss functions
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()
        self.temporal_loss = torch.nn.MSELoss()  # L2 Loss

    # Forward pass
    def forward(self, real_a, real_b, real_a_prev, real_b_prev):
        # Translate images between domains
        fake_b = self.AtoB(real_a)
        rec_a = self.BtoA(fake_b)
        fake_a = self.BtoA(real_b)
        rec_b = self.AtoB(fake_a)

        # Temporal consistency
        # Concatenate previous real frame with current generated frame
        input_next_a = torch.cat([real_a_prev, fake_a], dim=1)
        pred_next_a = self.nextA(input_next_a)
        input_next_b = torch.cat([real_b_prev, fake_b], dim=1)
        pred_next_b = self.nextB(input_next_b)

        return {
            'fake_b': fake_b,
            'rec_a': rec_a,
            'fake_a': fake_a,
            'rec_b': rec_b,
            'pred_next_a': pred_next_a,
            'pred_next_b': pred_next_b
        }

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_a, real_b, real_a_prev, real_b_prev = batch

        outputs = self.forward(real_a, real_b, real_a_prev, real_b_prev)
        fake_b = outputs['fake_b']
        rec_a = outputs['rec_a']
        fake_a = outputs['fake_a']
        rec_b = outputs['rec_b']
        pred_next_a = outputs['pred_next_a']
        pred_next_b = outputs['pred_next_b']

        # **Discriminator training**
        if optimizer_idx == 0:
            # Discriminator A
            pred_real_a = self.discriminatorA(real_a)
            pred_fake_a = self.discriminatorA(fake_a.detach())
            loss_d_a_real = self.adversarial_loss(pred_real_a, torch.ones_like(pred_real_a))
            loss_d_a_fake = self.adversarial_loss(pred_fake_a, torch.zeros_like(pred_fake_a))
            loss_d_a = (loss_d_a_real + loss_d_a_fake) * 0.5

            # Discriminator B
            pred_real_b = self.discriminatorB(real_b)
            pred_fake_b = self.discriminatorB(fake_b.detach())
            loss_d_b_real = self.adversarial_loss(pred_real_b, torch.ones_like(pred_real_b))
            loss_d_b_fake = self.adversarial_loss(pred_fake_b, torch.zeros_like(pred_fake_b))
            loss_d_b = (loss_d_b_real + loss_d_b_fake) * 0.5

            # Total discriminator loss
            loss_d = (loss_d_a + loss_d_b) * 0.5

            self.log('loss_d', loss_d, prog_bar=True)
            return {'loss': loss_d}

        # **Generator training**
        if optimizer_idx == 1:
            # Adversarial loss
            pred_fake_a = self.discriminatorA(fake_a)
            pred_fake_b = self.discriminatorB(fake_b)
            loss_g_adv_a = self.adversarial_loss(pred_fake_a, torch.ones_like(pred_fake_a))
            loss_g_adv_b = self.adversarial_loss(pred_fake_b, torch.ones_like(pred_fake_b))
            loss_g_adv = (loss_g_adv_a + loss_g_adv_b) * 0.5 * self.l_adv

            # Cycle consistency loss
            loss_cycle_a = self.cycle_loss(rec_a, real_a) * self.l_cycle
            loss_cycle_b = self.cycle_loss(rec_b, real_b) * self.l_cycle
            loss_cycle = (loss_cycle_a + loss_cycle_b) * 0.5

            # Identity loss
            idt_a = self.BtoA(real_a)
            idt_b = self.AtoB(real_b)
            loss_idt_a = self.identity_loss(idt_a, real_a) * self.l_iden
            loss_idt_b = self.identity_loss(idt_b, real_b) * self.l_iden
            loss_idt = (loss_idt_a + loss_idt_b) * 0.5

            # Temporal loss
            loss_temp_a = self.temporal_loss(pred_next_a, real_a) * self.l_temp
            loss_temp_b = self.temporal_loss(pred_next_b, real_b) * self.l_temp
            loss_temp = (loss_temp_a + loss_temp_b) * 0.5

            # Total generator loss
            loss_g = loss_g_adv + loss_cycle + loss_idt + loss_temp

            self.log('loss_g', loss_g, prog_bar=True)
            return {'loss': loss_g}

    def configure_optimizers(self):
        # Separate optimizers for generators and discriminators
        lr = self.learning_rate

        # Optimizer for discriminators
        optimizer_d = torch.optim.Adam(
            list(self.discriminatorA.parameters()) + list(self.discriminatorB.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )

        # Optimizer for generators
        optimizer_g = torch.optim.Adam(
            list(self.AtoB.parameters()) +
            list(self.BtoA.parameters()) +
            list(self.nextA.parameters()) +
            list(self.nextB.parameters()),
            lr=lr, betas=(0.5, 0.999)
        )

        return [optimizer_d, optimizer_g], []

    def discriminator_loss(self, pred_real, pred_fake):
        # Helper function to compute discriminator loss
        loss_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))
        loss_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))
        return (loss_real + loss_fake) * 0.5