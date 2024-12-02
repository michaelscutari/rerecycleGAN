import torch
import pytorch_lightning as pl
from torch.nn import functional as F

from generator import ResNet, UNet
from discriminator import MultiScale

class RecycleGAN(pl.LightningModule):
    def __init__(self, l_adv, l_cycle, l_iden, l_temp, learning_rate_d, learning_rate_g):
        super(RecycleGAN, self).__init__()
        # generators
        self.AtoB = ResNet()
        self.BtoA = ResNet()
        self.nextA = UNet()
        self.nextB = UNet()

        # discriminators
        self.discriminatorA = MultiScale()
        self.discriminatorB = MultiScale()

        # loss weights
        self.l_adv = l_adv
        self.l_cycle = l_cycle
        self.l_iden = l_iden
        self.l_temp = l_temp

        # learning rates
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g

        # loss functions
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()
        self.temporal_loss = torch.nn.MSELoss()

    # forward pass
    def forward(self, real_a, real_b, real_a_prev, real_b_prev):
        # translate
        fake_b = self.AtoB(real_a)
        rec_a = self.BtoA(fake_b)
        fake_a = self.BtoA(real_b)
        rec_b = self.AtoB(fake_a)

        # temporal
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
        real_a, real_b, real_a_prev, real_b_prev, real_a_next, real_b_next = batch

        outputs = self.forward(real_a, real_b, real_a_prev, real_b_prev)
        fake_b = outputs['fake_b']
        rec_a = outputs['rec_a']
        fake_a = outputs['fake_a']
        rec_b = outputs['rec_b']
        pred_next_a = outputs['pred_next_a']
        pred_next_b = outputs['pred_next_b']

        # discriminator
        if optimizer_idx == 0:
            pred_real_a = self.discriminatorA(real_a)
            pred_fake_a = self.discriminatorA(fake_a.detach())
            loss_d_a_real = self.adversarial_loss(pred_real_a, torch.ones_like(pred_real_a))
            loss_d_a_fake = self.adversarial_loss(pred_fake_a, torch.zeros_like(pred_fake_a))
            loss_d_a = (loss_d_a_real + loss_d_a_fake) * 0.5

            pred_real_b = self.discriminatorB(real_b)
            pred_fake_b = self.discriminatorB(fake_b.detach())
            loss_d_b_real = self.adversarial_loss(pred_real_b, torch.ones_like(pred_real_b))
            loss_d_b_fake = self.adversarial_loss(pred_fake_b, torch.zeros_like(pred_fake_b))
            loss_d_b = (loss_d_b_real + loss_d_b_fake) * 0.5

            loss_d = (loss_d_a + loss_d_b) * 0.5

            self.log('loss_d', loss_d, prog_bar=True)
            return {'loss': loss_d}

        # generator
        if optimizer_idx == 1:
            pred_fake_a = self.discriminatorA(fake_a)
            pred_fake_b = self.discriminatorB(fake_b)
            loss_g_adv_a = self.adversarial_loss(pred_fake_a, torch.ones_like(pred_fake_a))
            loss_g_adv_b = self.adversarial_loss(pred_fake_b, torch.ones_like(pred_fake_b))
            loss_g_adv = (loss_g_adv_a + loss_g_adv_b) * 0.5 * self.l_adv

            loss_cycle_a = self.cycle_loss(rec_a, real_a) * self.l_cycle
            loss_cycle_b = self.cycle_loss(rec_b, real_b) * self.l_cycle
            loss_cycle = (loss_cycle_a + loss_cycle_b) * 0.5

            idt_a = self.BtoA(real_a)
            idt_b = self.AtoB(real_b)
            loss_idt_a = self.identity_loss(idt_a, real_a) * self.l_iden
            loss_idt_b = self.identity_loss(idt_b, real_b) * self.l_iden
            loss_idt = (loss_idt_a + loss_idt_b) * 0.5

            loss_temp_a = self.temporal_loss(pred_next_a, real_a_next) * self.l_temp
            loss_temp_b = self.temporal_loss(pred_next_b, real_b_next) * self.l_temp
            loss_temp = (loss_temp_a + loss_temp_b) * 0.5

            loss_g = loss_g_adv + loss_cycle + loss_idt + loss_temp

            self.log('loss_g', loss_g, prog_bar=True)
            return {'loss': loss_g}

    def configure_optimizers(self):
        # lr discriminator
        lr_d = self.learning_rate_d
        # lr generator
        lr_g = self.learning_rate_g

        # optimizer discriminator
        optimizer_d = torch.optim.Adam(
            list(self.discriminatorA.parameters()) + list(self.discriminatorB.parameters()),
            lr=lr_d, betas=(0.5, 0.999)
        )

        # optimizer generator
        optimizer_g = torch.optim.Adam(
            list(self.AtoB.parameters()) +
            list(self.BtoA.parameters()) +
            list(self.nextA.parameters()) +
            list(self.nextB.parameters()),
            lr=lr_g, betas=(0.5, 0.999)
        )

        # scheduler discriminator
        scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=10, gamma=0.1)

        # scheduler generator
        scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=10, gamma=0.1)

        # log lr discriminator
        self.log('lr_discriminator', lr_d, prog_bar=True)
        # log lr generator
        self.log('lr_generator', lr_g, prog_bar=True)

        return [optimizer_d, optimizer_g], [scheduler_d, scheduler_g]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
        # clip grads
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        # step optimizer
        optimizer.step(closure=optimizer_closure)