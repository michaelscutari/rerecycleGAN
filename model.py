import torch
import pytorch_lightning as pl
from torch.nn import functional as F
import torch.nn as nn
from torchvision.utils import make_grid
import wandb

from generators import ResNet, UNet
from discriminator import PatchGAN, MultiScaleDiscriminator

class RecycleGAN(pl.LightningModule):
    def __init__(self, l_adv, l_cycle, l_iden, l_temp, learning_rate_d, learning_rate_g, learning_rate_p):
        super(RecycleGAN, self).__init__()
        self.automatic_optimization = False
        
        self.AtoB = UNet()
        self.BtoA = UNet()
        self.nextA = ResNet(num_residual_blocks=8)
        self.nextB = ResNet(num_residual_blocks=8)
        self.discriminatorA = MultiScaleDiscriminator()
        self.discriminatorB = MultiScaleDiscriminator()

        self.AtoB.apply(self.init_weights)
        self.BtoA.apply(self.init_weights)
        self.nextA.apply(self.init_weights)
        self.nextB.apply(self.init_weights)
        self.discriminatorA.apply(self.init_weights)
        self.discriminatorB.apply(self.init_weights)
        
        self.l_adv = l_adv
        self.l_cycle = l_cycle
        self.l_iden = l_iden
        self.l_temp = l_temp
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g
        self.learning_rate_p = learning_rate_p
        
        self.adversarial_loss = torch.nn.BCEWithLogitsLoss()
        self.cycle_loss = torch.nn.L1Loss()
        self.identity_loss = torch.nn.L1Loss()
        self.recycle_loss = torch.nn.MSELoss()
        self.recurrent_loss = torch.nn.L1Loss()

        self.fixed_real_a = None  # Placeholder for fixed samples
        self.fixed_real_b = None

    def on_train_start(self):
        # Save a fixed set of inputs
        dataset = self.trainer.datamodule.train_dataloader().dataset
        real_a, real_b, real_a_prev, real_b_prev, real_a_next, real_b_next = dataset[0]
        self.fixed_real_a = real_a.unsqueeze(0).to(self.device)
        self.fixed_real_b = real_b.unsqueeze(0).to(self.device)

        # Log the learning rates
        self.log('lr_discriminator', self.learning_rate_d)
        self.log('lr_generator', self.learning_rate_g)
    

    def on_train_epoch_end(self):
        if self.current_epoch % 5 == 0:
            # DEBUG
            print("Epoch end")
            # Generate outputs using fixed inputs
            with torch.no_grad():
                self.eval()  # Switch to evaluation mode
                outputs = self(self.fixed_real_a, self.fixed_real_b, self.fixed_real_a, self.fixed_real_b)

                # DEBUG
                print("Outputs generated")

                # Denormalize images for logging
                real_a = self.denormalize(self.fixed_real_a.cpu())
                fake_b = self.denormalize(outputs['fake_b'].cpu())
                rec_a = self.denormalize(outputs['rec_a'].cpu())
                real_b = self.denormalize(self.fixed_real_b.cpu())
                fake_a = self.denormalize(outputs['fake_a'].cpu())
                rec_b = self.denormalize(outputs['rec_b'].cpu())

                # DEBUG
                print("Images denormalized")

                # Create grids
                grid_real_a = make_grid(real_a, nrow=4, normalize=True)
                grid_fake_b = make_grid(fake_b, nrow=4, normalize=True)
                grid_rec_a = make_grid(rec_a, nrow=4, normalize=True)
                grid_real_b = make_grid(real_b, nrow=4, normalize=True)
                grid_fake_a = make_grid(fake_a, nrow=4, normalize=True)
                grid_rec_b = make_grid(rec_b, nrow=4, normalize=True)

                # DEBUG
                print("Grids created")

                # Log the images
                self.logger.experiment.log({
                    'Fixed Real A': [wandb.Image(grid_real_a, caption="Fixed Real A")],
                    'Fixed Fake B': [wandb.Image(grid_fake_b, caption="Fixed Fake B")],
                    'Fixed Rec A': [wandb.Image(grid_rec_a, caption="Fixed Rec A")],
                    'Fixed Real B': [wandb.Image(grid_real_b, caption="Fixed Real B")],
                    'Fixed Fake A': [wandb.Image(grid_fake_a, caption="Fixed Fake A")],
                    'Fixed Rec B': [wandb.Image(grid_rec_b, caption="Fixed Rec B")],
                    'epoch': self.current_epoch
                })
                self.train()  # Switch back to training mode

                # DEBUG
                print("Images logged")

    def forward(self, real_a=None, real_b=None):

        # generate fake a and b for use in other losses
        if real_a:
            fake_a = self.BtoA(real_b)
        if real_b:
            fake_b = self.AtoB(real_a)

        return {
            'fake_a': fake_a,
            'fake_b': fake_b,
        }
    
    def training_forward(self, real_a, real_b):

        # generate fake a and b for use in other losses
        fake_a = self.BtoA(real_b)
        fake_b = self.AtoB(real_a)

        # recurrent predictions
        pred_next_a = self.nextA(real_a)
        pred_next_b = self.nextB(real_b)

        # cycle loss
        cycle_a = self.BtoA(fake_b)
        cycle_b = self.AtoB(fake_a)

        # recycle loss
        with torch.no_grad():
            fake_next_a = self.nextA(fake_a.detach()) # P_A(f_A)
            fake_next_b = self.nextB(fake_b.detach()) # P_B(f_B)

        recycle_next_a = self.BtoA(fake_next_b.detach()) 
        recycle_next_b = self.AtoB(fake_next_a.detach())

        # identity
        allegedly_same_a = self.BtoA(real_a)
        allegedly_same_b = self.AtoB(real_b)

        return  {
            'cycle_a': cycle_a,
            'cycle_b': cycle_b,
            'pred_next_a': pred_next_a,
            'pred_next_b': pred_next_b,
            'fake_a': fake_a,
            'fake_b': fake_b,
            'recycle_next_a': recycle_next_a,
            'recycle_next_b': recycle_next_b,
            'allegedly_same_a': allegedly_same_a,
            'allegedly_same_b': allegedly_same_b
        }


    def training_step(self, batch, batch_idx):
        opt_d, opt_g, opt_p = self.optimizers()
        
        real_a, real_b, real_a_prev, real_b_prev, real_a_next, real_b_next = batch
        # generate images
        cycle_a, cycle_b, pred_next_a, pred_next_b, fake_a, fake_b, recycle_next_a, recycle_next_b, allegedly_same_a, allegedly_same_b = self.training_forward(real_a, real_b).values()

        # ------------------
        # Discriminator Training
        # ------------------
        opt_d.zero_grad()
        
        # Discriminator A
        pred_real_a = self.discriminatorA(real_a)
        pred_fake_a = self.discriminatorA(fake_a.detach())

        loss_d_a_real = self.compute_adversarial_loss(pred_real_a, self.make_ones_targets(pred_real_a))
        loss_d_a_fake = self.compute_adversarial_loss(pred_fake_a, self.make_zeros_targets(pred_fake_a))
        loss_d_a = (loss_d_a_real + loss_d_a_fake) * 0.5

        # Discriminator B
        pred_real_b = self.discriminatorB(real_b)
        pred_fake_b = self.discriminatorB(fake_b.detach())

        loss_d_b_real = self.compute_adversarial_loss(pred_real_b, self.make_ones_targets(pred_real_b))
        loss_d_b_fake = self.compute_adversarial_loss(pred_fake_b, self.make_zeros_targets(pred_fake_b))
        loss_d_b = (loss_d_b_real + loss_d_b_fake) * 0.5

        # Total Discriminator loss
        loss_d = (loss_d_a + loss_d_b) * 0.5

        self.manual_backward(loss_d)

        torch.nn.utils.clip_grad_norm_(self.discriminatorA.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.discriminatorB.parameters(), max_norm=1.0)

        opt_d.step()

        # ------------------
        # Generator Training
        # ------------------
        opt_g.zero_grad()
        
        # Get discriminator outputs for fake images
        pred_fake_a = self.discriminatorA(fake_a)
        pred_fake_b = self.discriminatorB(fake_b)

        # Adversarial loss for generators
        loss_g_adv_a = self.compute_adversarial_loss(pred_fake_a, self.make_ones_targets(pred_fake_a))
        loss_g_adv_b = self.compute_adversarial_loss(pred_fake_b, self.make_ones_targets(pred_fake_b))
        loss_g_adv = (loss_g_adv_a + loss_g_adv_b) * 0.5

        # cycle loss for generators
        loss_cycle_a = self.cycle_loss(cycle_a, real_a) * self.l_cycle
        loss_cycle_b = self.cycle_loss(cycle_b, real_b) * self.l_cycle
        loss_cycle = (loss_cycle_a + loss_cycle_b) * 0.5

        # identity loss for generators
        loss_idt_a = self.identity_loss(allegedly_same_a, real_a) * self.l_iden
        loss_idt_b = self.identity_loss(allegedly_same_b, real_b) * self.l_iden
        loss_idt = (loss_idt_a + loss_idt_b) * 0.5

        # recycle loss for generators
        loss_recycle_a = self.recycle_loss(recycle_next_a, real_a_next) * self.l_temp
        loss_recycle_b = self.recycle_loss(recycle_next_b, real_b_next) * self.l_temp
        loss_recycle = (loss_recycle_a + loss_recycle_b) * 0.5

        loss_g = loss_g_adv + loss_cycle + loss_idt + loss_recycle
        self.manual_backward(loss_g)
        
        torch.nn.utils.clip_grad_norm_(self.AtoB.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.BtoA.parameters(), max_norm=1.0)


        opt_g.step()

        # ------------------
        # Predictor Training
        # ------------------
        opt_p.zero_grad()

        # preidctor loss
        loss_p_a = self.recurrent_loss(pred_next_a, pred_next_b)
        loss_p_b = self.recurrent_loss(pred_next_b, pred_next_a)
        loss_temp = (loss_p_a + loss_p_b) * 0.5

        # scale
        loss_temp = loss_temp * self.l_temp

        # backprop
        self.manual_backward(loss_temp)
        torch.nn.utils.clip_grad_norm_(self.nextA.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.nextB.parameters(), max_norm=1.0)

        # update
        opt_p.step()

        # ------------------
        # Logging
        # ------------------

        # Logging
        self.log_dict({
            'loss_d': loss_d,
            'loss_g': loss_g,
            'loss_g_adv': loss_g_adv,
            'loss_cycle': loss_cycle,
            'loss_idt': loss_idt,
            'loss_temp': loss_temp
        }, prog_bar=True)

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(
            list(self.discriminatorA.parameters()) + 
            list(self.discriminatorB.parameters()),
            lr=self.learning_rate_d, betas=(0.5, 0.999)
        )
        
        opt_g = torch.optim.Adam(
            list(self.AtoB.parameters()) +
            list(self.BtoA.parameters()),
            lr=self.learning_rate_g, betas=(0.5, 0.999)
        )

        opt_p = torch.optim.Adam(
            list(self.nextA.parameters()) +
            list(self.nextB.parameters()),
            lr=self.learning_rate_p, betas=(0.5, 0.999)
        )
        
        sched_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=10, gamma=0.1)
        sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=10, gamma=0.1)
        sched_p = torch.optim.lr_scheduler.StepLR(opt_p, step_size=10, gamma=0.1)

        self.logger.log_hyperparams({
        'lr_discriminator': self.learning_rate_d,
        'lr_generator': self.learning_rate_g,
        'lr_predictor': self.learning_rate_p,
        'l_adv': self.l_adv,
        'l_cycle': self.l_cycle,
        'l_iden': self.l_iden,
        'l_temp': self.l_temp,
        })
        
        return [opt_d, opt_g, opt_p], [sched_d, sched_g, sched_p]

    def compute_adversarial_loss(self, preds, targets):
        # preds and targets are lists of tensors from multi-scale discriminator
        loss = 0
        for pred, target in zip(preds, targets):
            loss += self.adversarial_loss(pred, target)
        return loss / len(preds)  # average across scales

    def make_ones_targets(self, preds):
        return [torch.ones_like(pred) for pred in preds]

    def make_zeros_targets(self, preds):
        return [torch.zeros_like(pred) for pred in preds]
        
    def smooth_ones_like(self, tensor):
        ones = torch.ones_like(tensor)
        return ones * 0.9
    
    def smooth_zeros_like(self, tensor):
        zeros = torch.zeros_like(tensor)
        return zeros + 0.1

    # Denormalize!
    def denormalize(self, tensor):
        return tensor * 0.5 + 0.5
    
    def init_weights(self, module):
        """
        Initialize the weights of layers in the model using Kaiming He initialization.
        """
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm2d):
            if module.weight is not None:
                nn.init.ones_(module.weight)  # Gamma is initialized to 1
            if module.bias is not None:
                nn.init.zeros_(module.bias)