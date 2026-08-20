
import math

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
import time
import numpy as np
from .nets import device


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        # Store float directly in ctx instead of creating a tensor
        ctx.lambda_ = lambda_ 
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Multiply by negative lambda
        return grad_output.neg() * ctx.lambda_, None

def grad_reverse(x, lambda_):
    return GradientReversalFunction.apply(x, lambda_)


def train_one_epoch_dann(self, target_loader):
    """
    Differences from train_one_epoch_with_new_discriminator:
      1. A single backward pass per batch unifies the regression and
         domain-adaptation objectives, matching the DANN saddle-point
         formulation exactly.
      2. Both source and target latent vectors are passed through the
         Gradient Reversal Layer before the discriminator, so the encoder
         receives adversarial gradients from both domains rather than
         target only.
      3. Lambda follows the DANN exponential schedule (Eq. 14 in the paper),
         growing from 0 to 1 over training to suppress noisy discriminator
         signal in early epochs.
      4. opt_regression and opt_discriminator are both stepped after the
         single backward call.  opt_generator is not used.
    """
    start = time.time()
    reg_losses = []
    recon_losses = []
    phase1_losses = []
    disc_losses = []
    disc_grad_norms = []
    gen_losses = []
    disc_accs = []
    disc_acc_source = []
    disc_acc_target = []
    gen_fool_accs = []

    target_iter = iter(target_loader)
    num_batches = len(self.trainloader)
    max_epochs = getattr(self, "max_epochs", 100)
    lambda_p = 0.0

    for batch_idx, (xbatch_source, ybatch_source, info_source) in \
            enumerate(self.trainloader):

        ybatch_source = ybatch_source.to(device, non_blocking=True)

        try:
            xbatch_target, _, _ = next(target_iter)
        except StopIteration:
            target_iter = iter(target_loader)
            xbatch_target, _, _ = next(target_iter)

        source_batch_size = ybatch_source.shape[0]
        if isinstance(xbatch_target, dict):
            target_batch_size = xbatch_target["flow"].shape[0]
        else:
            target_batch_size = xbatch_target.shape[0]

        current_batch_size = min(source_batch_size, target_batch_size)
        if current_batch_size <= 0:
            continue

        xbatch_source = self._slice_batch_inputs(xbatch_source, current_batch_size)
        ybatch_source = ybatch_source[:current_batch_size]
        info_source = self._slice_batch_info(info_source, current_batch_size)
        xbatch_target = self._slice_batch_inputs(xbatch_target, current_batch_size)

        xbatch_source = self._apply_training_batch_transforms(xbatch_source)
        xbatch_target = self._apply_training_batch_transforms(xbatch_target)

        # ── Dynamic lambda schedule (DANN Eq. 14) ─────────────────────────
        # p increases linearly from 0 to 1 over the full training run.
        # lambda_p grows from 0 to 1 following a sigmoid-like curve,
        # suppressing the adversarial signal in early epochs and
        # amplifying it as training stabilises.
        p = (self.epoch * num_batches + batch_idx) / \
            float(max_epochs * num_batches)
        lambda_p = 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

        # ── Single unified forward + backward pass ─────────────────────────
        self.opt_regression.zero_grad()
        self.opt_discriminator.zero_grad()

        # Encode both domains.
        pred_source, z_source = self.net.forward_with_latent(xbatch_source)
        _, z_target = self.net.forward_with_latent(xbatch_target)
        pred = pred_source.squeeze(1)

        if self.subplan_level_outputs:
            idxs = torch.zeros(pred.shape, dtype=torch.bool)
            for i, nt in enumerate(info_source["num_tables"]):
                if nt >= 10:
                    nt = 10
                nt -= 1
                idxs[i, nt] = True
            pred = pred[idxs]

        # Regression loss on labeled source queries only.
        loss_reg = self._compute_regression_loss(pred, ybatch_source, info_source)

        # Domain adaptation loss via GRL.
        # Applying the GRL to both z_source and z_target means:
        #   - The discriminator receives normal gradients (it sees clean z).
        #   - The encoder receives reversed gradients (it is pushed to
        #     confuse the discriminator on both domains), matching the
        #     DANN saddle-point update in Eq. (4) of the paper.
        z_source_grl = grad_reverse(z_source, lambda_p)
        z_target_grl = grad_reverse(z_target, lambda_p)

        labels_source = torch.ones(current_batch_size, 1, device=device)
        labels_target = torch.zeros(current_batch_size, 1, device=device)

        pred_source_disc = self.net.discriminate(z_source_grl)
        pred_target_disc = self.net.discriminate(z_target_grl)

        loss_d_source = self.bce_loss(pred_source_disc, labels_source)
        loss_d_target = self.bce_loss(pred_target_disc, labels_target)
        loss_d = 0.5 * (loss_d_source + loss_d_target)

        # Total loss: regression + reconstruction + domain adaptation.
        # Because the GRL is already inside z_source_grl / z_target_grl,
        # adding loss_d here causes:
        #   - discriminator params: normal gradient from loss_d
        #   - encoder params:       regression gradient from loss_reg
        #                         + REVERSED domain gradient from loss_d
        total_loss = loss_reg + loss_d
        total_loss.backward()

        disc_grad_sq = 0.0
        for param in self.net.discriminator.parameters():
            if param.grad is None:
                continue
            disc_grad_sq += param.grad.detach().pow(2).sum().item()
        disc_grad_norm = math.sqrt(disc_grad_sq)

        if self.clip_gradient is not None:
            clip_grad_norm_(
                self.opt_regression.param_groups[0]["params"],
                self.clip_gradient,
            )

        # Both optimizers are stepped from the same backward pass.
        self.opt_regression.step()
        self.total_grad_steps += 1
        self.opt_discriminator.step()

        # ── Metrics (no_grad) ──────────────────────────────────────────────
        with torch.no_grad():
            disc_pred_source_cls = (pred_source_disc >= 0.5).float()
            disc_pred_target_cls = (pred_target_disc >= 0.5).float()

            src_acc = (disc_pred_source_cls == labels_source).float().mean().item()
            tgt_acc = (disc_pred_target_cls == labels_target).float().mean().item()

            combined_preds  = torch.cat([disc_pred_source_cls, disc_pred_target_cls], dim=0)
            combined_labels = torch.cat([labels_source, labels_target], dim=0)
            d_acc = (combined_preds == combined_labels).float().mean().item()

            # Fool accuracy: fraction of target samples classified as source.
            fool_acc = (pred_target_disc >= 0.5).float().mean().item()

        reg_losses.append(loss_reg.item())
        disc_losses.append(loss_d.item())
        disc_grad_norms.append(disc_grad_norm)
        gen_losses.append(loss_d.item())   # same signal seen by encoder
        disc_accs.append(d_acc)
        disc_acc_source.append(src_acc)
        disc_acc_target.append(tgt_acc)
        gen_fool_accs.append(fool_acc)

    metrics = {
        "loss_reg":       float(np.mean(reg_losses))   if reg_losses   else 0.0,
        "loss_recon":     float(np.mean(recon_losses)) if recon_losses else 0.0,
        "loss_phase1":    float(np.mean(phase1_losses)) if phase1_losses else 0.0,
        "loss_d":         float(np.mean(disc_losses))  if disc_losses  else 0.0,
        "disc_grad_norm": float(np.mean(disc_grad_norms)) if disc_grad_norms else 0.0,
        "loss_g":         float(np.mean(gen_losses))   if gen_losses   else 0.0,
        "disc_acc":       float(np.mean(disc_accs))    if disc_accs    else 0.0,
        "disc_acc_source":float(np.mean(disc_acc_source)) if disc_acc_source else 0.0,
        "disc_acc_target":float(np.mean(disc_acc_target)) if disc_acc_target else 0.0,
        "gen_fool_acc":   float(np.mean(gen_fool_accs)) if gen_fool_accs else 0.0,
        "epoch_seconds":  round(time.time() - start, 2),
        "lambda_p":       round(lambda_p, 4),
    }
    return metrics
