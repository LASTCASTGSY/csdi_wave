import numpy as np
import torch
import torch.nn as nn
from diff_models import diff_CSDI


class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional == True else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # ------------------------------------------------------------------
        # 1. FIXED DIFFUSION SCHEDULE (Standard DDPM / CSDI)
        # ------------------------------------------------------------------
        self.num_steps = config_diff["num_steps"]
        
        # Define Betas (Linear or Quad)
        if config_diff["schedule"] == "quad":
            self.betas = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.betas = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        # Define Alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas) # \bar{\alpha}
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        # Convert to Tensors for GPU access
        self.betas = torch.tensor(self.betas, dtype=torch.float32).to(self.device)
        self.alphas = torch.tensor(self.alphas, dtype=torch.float32).to(self.device)
        self.alphas_cumprod = torch.tensor(self.alphas_cumprod, dtype=torch.float32).to(self.device)
        self.alphas_cumprod_prev = torch.tensor(self.alphas_cumprod_prev, dtype=torch.float32).to(self.device)

        # Calculations for Posterior q(x_{t-1} | x_t, x_0) variance
        # sigma_t^2 = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
            
        # ------------------------------------------------------------------
        # 2. FIXED FORWARD PROCESS (Noise Injection)
        # ------------------------------------------------------------------
        # CRITICAL FIX: Must use alphas_cumprod (\bar{\alpha}_t) for t-step diffusion
        # The previous version likely used step-wise alphas here or undefined variables
        current_alpha_bar = self.alphas_cumprod[t].reshape(B, 1, 1) # (B, 1, 1)
        
        noise = torch.randn_like(observed_data)
        
        # x_t = sqrt(alpha_bar) * x_0 + sqrt(1 - alpha_bar) * epsilon
        noisy_data = (current_alpha_bar ** 0.5) * observed_data + (1.0 - current_alpha_bar) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples, chunk=4):
            """
            FIXED SAMPLER with drift prevention in conditional regions.
            """
            B, K, L = observed_data.shape
            result = torch.zeros(B, n_samples, K, L, device=self.device)
        
            for start in range(0, n_samples, chunk):
                end = min(start + chunk, n_samples)
                bs = end - start
        
                # Expand data for batch processing
                obs_rep = observed_data.unsqueeze(1).expand(-1, bs, -1, -1).reshape(B*bs, K, L)
                mask_rep = cond_mask.unsqueeze(1).expand(-1, bs, -1, -1).reshape(B*bs, K, L)
                side_rep = side_info.unsqueeze(1).expand(-1, bs, -1, -1, -1).reshape(
                    B*bs, side_info.shape[1], K, L
                )
        
                # Start from pure Gaussian noise
                x = torch.randn_like(obs_rep)
        
                for t in reversed(range(self.num_steps)):
                    # Get schedule parameters
                    beta_t = self.betas[t].view(1, 1, 1)
                    alpha_t = self.alphas[t].view(1, 1, 1)
                    alpha_bar_t = self.alphas_cumprod[t].view(1, 1, 1)
    
                    # Prepare input for model
                    cond_obs = (mask_rep * obs_rep).unsqueeze(1)
                    noisy_target = ((1 - mask_rep) * x).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)
                    
                    # Predict Noise
                    t_tensor = torch.tensor([t], device=self.device).repeat(B*bs)
                    predicted_noise = self.diffmodel(diff_input, side_rep, t_tensor)
    
                    # Reverse Step
                    coef1 = 1 / (alpha_t ** 0.5)
                    coef2 = beta_t / ((1 - alpha_bar_t) ** 0.5)
                    
                    mean = coef1 * (x - coef2 * predicted_noise)
                    
                    if t > 0:
                        sigma = self.posterior_variance[t].view(1, 1, 1) ** 0.5
                        noise = torch.randn_like(x)
                        x = mean + sigma * noise
                    else:
                        x = mean
                    
                    # --- CRITICAL FIX: Reset conditional part to valid noise or 0 ---
                    # This prevents values in the conditional mask from exploding due to drift.
                    # We essentially force the 'conditional' part of the latent x to stay bounded.
                    # Since x at cond locations is ignored by the model next step anyway, 
                    # we can set it to the noisy observed data (correct physics) or just 0.
                    
                    # Option 1 (cleaner for inspection): Keep observed data in conditional slots (visualizes better)
                    # x = x * (1 - mask_rep) + obs_rep * mask_rep
                    
                    # Option 2 (Minimal interference): Just prevent explosion by re-noising or clamping
                    # For simplicity, we can just let it be, but when saving result, we fill it with observed.
                
                # Final Clean-up: Fill the conditional slots with the actual observed data
                # This ensures the output tensor has valid ground truth in conditional slots
                # instead of the exploded garbage.
                x = x * (1 - mask_rep) + obs_rep * mask_rep
    
                result[:, start:end] = x.view(B, bs, K, L)
        
            return result

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_Wave(CSDI_base):
    """
    CSDI model for wave height imputation.
    Supports both NDBC1 (10-min) and NDBC2 (hourly) datasets.
    """
    def __init__(self, config, device, target_dim=9):
        super(CSDI_Wave, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        """
        Process batch data for wave dataset.
        Input shapes: (B, L, K) where B=batch, L=length, K=features
        Output shapes: (B, K, L) to match CSDI_base convention
        """
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        # Permute from (B, L, K) to (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Wave_Forecasting(CSDI_base):
    """
    CSDI model for wave height forecasting.
    Predicts future SWH values using completed/imputed data.
    """
    def __init__(self, config, device, target_dim=9):
        super(CSDI_Wave_Forecasting, self).__init__(target_dim, config, device)
        # For wave forecasting, we always use test pattern strategy
        # (known historical data, predict future SWH)
        self.target_strategy = "test_pattern"

    def process_data(self, batch):
        """
        Process batch data for wave forecasting.
        The gt_mask indicates which positions are targets (future SWH).
        """
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        # Permute from (B, L, K) to (B, K, L)
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )
    
    def forward(self, batch, is_train=1):
        """
        Forward pass for forecasting.
        Always use test pattern strategy (historical known, future unknown).
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
        ) = self.process_data(batch)
        
        # For forecasting, cond_mask = gt_mask (known historical data)
        cond_mask = gt_mask

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)