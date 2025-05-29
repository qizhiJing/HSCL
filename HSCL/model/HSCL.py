import torch
import torch.nn.functional as F
from torch import nn, einsum

class Encoder(nn.Module):
    def __init__(self, layers_per_network=3, hidden_sizes=[512, 256, 128], output_size=16):
        super(Encoder, self).__init__()

        # Create layers dynamically based on the specified depth (number of layers)
        self.encoder_mri = self._create_encoder(90, layers_per_network, hidden_sizes, output_size)
        self.encoder_pet = self._create_encoder(90, layers_per_network, hidden_sizes, output_size)
        self.encoder_csf = self._create_encoder(3, layers_per_network, hidden_sizes,
                                                output_size)  # CSF has 3 input features

    def _create_encoder(self, input_size, layers_per_network, hidden_sizes, output_size):
        layers = []
        current_size = input_size

        # Create hidden layers dynamically
        for i in range(layers_per_network):
            hidden_size = hidden_sizes[i] if i < len(hidden_sizes) else hidden_sizes[-1]
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_size = hidden_size

        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.BatchNorm1d(output_size))

        return nn.Sequential(*layers)

    def forward(self, mri, pet, csf):
        # Return the latent representations for MRI, PET, and CSF
        return self.encoder_mri(mri), self.encoder_pet(pet), self.encoder_csf(csf)

class HSCL(nn.Module):
    def __init__(self):
        super(HSCL, self).__init__()

        self.encoder = Encoder()
        self.crossentropy = nn.CrossEntropyLoss()
        self.lp = nn.Sequential(
            nn.Linear(32, 2),
            nn.Softmax(dim=1)
        )
        self.pj_mri = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16)
        )
        self.pj_csf = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16)
        )
        self.pj_pet = nn.Sequential(
            nn.Linear(16, 16),
            nn.BatchNorm1d(16)
        )

    def log(self, t, eps=1e-20):
        return torch.log(t + eps)

    def l2norm(self, t):
        return F.normalize(t, dim=-1, p=2)

    def scl_intra(self, A, y):
        # Calculate similarity matrix
        A = self.l2norm(A)
        tau = 0.1
        similarity = einsum('md, nd -> mn', A, A) / tau

        # Set diagonal elements to very large negative value
        torch.diagonal(similarity).fill_(float('-inf'))

        # Exponentiate the similarity matrix
        similarity_exp = torch.exp(similarity)

        # Create mask for positive samples based on labels
        label_mask = (y.unsqueeze(0) == y.unsqueeze(1)).int()

        # Extract positive samples (mask elements)
        similarity_pos = (similarity_exp * label_mask).sum(dim=-1)

        # Compute denominator by summing over each row
        similarity_denom = similarity_exp.sum(dim=-1)

        # Calculate the contrastive loss
        loss = -self.log(similarity_pos / similarity_denom).mean()

        return loss

    def scl_inter(self, latents_a, latents_b, y):
        tau = 0.1  # Temperature for similarity scaling
        label_mask = (y.unsqueeze(0) == y.unsqueeze(1)).int()

        latents_a, latents_b = map(self.l2norm, (latents_a, latents_b))

        # Calculate similarity matrices
        a_to_b = einsum('md, nd -> mn', latents_a, latents_b) / tau
        b_to_a = einsum('md, nd -> mn', latents_b, latents_a) / tau

        # Exponentiate the similarity matrices
        a_to_b_exp = torch.exp(a_to_b)
        b_to_a_exp = torch.exp(b_to_a)

        # Extract positive samples (mask elements)
        a_to_b_pos = (a_to_b_exp * label_mask).sum(dim=-1)
        b_to_a_pos = (b_to_a_exp * label_mask).sum(dim=-1)

        # Compute denominators by summing over each row
        a_to_b_denom = a_to_b_exp.sum(dim=-1)
        b_to_a_denom = b_to_a_exp.sum(dim=-1)

        # Calculate the contrastive losses
        a_to_b_loss = -self.log(a_to_b_pos / a_to_b_denom).mean()
        b_to_a_loss = -self.log(b_to_a_pos / b_to_a_denom).mean()

        # Symmetric contrastive loss
        loss = a_to_b_loss + b_to_a_loss

        return loss

    def forward(self, *, mri, pet, csf, y, lambda_):

        mri_latents, pet_latents, csf_latents = self.encoder(mri=mri, pet=pet, csf=csf)
        mri_latents_pj, pet_latents_pj, csf_latents_pj = self.pj_mri(mri_latents), self.pj_pet(pet_latents), self.pj_csf(csf_latents)


        loss_intra_mri = self.scl_intra(mri_latents, y)
        loss_intra_pet = self.scl_intra(pet_latents, y)
        loss_intra_csf = self.scl_intra(csf_latents, y)
        loss_intra = (loss_intra_mri + loss_intra_pet + loss_intra_csf)

        loss_inter_mri_pet = self.scl_inter(mri_latents_pj, pet_latents_pj, y)
        loss_inter_mri_csf = self.scl_inter(csf_latents_pj, pet_latents_pj, y)
        loss_inter_pet_csf = self.scl_inter(mri_latents_pj, csf_latents_pj, y)

        loss_inter = loss_inter_mri_pet + loss_inter_mri_csf + loss_inter_pet_csf

        output = self.lp(torch.concatenate((mri_latents, pet_latents), dim=1))
        loss_classification = self.crossentropy(output, y)

        loss_total = loss_classification + (1 - lambda_) * loss_inter + lambda_ * loss_intra

        return loss_total, output, loss_classification










