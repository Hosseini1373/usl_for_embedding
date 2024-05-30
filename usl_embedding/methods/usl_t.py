import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import Adam

from usl_embedding.models.ssl_t_models.clustering_model import ClusteringModel









def get_device():
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'
    print("Device: ", device)
    return device
    
    

def find_duplicates(input_list):
    seen = set()
    duplicates = set()
    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)
    return list(duplicates)




###### Loss functions for USL-t:-----------------

# Credit to PAWS: https://github.com/facebookresearch/suncet/blob/main/src/losses.py
def sharpen(p, T):  # T: sharpen temperature
    sharp_p = p ** (1. / T)
    sharp_p = sharp_p / torch.sum(sharp_p, dim=1, keepdim=True)
    return sharp_p




class OursLossLocal(nn.Module):
    def __init__(self, num_classes, num_heads, momentum=None, adjustment_weight=None, sharpen_temperature=None):
        super(OursLossLocal, self).__init__()
        self.momentum = momentum

        self.adjustment_weight = adjustment_weight

        self.num_heads = num_heads

        self.register_buffer("prob_ema", torch.ones(
            (num_heads, num_classes)) / num_classes)

        self.sharpen_temperature = sharpen_temperature

    def forward(self, head_id, anchors, neighbors):
        # This is ours v2 with multi_headed prob_ema support
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        head_prob_ema = self.prob_ema[head_id]
        neighbors_adjusted = neighbors - self.adjustment_weight * \
            torch.log(head_prob_ema).view((1, -1))

        anchors_prob = F.softmax(anchors, dim=1)
        positives_prob = F.softmax(neighbors_adjusted, dim=1)
        log_anchors_prob = F.log_softmax(anchors, dim=1)

        positives_original_prob = F.softmax(neighbors, dim=1)
        head_prob_ema = head_prob_ema * self.momentum + \
            positives_original_prob.detach().mean(dim=0) * (1 - self.momentum)
        head_prob_ema = head_prob_ema / head_prob_ema.sum()

        self.prob_ema[head_id] = head_prob_ema

        consistency_loss = F.kl_div(log_anchors_prob, sharpen(
            positives_prob.detach(), T=self.sharpen_temperature), reduction="batchmean")

        # Total loss
        total_loss = consistency_loss

        return total_loss


class OursLossGlobal(nn.Module):
    # From ConfidenceBasedCE
    def __init__(self, threshold, reweight, num_classes, num_heads, mean_outside_mask=False, use_count_ema=False, momentum=0., data_len=None, reweight_renorm=False):
        super(OursLossGlobal, self).__init__()
        self.threshold = threshold
        self.reweight = reweight
        # setting reweight_renorm to True ignores reweight
        self.reweight_renorm = reweight_renorm

        if self.reweight_renorm:
            print("Reweight renorm is enabled")
        else:
            print("Reweight renorm is not enabled")

        self.mean_outside_mask = mean_outside_mask
        self.use_count_ema = use_count_ema

        self.num_classes = num_classes
        self.num_heads = num_heads

        self.momentum = momentum

        if use_count_ema:
            print("Data length:", data_len)
            self.data_len = data_len
            self.register_buffer("count_ema", torch.ones(
                (num_heads, num_classes)) / num_classes)
        self.register_buffer("num_counts", torch.zeros(1, dtype=torch.long))

    # Equivalent to: https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html
    # With one-hot target
    def kl_div_loss(self, input, target, mask, weight, mean_outside_mask):
        if torch.all(mask == 0):
            # Return 0 as loss if nothing is in mask
            return torch.tensor(0., device=input.device)

        b = input.shape[0]

        # Select samples that pass the confidence threshold
        input = torch.masked_select(
            input, mask.view(b, 1)).view((-1, input.shape[1]))
        target = torch.masked_select(target, mask)

        log_prob = -F.log_softmax(input, dim=1)
        if weight is not None:
            # Weighted KL divergence
            log_prob = log_prob * weight.view((1, -1))
        loss = torch.gather(log_prob, 1, target.view((-1, 1))).view(-1)

        if mean_outside_mask:
            # Normalize by a constant (batch size)
            return loss.sum(dim=0) / b
        else:
            if weight is not None:
                # Take care of weighted sum
                weight_sum = weight[target].sum(dim=0)
                return (loss / weight_sum).sum(dim=0)
            else:
                return loss.mean(dim=0)

    def forward(self, head_id, anchors_weak, anchors_strong):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors

        weak_anchors_prob = F.softmax(anchors_weak, dim=1)

        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        if self.use_count_ema:
            with torch.no_grad():
                head_count_ema = self.count_ema[head_id]

                # Normalized and adjusted with data_len
                count_in_batch = torch.bincount(
                    target_masked, minlength=c) / n * self.data_len
                head_count_ema = head_count_ema * self.momentum + \
                    count_in_batch * (1 - self.momentum)
                self.count_ema[head_id] = head_count_ema

        if head_id == 0:
            self.num_counts += 1

        # Class balancing weights
        # This is also used for debug purpose

        # reweight_renorm is equivalent to reweight when mean_outside_mask is False
        if self.reweight_renorm:
            idx, counts = torch.unique(target_masked, return_counts=True)
            # if self.use_count_ema:
            #     print("WARNING: count EMA used with class balancing")
            freq = float(n) / len(idx) / counts.float()
            weight = torch.ones(c).cuda()
            weight[idx] = freq
        elif self.reweight:
            idx, counts = torch.unique(target_masked, return_counts=True)
            if self.use_count_ema:
                print("WARNING: count EMA used with class balancing")
            freq = 1/(counts.float()/n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq
        else:
            weight = None

        # Loss

        loss = self.kl_div_loss(input=anchors_strong, target=target, mask=mask,
                                weight=weight, mean_outside_mask=self.mean_outside_mask)

        if head_id == 0 and self.num_counts % 200 == 1:
            with torch.no_grad():
                idx, counts = torch.unique(target_masked, return_counts=True)
            if self.use_count_ema:
                print("use_count_ema max: {:.3f}, min: {:.3f}, median: {:.3f}, mean: {:.3f}".format(head_count_ema.max().item(),
                                                                                                      head_count_ema.min().item(), torch.median(head_count_ema).item(), head_count_ema.mean().item()))
            print("weak_anchors_prob, mean across batch (from weak anchor of global loss): {}".format(
                weak_anchors_prob.detach().mean(dim=0)))
            print("Mask: {} / {} ({:.2f}%)".format(mask.sum(),
                                                     mask.shape[0], mask.sum() * 100. / mask.shape[0]))
            print("idx: {}, counts: {}".format(idx, counts))

            if True:  # Verbose: print max confidence of each class
                m = torch.zeros((self.num_classes,))
                for i in range(self.num_classes):
                    v = max_prob[target == i]
                    if len(v):
                        m[i] = v.max()

                print("Max of each cluster: {}".format(m))

        return loss


###### Training the USL Model:-----------------         









def usl_t_pretrain_with_early_stopping(embeddings, device,learning_rate,batch_size,n_clusters,num_epochs_cluster,num_heads):
    # Convert embeddings to PyTorch tensors
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)

    # Create a TensorDataset and DataLoader without labels
    dataset = TensorDataset(embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize your ClusteringModel
    model = ClusteringModel(nclusters=n_clusters, embedding_dim=embeddings_tensor.size(1), nheads=num_heads).to(device)

    # Initialize the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Define your local and global loss functions
    criterion_local = OursLossLocal(num_classes=n_clusters, num_heads=num_heads, momentum=0.1, adjustment_weight=0.1, sharpen_temperature=0.5).to(device)
    criterion_global = OursLossGlobal(threshold=0.8, reweight=True, num_classes=n_clusters, num_heads=num_heads, mean_outside_mask=False, use_count_ema=False, momentum=0.1, data_len=len(dataset)).to(device)

    # Training loop
    for epoch in range(num_epochs_cluster):
        model.train()
        total_loss, total_local_loss, total_global_loss = 0.0, 0.0, 0.0

        for embeddings_batch in dataloader:
            embeddings_batch = embeddings_batch[0].to(device)

            # Forward pass
            outputs = model(embeddings_batch)

            local_loss_sum = torch.tensor(0.0).to(device)
            global_loss_sum = torch.tensor(0.0).to(device)
            for head_id, output in enumerate(outputs):
                # Calculate local loss
                local_loss = criterion_local(head_id=head_id, anchors=output, neighbors=output)
                local_loss_sum += local_loss

                # Calculate global loss
                global_loss = criterion_global(head_id=head_id, anchors_weak=output, anchors_strong=output)
                global_loss_sum += global_loss

            # Combine losses
            loss = local_loss_sum + global_loss_sum

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_local_loss += local_loss_sum.item()
            total_global_loss += global_loss_sum.item()
        
    return model


def usl_t_selective_labels(model, embeddings, device,n_clusters):
    model.eval()  # Set the model to evaluation mode

    # Assuming 'embeddings' is a NumPy array of shape (num_samples, embedding_dim)
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float).to(device)  # Ensure tensor is on the correct device
    dataset = TensorDataset(embeddings_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)  # Adjust batch size as needed

    def get_sample_selection_indices(dataloader, model, final_sample_num=100):
        all_probs_list = []
        for batch in dataloader:
            embeddings_batch = batch[0].to(device)  # Ensure batch is on the correct device
            outputs = model(embeddings_batch)
            # Compute softmax probabilities for each head
            probs = [torch.softmax(output, dim=1) for output in outputs]
            all_probs_list.append(probs)

        # Concatenate probabilities across all batches
        all_probs_list = list(zip(*all_probs_list))  # Re-arrange to group by heads
        all_probs_list = [torch.cat(probs, dim=0) for probs in all_probs_list]  # Concatenate across batches

        # Average probabilities across heads
        avg_probs = torch.stack(all_probs_list).mean(dim=0)

        # Select samples based on averaged probabilities
        _, selected_indices = torch.topk(avg_probs.max(dim=1).values, final_sample_num)

        return selected_indices.cpu().numpy()

    selected_indices = get_sample_selection_indices(dataloader, model, final_sample_num=n_clusters)
    return selected_indices

# Make sure to use `device` consistently across all your code where tensors or models are involved.





###### Training the SSL Model:-----------------

def train(embeddings,learning_rate,batch_size,n_clusters,num_epochs_cluster,num_heads):
    # Set random seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        device = 'cuda'
    else:
        device = 'cpu'
        
    print("Device: ", device)
    print("Training the USL-t SSL model...:  ") 

    
    print("Recalculating indices...")
    model=usl_t_pretrain_with_early_stopping(embeddings, device,learning_rate,batch_size,n_clusters,num_epochs_cluster,num_heads)
    selected_indices = usl_t_selective_labels(model,embeddings,device,n_clusters)
    return selected_indices
   
   
  