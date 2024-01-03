from config import *


def device_as(t1, t2):
   """
   Moves t1 to the device of t2
   """
   return t1.to(t2.device)


class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called NTXentLoss as in SimCLR paper
   """
   def __init__(self, batch_size, running_batch_size, temperature):
       super().__init__()
       self.batch_size = batch_size
       self.running_batch_size = running_batch_size
       self.temperature = temperature
       # self.mask = (~T.eye(running_batch_size * 2, running_batch_size * 2, dtype=bool)).float()

   def calc_similarity_batch(self, a, b):
       representations = T.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, z_i, z_j):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       # batch_size = proj_1.shape[0]
       # z_i = F.normalize(proj_1, p=2, dim=1)
       # z_j = F.normalize(proj_2, p=2, dim=1)

       neg_size = self.running_batch_size - self.batch_size
       real_batch_size = z_i.size(0) - neg_size
       # print(f"self.batch_size: {self.batch_size}; neg_size: {neg_size}; real_batch_size: {real_batch_size}")

       similarity_matrix = self.calc_similarity_batch(z_i, z_j)

       # sim_ij = T.diag(similarity_matrix[:(2*self.batch_size), :(2*self.batch_size)], self.batch_size)
       # sim_ji = T.diag(similarity_matrix[:(2*self.batch_size), :(2*self.batch_size)], -self.batch_size)
       # sim_zeros = T.zeros(self.running_batch_size - self.batch_size)
       sim_ij = T.diag(similarity_matrix[:(2*real_batch_size), :(2*real_batch_size)], real_batch_size)
       sim_ji = T.diag(similarity_matrix[:(2*real_batch_size), :(2*real_batch_size)], -real_batch_size)
       sim_zeros = T.zeros(neg_size)
       sim_zeros = device_as(sim_zeros, sim_ij)
       sim_ij = T.cat((sim_ij, sim_zeros), dim=0)
       sim_ji = T.cat((sim_ji, sim_zeros), dim=0)
       real_running_batch_size = real_batch_size + neg_size
       # print(f"sim_ij size: {sim_ij.size()}; sim_ji size: {sim_ji.size()}; real_running_batch_size: {real_running_batch_size}")
       mask = (~T.eye(real_running_batch_size * 2, real_running_batch_size * 2, dtype=bool)).float()

       positives = T.cat((sim_ij, sim_ji), dim=0)

       nominator = T.exp(positives / self.temperature)

       denominator = device_as(mask, similarity_matrix) * T.exp(similarity_matrix / self.temperature)

       all_losses = -T.log(nominator / T.sum(denominator, dim=1))
       # loss = T.sum(all_losses) / (2 * self.running_batch_size)
       loss = T.sum(all_losses) / (2 * real_running_batch_size)
       
       return loss


class BYOLLoss(nn.Module):
    def __init__(self):
        super(BYOLLoss, self).__init__()
    
    def _loss_fn(self, online, target):
        x = F.normalize(online, dim=-1, p=2)
        y = F.normalize(target, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)

    
    def forward(self, online_one, online_two, target_one, target_two):
        loss_one = self._loss_fn(online_one, target_two.detach())
        loss_two = self._loss_fn(online_two, target_one.detach())

        loss = loss_one + loss_two
        return loss.mean()
