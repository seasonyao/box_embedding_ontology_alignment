#!/bin/env python

from boxes import *
from learner import *
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

#read alignment data and aml_score
with open('/mnt/nfs/work1/llcao/zonghaiyao/ontology_alignment/box_for_ontology_alignment/data/ontologies/anatomy/alignment_data.pkl', 'rb') as f:
    all_alignment_data = pickle.load(f)

hm_align_score = pd.read_csv('/mnt/nfs/work1/llcao/zonghaiyao/ontology_alignment/box_for_ontology_alignment/data/ontologies/anatomy/human_mouse_alignment_score.tsv', index_col=0)

#--------------------------------------------------------------------
#override from boxes/callback

#Rewrite RandomNegativeSampling, reshape the dimension of the data to facilitate the calculation of max margin loss. Modified the split_data function to distinguish data as mouse/human/align, so that only align can be operated accordingly

#Assuming batch size=2, sample ratio=2, 
#the input is [pos1, pos2]
#original output is [pos1, pos2, neg1, neg2, neg3, neg4]
#this output is [[pos1, neg1, neg2], [pos2, neg3, neg4]]
#y will also change from [1,1,0,0,0,0] to [[1,0,0],[1,0,0]]
@dataclass
class RandomNegativeSampling(Callback):
    """
    Given a pair (u,v) generates random pairs of the form (u,x) or (y,v)
    """
    num_entities: int
    ratio: int
    
    def batch_begin(self, l: Learner):
        with torch.no_grad():
#             print("batch_in before rns:", l.batch_in)
#             print("batch_out before rns:", l.batch_out)

            batch_in = l.batch_in.to("cpu") # Should be a tensor of indices, shape (batch_size, k)            
            batch_size, k = batch_in.shape
            num_neg_samples = batch_in.shape[0] * self.ratio
            negative_samples = batch_in.repeat(self.ratio, 1) # shape (batch_size * ratio, k)
            negative_probs = torch.zeros(num_neg_samples).to(l.batch_out.device)
            negative_samples.scatter_(1, torch.randint(k,(num_neg_samples,1)), torch.randint(self.num_entities, (num_neg_samples,1)))  
            negative_samples = negative_samples.to(l.batch_in.device)
            
            #for max_margin to re-construct the data_in and data_out
            negative_samples = torch.reshape(negative_samples, (batch_size, self.ratio, k))
            negative_probs = torch.reshape(negative_probs, (batch_size, self.ratio))

            l.batch_in = torch.cat((l.batch_in.unsqueeze(1), negative_samples), 1)
            l.batch_out = torch.cat((l.batch_out.unsqueeze(-1), negative_probs), -1)
            

#Based on the modified version of RandomNegativeSampling, when processing the data_out of the alignment data, 1/0 is no longer used, but look up the hm_align_score matrix and use the corresponding score to assign data_out
@dataclass
class RandomNegativeSamplingWithAMLScore(Callback):
    """
    Given a pair (u,v) generates random pairs of the form (u,x) or (y,v)
    """
    num_entities: int
    ratio: int
    
    def batch_begin(self, l: Learner):
        with torch.no_grad():
#             print("batch_in before rns:", l.batch_in)
#             print("batch_out before rns:", l.batch_out)

            batch_in = l.batch_in.to("cpu") # Should be a tensor of indices, shape (batch_size, k)            
            batch_size, k = batch_in.shape
            num_neg_samples = batch_in.shape[0] * self.ratio
            negative_samples = batch_in.repeat(self.ratio, 1) # shape (batch_size * ratio, k)
            negative_probs = torch.zeros(num_neg_samples).to(l.batch_out.device)
            negative_samples.scatter_(1, torch.randint(k,(num_neg_samples,1)), torch.randint(1, self.num_entities - 1, (num_neg_samples,1)))  
            negative_samples = negative_samples.to(l.batch_in.device)

            #check whether or not it is an alignment edge, if yes, find their AML_scores
            for i in range(batch_size):
                if self.split_data(l.batch_in[i], 2737) == 1:
                    #df[5588][773]
                    l.batch_out[i] = hm_align_score[str(int(l.batch_in[i][0]))][int(l.batch_in[i][1])]
                elif self.split_data(l.batch_in[i], 2737) == 2:
                    l.batch_out[i] = hm_align_score[str(int(l.batch_in[i][1]))][int(l.batch_in[i][0])]
                               
            for i in range(num_neg_samples):
                if self.split_data(negative_samples[i], 2737) == 1:
                    negative_probs[i] = hm_align_score[str(int(negative_samples[i][0]))][int(negative_samples[i][1])]
                    
                elif self.split_data(negative_samples[i], 2737) == 2:
                    negative_probs[i] = hm_align_score[str(int(negative_samples[i][1]))][int(negative_samples[i][0])]


            #for max_margin to re-construct the data_in and data_out
            negative_samples = torch.reshape(negative_samples, (batch_size, self.ratio, k))
            negative_probs = torch.reshape(negative_probs, (batch_size, self.ratio))

            l.batch_in = torch.cat((l.batch_in.unsqueeze(1), negative_samples), 1)
            l.batch_out = torch.cat((l.batch_out.unsqueeze(-1), negative_probs), -1)
            
    def split_data(self, data, split):
        a, b = data > split
        
        #check if it is an alignment data
        if not a and not b:
            return 0
        elif a and b:
            return 0
        elif a:
            return 1 #human first
        else:
            return 2 #mouse first
        
#------------------------------------------------------------------------
#override from boxes/loss_functions

#now the data_in and data_out is [[pos1, neg1, neg2], [pos2, neg3, neg4]] and [[1,0,0],[1,0,0]], so it's easy to use following functions to cal their max_margin_loss
def human_cond_margin_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    if type(model_out) == dict:
        return kl_div_sym(model_out["P(A|B)"], target, eps).mean()
    
    model_out_pos, model_out_neg = model_out
    target_pos = target[0]
    target_neg = target[1]
        
    margin_loss = margin_loss_term_pos(model_out_pos["P(A|B)"], target_pos).mean()
    margin_loss += margin_loss_term_neg(model_out_neg["P(A|B)"], target_neg).mean()
      
    return margin_loss

def mouse_cond_margin_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    if type(model_out) == dict:
        return kl_div_sym(model_out["P(A|B)"], target, eps).mean()

    model_out_pos, model_out_neg = model_out
    target_pos = target[0]
    target_neg = target[1]
    
    
    margin_loss = margin_loss_term_pos(model_out_pos["P(A|B)"], target_pos).mean()
    margin_loss += margin_loss_term_neg(model_out_neg["P(A|B)"], target_neg).mean()
       
    return margin_loss

def align_cond_margin_loss(model_out: ModelOutput, target: Tensor, eps: float = torch.finfo(torch.float32).tiny) -> Tensor:
    if type(model_out) == dict:
        return kl_div_sym(model_out["P(A|B)"], target, eps).mean()
    model_out_pos, model_out_neg = model_out
    target_pos = target[0]
    target_neg = target[1]
        
    margin_loss = margin_loss_term_pos(model_out_pos["P(A|B)"], target_pos).mean()
    margin_loss += margin_loss_term_neg(model_out_neg["P(A|B)"], target_neg).mean()
       
    return margin_loss


# max(gamma_1 - vol(i \int j) , 0) + sum over all negative (max(vol(i \int j) - gamma_2, 0))

def margin_loss_term_pos(p_pos: Tensor, q_pos: Tensor, margin = 0.8) -> Tensor:
    return (margin - p_pos).clamp_min(torch.finfo(torch.float32).tiny)
    
def margin_loss_term_neg(p_neg: Tensor, q_neg: Tensor, margin = 0.2) -> Tensor:
    return (p_neg - margin).clamp_min(torch.finfo(torch.float32).tiny)



# See boxes/loss_functions.py file for more options. Note that you may have to changed them to fit your use case.
# Also note that "kl_div_sym" is just binary cross-entropy.



#------------------------------------------------------------------------
#override from learner/loss_functions

#The main change is also to adjust the errors in various dimensions for max_margin, and other parts are the same
class LossPieces(Module):

    def __init__(self, *loss_funcs: FuncInput):
        """

        :param functions: List of functions or tuples (weight, function)
        :type functions: Collection[Union[Callable, Tuple[float, Callable]]]
        """
        super().__init__()
        self.unweighted_funcs = unweighted_func_dict(*loss_funcs)
        self.loss_funcs = func_list_to_dict(*loss_funcs)

    def forward(self, model_out, true_out: Tensor,
                learner: Optional[Learner] = None,
                recorder: Optional[Recorder] = None,
                weighted: bool = True,
                categories: bool = False,
                use_wandb: bool = False,
                **kwargs) -> Tensor:
        """
        Weighted sum of all loss functions. Tracks values in Recorder.
        """
        if weighted:
            loss_funcs = self.loss_funcs
        else:
            loss_funcs = self.unweighted_funcs
        
        grad_status = torch.is_grad_enabled()
        if learner is None:
            torch.set_grad_enabled(False)
        
        # change the loss function for margin loss
        try:
            if categories:
                loss_pieces = {}
                for k,l in loss_funcs.items():
                    if type(model_out) == tuple:
                        if 'mouse' in k:
                            loss_pieces['mouse_cond_kl_loss'] = l((model_out[0][0], model_out[1][0]), (true_out[0][0], true_out[1][0]))
                        elif 'human' in k:
                            loss_pieces['human_cond_kl_loss'] = l((model_out[0][1], model_out[1][1]), (true_out[0][1], true_out[1][1]))
                        else:
                            loss_pieces['align_cond_kl_loss'] = l((model_out[0][2], model_out[1][2]), (true_out[0][2], true_out[1][2]))
                    else:
                        if 'mouse' in k:
                            loss_pieces['mouse_cond_kl_loss'] = l(model_out[0], true_out[0])
                        elif 'human' in k:
                            loss_pieces['human_cond_kl_loss'] = l(model_out[1], true_out[1])
                        else:
                            loss_pieces['align_cond_kl_loss'] = l(model_out[2], true_out[2])
                if use_wandb:
                    # Log metrics inside your training loop
                    metrics = {'mouse_loss': torch.sum(loss_pieces['mouse_cond_kl_loss']), 
                               'human_loss': torch.sum(loss_pieces['human_cond_kl_loss']), 
                               'alignmnt_loss': torch.sum(loss_pieces['align_cond_kl_loss'])}
                    wandb.log(metrics)
                
                loss = 0
                for key,val in loss_pieces.items():
                    if not isnan(val):
                        loss += val
                        

            else:
                loss_pieces = {k: l(model_out, true_out) for k, l in loss_funcs.items()}
                loss = sum(loss_pieces.values())
          

            loss_pieces['loss'] = loss
            if learner is not None:
                if recorder is not None:
                    recorder.update_(loss_pieces, learner.progress.partial_epoch_progress())
                else:
                    self.recorder.update_(loss_pieces, learner.progress.partial_epoch_progress())
        finally:
            torch.set_grad_enabled(grad_status)
        return loss
    
    
#------------------------------------------------------------------------
#override from boxes/metrics

#There may be some problems in implementation and need to be re-implemented
def mean_reciprocal_rank(model, data_in, data_out) -> Tensor: #(maybe wrong, need re-implement)
    ratio = 5
    
    batch_in = data_in.to("cpu") # Should be a tensor of indices, shape (batch_size, k) 
    batch_size, k = batch_in.shape
    num_neg_samples = batch_in.shape[0] * ratio
    negative_samples = batch_in.repeat(ratio, 1) # shape (batch_size * ratio, k)
    negative_probs = torch.zeros(num_neg_samples).to(data_out.device)
    negative_samples.scatter_(1, torch.randint(k,(num_neg_samples,1)), torch.randint(num_boxes, (num_neg_samples,1)))
    negative_samples = negative_samples.to(batch_in.device)
    positive_data = batch_in
    negative_data = negative_samples 
    
    
    positive_score = torch.reshape(model(positive_data)["P(A|B)"], (batch_size, 1))
    negative_score = torch.reshape(model(negative_data)["P(A|B)"], (batch_size, ratio))
    
    
    scores = torch.cat((positive_score, negative_score), dim=-1)
    
    _, loss_idx = scores.sort(dim=1, descending=True)
    _, idx_rank = loss_idx.sort(dim=1)
    MRR = idx_rank[:, 0].sum()/float(batch_size)
    print(MRR)
    
    return MRR
    
    
#------------------------------------------------------------------------
#override from learner/learner

#The main change is also to adjust the errors in various dimensions for max_margin. Also modified the split_data function to distinguish data as mouse/human/align, so that only align can be operated accordingly. And other parts are the same.
@dataclass
class Learner:
    train_dl: DataLoader
    model: Module
    loss_fn: Callable
    opt: optim.Optimizer
    callbacks: CallbackCollection = field(default_factory=CallbackCollection)
    recorder: Recorder = field(default_factory=Recorder)
    categories: bool = False
    use_wandb: bool = False
    reraise_keyboard_interrupt: bool = False
    reraise_stop_training_exceptions: bool = False

    def __post_init__(self):
        self.progress = Progress(0,0,len(self.train_dl))
        self.callbacks.learner_post_init(self)

    #the split parameter will be used to find human/mouse/align data, so you need change it when using diff dataset(index)
    def split_data(self, batch_in, batch_out, split):
        category = torch.zeros(size=(batch_in.shape[0],), dtype=int)
        batch_class = batch_in > split
        need_pos_neg = 0
        
        if len(list(batch_class.shape))==3:
            need_pos_neg = 1
            
        if need_pos_neg:
            batch_class = batch_class[:, 0]
        
        for i, (a,b) in enumerate(batch_class):
            if not a and not b:
                category[i] = 0
            elif a and b:
                category[i] = 1
            else:
                category[i] = 2

        self.mouse_in = batch_in[category == 0]
        self.human_in = batch_in[category == 1]
        self.align_in = batch_in[category == 2]

        self.mouse_out = batch_out[category == 0]
        self.human_out = batch_out[category == 1]
        self.align_out = batch_out[category == 2]

        if need_pos_neg:
            # INPUT TO THE MODEL:
            data_in_positive = (self.mouse_in[:, 0], self.human_in[:, 0], self.align_in[:, 0])
            # TARGET/LABEL:
            data_out_positive = (self.mouse_out[:, 0], self.human_out[:, 0], self.align_out[:, 0])
            # INPUT TO THE MODEL:
            data_in_negative = (self.mouse_in[:, 1], self.human_in[:, 1], self.align_in[:, 1])
            # TARGET/LABEL:
            data_out_negative = (self.mouse_out[:, 1], self.human_out[:, 1], self.align_out[:, 1])

            data_in = (data_in_positive, data_in_negative)
            data_out = (data_out_positive, data_out_negative)
        else:
            # INPUT TO THE MODEL:
            data_in = (self.mouse_in, self.human_in, self.align_in)
            # TARGET/LABEL:
            data_out = (self.mouse_out, self.human_out, self.align_out)

        return data_in, data_out

    def TensorNaN(self, size:Union[None,List[int], Tuple[int]]=None, device=None, requires_grad:bool=True):
        if size is None:    
            return torch.tensor(float('nan'), device=device, requires_grad=requires_grad)
        else:
            return float('nan') * torch.zeros(size=size, device=device, requires_grad=requires_grad)


    def train(self, epochs, progress_bar = True):
        try:
            self.callbacks.train_begin(self)
            for epoch in trange(epochs, desc="Overall Training:", disable=not progress_bar):
                self.callbacks.epoch_begin(self)
                for iteration, batch in enumerate(tqdm(self.train_dl, desc="Current Batch:", leave=False, disable=not progress_bar)):
                    if len(batch) == 2: # KLUDGE
                        self.batch_in, self.batch_out = batch
                    else:
                        self.batch_in = batch[0]
                        self.batch_out = None
                    self.progress.increment()
                    self.callbacks.batch_begin(self)
                    self.opt.zero_grad()
                    
                    if self.categories:
                        self.data_in, self.data_out = self.split_data(self.batch_in, self.batch_out, split=2737)
                        #print("fffffffffffffffff")
                        #print(self.data_in[0][0][0], self.data_in[1][0][0], self.data_out[0][0][0], self.data_out[1][0][0])
                        #2737 is max mouse index
                        self.model_pred_pos = [self.model(item) if len(item)>0 else {'P(A|B)':self.TensorNaN(device=self.batch_in.device)} for item in self.data_in[0]]
                        self.model_pred_neg = [self.model(item) if len(item)>0 else {'P(A|B)':self.TensorNaN(device=self.batch_in.device)} for item in self.data_in[1]]
                        
                        self.model_pred = (self.model_pred_pos, self.model_pred_neg)
                        
                        self.loss = self.loss_fn(self.model_pred, self.data_out, self, self.recorder, categories=True, use_wandb=self.use_wandb)                        
                    else:
                        self.model_out = self.model(self.batch_in)
                        if self.batch_out is None:
                            self.loss = self.loss_fn(self.model_out, self, self.recorder, categories=True, use_wandb=self.use_wandb)
                        else:
                            self.loss = self.loss_fn(self.model_out, self.batch_out, self, self.recorder, categories=True, use_wandb=self.use_wandb)
                        
                    # Log metrics inside your training loop
                    if self.use_wandb:
                        metrics = {'epoch': epoch, 'loss': self.loss}
                        wandb.log(metrics)

                    # print(self.recorder.dataframe)
                    self.loss.backward()
                    self.callbacks.backward_end(self)
                    self.opt.step()
                    self.callbacks.batch_end(self)
                # print(self.recorder.dataframe)
                
                # run evaluating at the end of every epoch
                self.evaluation([0.5])
                
                
                self.callbacks.epoch_end(self)
        except StopTrainingException as e:
            print(e)
            if self.reraise_stop_training_exceptions:
                raise e
        except KeyboardInterrupt:
            print(f"Stopped training at {self.progress.partial_epoch_progress()} epochs due to keyboard interrupt.")
            if self.reraise_keyboard_interrupt:
                raise KeyboardInterrupt
        finally:
            self.callbacks.train_end(self)


    def evaluation(self, trials, progress_bar=True):
        with torch.no_grad():
            # self.callbacks.eval_begin(self)
            for t in trials:
                self.callbacks.eval_align(self, t)
            self.callbacks.metric_plots(self)
            self.callbacks.bias_metric(self)
            self.callbacks.eval_end(self)
            
            
#still implementing now
# @dataclass
# class TrueNegativeSampling(Callback):
#     """
#     Given a pair (u,v) generates random pairs of the form (u,x) or (y,v)
#     """
#     num_entities: int
#     ratio: int
    
#     def batch_begin(self, l: Learner):
#         with torch.no_grad():
# #             print("batch_in before rns:", l.batch_in)
# #             print("batch_out before rns:", l.batch_out)

#             batch_in = l.batch_in.to("cpu") # Should be a tensor of indices, shape (batch_size, k)            
#             batch_size, k = batch_in.shape
#             num_neg_samples = batch_in.shape[0] * self.ratio
#             negative_samples = batch_in.repeat(self.ratio, 1) # shape (batch_size * ratio, k)
#             negative_probs = torch.zeros(num_neg_samples).to(l.batch_out.device)
#             #negative_samples.scatter_(1, torch.randint(k,(num_neg_samples,1)), torch.randint(self.num_entities, (num_neg_samples,1)))  
#             for i in range(batch_size):
#                 if self.split_data(l.batch_in[i], 2737) == 0:#all mouse
                    
#                 if self.split_data(l.batch_in[i], 2737) == 1:#all human
                    
#                 if self.split_data(l.batch_in[i], 2737) == 2:#human first
                    
#                 else:                                        #mouse first
                    
            
#             generate_true_neg_alignments(all_alignment_data, )
            
#             negative_samples = negative_samples.to(l.batch_in.device)
            
#             #for max_margin to re-construct the data_in and data_out
#             negative_samples = torch.reshape(negative_samples, (batch_size, self.ratio, k))
#             negative_probs = torch.reshape(negative_probs, (batch_size, self.ratio))

#             l.batch_in = torch.cat((l.batch_in.unsqueeze(1), negative_samples), 1)
#             l.batch_out = torch.cat((l.batch_out.unsqueeze(-1), negative_probs), -1)

            
#     def get_siblings(self, parents:dict, children:dict, node:int):
#         siblings = []

#         # There should only be only one node that doesn't have any parents, the root node
#         if node in parents:
#             parents_of_node = parents[node]

#             # Cycle through all possible parents of the given node
#             for p in parents_of_node:

#                 # if the parent node has any children, add them to the siblings list
#                 if p in children:
#                     siblings = siblings + children[p]

#                     # remove the node from the siblings list
#                     siblings.remove(node)

#             # if there are any siblings, return the list of them
#             if siblings:
#                 return siblings

#             # if there are no siblings, return -1
#             else:
#                 print("Given node does not have any siblings:", node)
#                 return -1

#         # if the node does not have any parents, return -1
#         else:
#             print("Given node does not have any parents:", node)
#             return -1
#         # ---- 

#     def generate_true_neg_alignments(self, alignments:list, alignment_split:float=0.5, ratio:float=1.0):

#         true_negatives = []
#         numFailures = 0
#         num_samples = int(len(alignments) * alignment_split * ratio)

#         while (len(true_negatives) < num_samples) and (numFailures < 100):
#             # Select a random alignment within the list of all alignments
#             rdm_align = random.choice(alignments)

#             # Pick a node to alter within the randomly chosen alignment 
#             const_node = rdm_align[0]
#             change_node = rdm_align[1]

#             # generate all siblings within the human ontology of the chosen node
#             if change_node in h_parents:
#                 siblings = get_siblings(h_parents, h_children, change_node)

#             # generate all siblings within the mouse ontology of the chosen node
#             elif change_node in m_parents:
#                 siblings = get_siblings(m_parents, m_children, change_node)

#             # This shouldn't be triggered -- every node should have a parent node
#             # The only possible node that could trigger the below statement is the root node
#             else:
#                 print("Node not found in either Ontology or does not have any parents")


#             # This error will typically be thrown if the chosen node does not have any siblings
#             if siblings == -1:
#                 print("Error thrown when retrieving siblings")

#             else:
#                 # Choose some random siblings to be make the true negative
#                 negative_alignment = (const_node, random.choice(siblings))

#                 if negative_alignment in alignments:
#                     numFailures += 1
#                     print("Generated negative is an existing alignment:", negative_alignment, "OG random:", rdm_align, siblings)
#                     pass

#                 elif negative_alignment in true_negatives:
#                     numFailures += 1
#                     print("Generated negative already in true_negatives:", negative_alignment)
#                     pass

#                 # include this negative alignment in the true_negatives list
#                 else:
#                     true_negatives.append(negative_alignment)
#                     true_negatives.append((negative_alignment[1], const_node))
#                     numFailures = 0


#         return true_negatives
    
#     def split_data(self, data, split):
#         a, b = data > split
        
#         #check if it is an alignment data
#         if not a and not b: #all mouse
#             return 0
#         elif a and b:       #all human
#             return 1
#         elif a:
#             return 2        #human first
#         else:
#             return 3        #mouse first
            