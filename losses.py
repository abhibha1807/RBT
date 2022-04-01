import torch
from dataset import *
import gc
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss1(inputs, model, idxs, A, batch_size, vocab):
    A_idx = A(idxs)
    batch_loss = 0
    for i in range(inputs.size(0)):
      
        input_train = inputs[i][0]
        target_train = inputs[i][1]
        idx = A_idx[i]
        enc_hidden, enc_outputs = model.enc_forward(input_train)
        loss = model.dec_forward(target_train, enc_hidden, enc_outputs) 
        loss = loss * idx
        batch_loss += loss 
    return batch_loss/inputs.size(0)
      
    

def loss2(un_inputs, model1, model2, batch_size, vocab):
    print('in loss 2')
    batch_loss = 0
    
    #generate pseudo target by passing through decoder
    for i in range(batch_size):  
      
            input_un = un_inputs[i][0]
            enc_hidden, enc_outputs = model1.enc_forward(input_un)

            decoder_input = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
            decoder_hidden = enc_hidden
            
            dec_soft_idxs = []
            decoder_outputs = []
            for di in range(MAX_LENGTH):
                embedded = model1.embedding_dec(decoder_input).view(1, 1, -1)
                embedded = embedded/math.sqrt(256)
                decoder_output, decoder_hidden = model1.dec(
                    embedded, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                
                dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
                dec_soft_idxs.append(dec_soft_idx) #save differentiable outputs
                decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))
                if decoder_input.item() == EOS_token:
                    break

            #print(decoder_outputs) #pseudo target
            print('before dec_soft_idxs:', dec_soft_idxs)# every tensor has grad fun assopciated with it.
            decoder_outputs = torch.stack(decoder_outputs)#differentiable,no break in computation graph

            print('decoder_outputs:',decoder_outputs, decoder_outputs.grad_fn)

            # gumbel softmax (prepare target for generating pseudo input using encoder)
            onehot_input_encoder1 = torch.zeros(decoder_outputs.size(0), vocab, device = device)
            index_tensor = decoder_outputs
            dec_soft_idxs = (torch.stack(dec_soft_idxs))
            onehot_input_encoder1 = onehot_input_encoder1.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()

            enc_hidden_, enc_outputs_ = model1.enc_forward(onehot_input_encoder1)
      
            
            pseudo_target = decoder_outputs

            # print('pseudo target:', pseudo_target, pseudo_target.size())
            # greedy decoding -> similar to model.generate() (hugging face)
            decoder_input = torch.tensor([[SOS_token]], device=device)#where to put SOS_token
            decoder_hidden = enc_hidden_

            dec_soft_idxs = []
            decoder_outputs = []
            for di in range(MAX_LENGTH):
                embedded = model1.embedding_dec(decoder_input).view(1, 1, -1)
                embedded = embedded/math.sqrt(256)
                decoder_output, decoder_hidden = model1.dec(
                    embedded, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
                dec_soft_idxs.append(dec_soft_idx)
                decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))

                if decoder_input.item() == EOS_token:
                    break
            
            # gumbel softmax 
            input_to_model2 = torch.stack(decoder_outputs)
            onehot_input_model2 = torch.zeros(input_to_model2.size(0), vocab, device = device)
            index_tensor = input_to_model2
            dec_soft_idxs = (torch.stack(dec_soft_idxs))
            onehot_input_model2 = onehot_input_model2.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()

            pseudo_input = onehot_input_model2 

            #model2 forward pass
            enc_hidden, enc_outputs = model2.enc_forward(pseudo_input)
            loss = model2.dec_forward(pseudo_target, enc_hidden, enc_outputs) # todo: find loss for each instnce and multiply A with the loss vec.
            batch_loss += loss 

            del onehot_input_encoder1 , onehot_input_model2
                
            gc.collect()   

       
    return batch_loss/batch_size

