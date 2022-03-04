def _concat(xs):
    return torch.cat([x.view(-1) for x in xs if x is not None])


# class GreedyDecoder(nn.Module):
#    def __init__(self, hidden_size, output_size):
#         super(GreedyDecoder, self).__init__()
#         self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.LogSoftmax(dim=1)


class Architect(object):
  def __init__(self, model1, model1_mom, model1_wd, A, A_lr, A_wd, device, model2, model2_wd, model2_mom):
    self.model1 = model1
    self.model1_mom = model1_mom
    self.model1_wd = model1_wd
    self.A = A
    self.A_optim =  torch.optim.Adam(self.A.parameters(),
          lr=A_lr, betas=(0.5, 0.999), weight_decay=A_wd)
    self.device = device
    self.model2 = model2
    self.model2_wd = model2_wd
    self.model2_mom = model2_mom
  

  def loss1(self,train_inputs, idxs, criterion):
    A_idx = self.A(idxs)
    #print('A_idx:', A_idx)
    batch_loss = 0
    for i in range(batch_size):
      input_train = train_inputs[i][0]
      onehot_input = torch.zeros(input_train.size(0), vocab)
      index_tensor = input_train
      onehot_input.scatter_(1, index_tensor, 1.)
      input_train = onehot_input
      #print(input_train.size())
      target_train = train_inputs[i][1]
      idx = A_idx[i]
      enc_hidden, enc_outputs = self.model1.enc_forward(input_train)
      loss = self.model1.dec_forward(target_train, enc_hidden) # todo: find loss for each instnce and multiply A with the loss vec.
      #print('loss and idx size:', loss.size(), idx.size())
      loss = loss * idx
      batch_loss += loss 
    
    return batch_loss

  def _compute_unrolled_enc_dec_model(self, train_inputs, model1_lr, idxs, criterion):
    batch_loss = self.loss1(train_inputs, idxs, criterion)
    #Unrolled model
    theta_enc = _concat(self.model1.enc.parameters()).data
    #print(theta_enc, len(theta_enc))
    try:
        moment = _concat(model1.enc_optim.state[v]['momentum_buffer'] for v in self.model1.enc.parameters()).mul_(self.model1_mom)
    except:
        moment = torch.zeros_like(theta_enc)
    #print(moment)
    dtheta = _concat(torch.autograd.grad(batch_loss, self.model1.enc.parameters(), retain_graph = True )).data + self.model1_wd*theta_enc
    #print(dtheta)
    # # convert to the model
    unrolled_enc = self._construct_enc_from_theta(theta_enc.sub(model1_lr, moment+dtheta))
    #print(unrolled_enc)

    theta_dec = _concat(self.model1.dec.parameters()).data
    #print(theta_dec, len(theta_dec))
    try:
        moment = _concat(model1.dec_optim.state[v]['momentum_buffer'] for v in self.model1.dec.parameters()).mul_(self.model1_mom)
    except:
        moment = torch.zeros_like(theta_dec)
    #print(moment)
    dtheta = _concat(torch.autograd.grad(batch_loss, self.model1.dec.parameters(), retain_graph = True )).data + self.model1_wd*theta_dec
    print(dtheta)
    # # convert to the model
    unrolled_dec = self._construct_dec_from_theta(theta_dec.sub(model1_lr, moment+dtheta))
    #print(unrolled_dec)

    return unrolled_enc, unrolled_dec

  
  def _construct_enc_from_theta(self, theta):

    model_dict = self.model1.enc.state_dict()

    # create the new gpt model, input_lang, hidden_size, device
    enc_new = self.model1.enc.new(vocab)

    # encoder update
    params, offset = {}, 0
    for k, v in self.model1.enc.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    enc_new.load_state_dict(model_dict)
    # print([enc_new.state_dict()])
    return enc_new

  def _construct_dec_from_theta(self, theta):
    model_dict = self.model1.dec.state_dict()

    # create the new gpt model, input_lang, hidden_size, device
    dec_new = self.model1.dec.new(vocab)

    # encoder update
    params, offset = {}, 0
    for k, v in self.model1.dec.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    dec_new.load_state_dict(model_dict)
    # print([enc_new.state_dict()])
    return dec_new

  def loss2(self, un_inputs):
    batch_loss = 0

    for i in range(batch_size):  
      input_un = un_inputs[i][0]
      decoder_input = torch.tensor([[SOS_token]], device='cpu')#where to put SOS_token
      decoder_hidden = self.model1.dec.initHidden()
      print('forward pass through decoder')
     
      
      dec_soft_idxs = []
      decoder_outputs = []
      for di in range(MAX_LENGTH):
          decoder_output, decoder_hidden = self.model1.dec(
              decoder_input, decoder_hidden)
          topv, topi = decoder_output.topk(1)
          decoder_input = topi.squeeze().detach()  # detach from history as input
          #print('decoder output:', decoder_output.size())
          dec_soft_idx, dec_idx = torch.max(decoder_output, dim = -1, keepdims = True)
          dec_soft_idxs.append(dec_soft_idx)
          print('dec soft idx size:', dec_soft_idx)
          #print(dec_soft_idxs)
          decoder_outputs.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1))

          if decoder_input.item() == EOS_token:
              break

      print(decoder_outputs) #pseudo target
      decoder_outputs = torch.stack(decoder_outputs)

      #print(decoder_outputs.size())
      
      onehot_input = torch.zeros(decoder_outputs.size(0), vocab)
      #print(onehot_input.size())
      index_tensor = decoder_outputs
      #print(index_tensor.size())
      dec_soft_idxs = (torch.stack(dec_soft_idxs))
      onehot_input = onehot_input.scatter_(1, index_tensor, 1.).float().detach() + (dec_soft_idxs).sum() - (dec_soft_idxs).sum().detach()
      print(onehot_input.size(), onehot_input[0])

      enc_hidden, enc_outputs = self.model1.enc_forward(onehot_input)
      
      pseudo_target = decoder_outputs
      pseudo_input = enc_outputs

      print('pseudo target:', pseudo_target, pseudo_target.size()) # words (10) x 1 (index)
      print('pseudo input:', pseudo_input, pseudo_input.size()) # size = words (10) x hidden_size (256)
      
     


  def _compute_unrolled_model2(self, un_inputs, unrolled_enc, unrolled_dec, idxs, model2_lr, model2_optim):
      batch_loss = self.loss2(un_inputs)


      
  def step(self, train_inputs, model1_lr, A, idxs, criterion, model2_lr, model2_optim):
      self.A_optim.zero_grad()
      unrolled_enc, unrolled_dec = self._compute_unrolled_enc_dec_model(train_inputs, model1_lr, idxs, criterion)
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      unrolled_enc.to(device)
      unrolled_enc.train()
      unrolled_dec.to(device)
      unrolled_dec.train()
      #replace unlabled dataset
      unrolled_model2 = self._compute_unrolled_model2(train_inputs, unrolled_enc, unrolled_dec, idxs, model2_lr, model2_optim)

