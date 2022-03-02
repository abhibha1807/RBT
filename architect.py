#implments step 1 and 2 for now.

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs if x is not None])

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
    batch_loss = 0
    for i in range(batch_size): # for each instance in batch
      input_train = train_inputs[i][0] # input indices
      onehot_input = torch.zeros(input_train.size(0), input_lang.n_words) #input to custom embedding layer of model 1's encoder
      index_tensor = input_train
      onehot_input.scatter_(1, index_tensor, 1.)
      input_train = onehot_input
      target_train = train_inputs[i][1]
      idx = A_idx[i] # ai
      enc_hidden, enc_outputs = self.model1.enc_forward(input_train)
      loss = self.model1.dec_forward(target_train, enc_hidden) # todo: find loss for each instnce and multiply A with the loss vec.
      loss = loss * idx #multiplying by ai 
      batch_loss += loss 
    
    return batch_loss

  # step 1
  def _compute_unrolled_enc_dec_model(self, train_inputs, model1_lr, idxs, criterion):
    batch_loss = self.loss1(train_inputs, idxs, criterion)
    #Unrolled enc model
    theta_enc = _concat(self.model1.enc.parameters()).data
    try:
        moment = _concat(model1.enc_optim.state[v]['momentum_buffer'] for v in self.model1.enc.parameters()).mul_(self.model1_mom)
    except:
        moment = torch.zeros_like(theta_enc)
    dtheta = _concat(torch.autograd.grad(batch_loss, self.model1.enc.parameters(), retain_graph = True )).data + self.model1_wd*theta_enc
    # # convert to the model
    unrolled_enc = self._construct_enc_from_theta(theta_enc.sub(model1_lr, moment+dtheta))

    #Unrolled dec model
    theta_dec = _concat(self.model1.dec.parameters()).data
    try:
        moment = _concat(model1.dec_optim.state[v]['momentum_buffer'] for v in self.model1.dec.parameters()).mul_(self.model1_mom)
    except:
        moment = torch.zeros_like(theta_dec)
    dtheta = _concat(torch.autograd.grad(batch_loss, self.model1.dec.parameters(), retain_graph = True )).data + self.model1_wd*theta_dec
    # # convert to the model
    unrolled_dec = self._construct_dec_from_theta(theta_dec.sub(model1_lr, moment+dtheta))

    return unrolled_enc, unrolled_dec

  
  def _construct_enc_from_theta(self, theta):

    model_dict = self.model1.enc.state_dict()

    enc_new = self.model1.enc.new()

    # encoder update
    params, offset = {}, 0
    for k, v in self.model1.enc.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    enc_new.load_state_dict(model_dict)
    return enc_new

  def _construct_dec_from_theta(self, theta):
    model_dict = self.model1.dec.state_dict()

    dec_new = self.model1.dec.new()

    # encoder update
    params, offset = {}, 0
    for k, v in self.model1.dec.named_parameters():
        v_length = np.prod(v.size())
        params[k] = theta[offset: offset+v_length].view(v.size())
        offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    dec_new.load_state_dict(model_dict)
    return dec_new

  def loss2(self, un_inputs):
    batch_loss = 0

    for i in range(batch_size):  # for each instance in a batch
      input_un = un_inputs[i][0]
      decoder_input = torch.tensor([[SOS_token]], device='cpu')
      decoder_hidden = self.model1.dec.initHidden()
      print('forward pass through decoder to gnerate pseudo targets')
     
      ip_encoder = []   
      pseudo_target = []
      for di in range(MAX_LENGTH):
          decoder_output, decoder_hidden = self.model1.dec(
              decoder_input, decoder_hidden)
          topv, topi = decoder_output.topk(1)
          decoder_input = topi.squeeze().detach()  # detach from history as input
          print('decoder output:', decoder_output.size())
          one_hot = F.gumbel_softmax(decoder_output, tau=1, hard=True) # gumbel softmax (will change this)
          ip_encoder.append(one_hot)
          pseudo_target.append(torch.unsqueeze(torch.argmax(decoder_output), dim = -1)) #converting to pseudo target indices by performing argmax on output probabilities 

          if decoder_input.item() == EOS_token:
              break

     
      #batch decode eng to french (prepare target for input to encoder to generate pseudo source)
      #input_lang = french, output_lang = english
      pseudo_input=[]
      for i in range(len(pseudo_target)):
        decoded_word = output_lang.index2word[int(torch.squeeze(pseudo_target[i], dim=0))] # get corrsponding english word using dictionary
        try:
          input_lang.addWord(decoded_word)# add eng word to french vocab
        except:
          print('word already present') 
        encoded_word = input_lang.word2index[decoded_word] #convert eng word to french language indices
        pseudo_input.append(encoded_word)

     



  def _compute_unrolled_model2(self, un_inputs, unrolled_enc, unrolled_dec, idxs, model2_lr, model2_optim):
      batch_loss = self.loss2(un_inputs)


      
  def step(self, train_inputs, un_inputs ,model1_lr, A, idxs, criterion, model2_lr, model2_optim):
      self.A_optim.zero_grad()
      #step 1
      unrolled_enc, unrolled_dec = self._compute_unrolled_enc_dec_model(train_inputs, model1_lr, idxs, criterion)
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      unrolled_enc.to(device)
      unrolled_enc.train()
      unrolled_dec.to(device)
      unrolled_dec.train()
      #step 2
      unrolled_model2 = self._compute_unrolled_model2(un_inputs, unrolled_enc, unrolled_dec, idxs, model2_lr, model2_optim)

