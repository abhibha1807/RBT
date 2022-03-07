from Enc_Dec import *
from dataset import *

class Model1(nn.Module):
  def __init__(self, input_size, output_size, criterion, enc_hidden_size=256, dec_hidden_size=256):
    super(Model1, self).__init__()
    self.enc = EncoderRNN(input_size, enc_hidden_size)
    self.dec = DecoderRNN(dec_hidden_size, output_size)
    self.criterion = criterion

  def enc_forward(self, input):
    print('forward pass through encoder')
    encoder_hidden = self.enc.initHidden()
    input_length = input.size(0)
    encoder_outputs = torch.zeros(input_length, self.enc.hidden_size, device='cpu')# how to pass max_length
    
    for ei in range(input_length):
      encoder_output, encoder_hidden = self.enc(
          input[ei], encoder_hidden)
      encoder_outputs[ei] = encoder_output[0, 0]
    
    return encoder_hidden, encoder_outputs

  
  def dec_forward(self, target, encoder_hidden):
    print('forward pass through decoder')
    target_length = target.size(0)
    decoder_input = torch.tensor([[SOS_token]], device='cpu') #where to put SOS_token
    decoder_hidden = encoder_hidden
    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden = self.dec(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += self.criterion(decoder_output, target[di])
        if decoder_input.item() == EOS_token:
            break
    return loss

  def new(self, vocab):
    new = Model1(vocab, vocab, self.criterion).to('cpu')
    new.load_state_dict(self.state_dict())
    return new

  def generate(self, input, tokenizer, vocab):
    print('generating')
    onehot_input = torch.zeros(input.size(0), vocab)
    index_tensor = torch.squeeze(input, dim=-1)
    onehot_input.scatter_(1, index_tensor, 1.)
    input_train = onehot_input
    enc_hidden, enc_outputs = self.enc_forward(input_train)
    decoder_input = torch.tensor([[SOS_token]], device='cpu') #where to put SOS_token
    decoder_hidden = enc_hidden
    loss = 0
    outputs = []
    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden = self.dec(
            decoder_input, decoder_hidden)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        index = torch.argmax(decoder_output)
        outputs.append(int(index))
        if decoder_input.item() == EOS_token:
            break
    decoded_sentence = tokenizer.decode(outputs)
    return decoded_sentence


  