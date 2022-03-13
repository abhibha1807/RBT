from Enc_Dec import *
from dataset import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model2(nn.Module):
  def __init__(self, input_size, output_size, criterion, enc_hidden_size=256, dec_hidden_size=256):
    super(Model2, self).__init__()
    self.enc = EncoderRNN(input_size, enc_hidden_size)
    # self.dec = DecoderRNN(dec_hidden_size, output_size)
    self.dec = AttnDecoderRNN(dec_hidden_size, output_size)
    self.criterion = criterion

  def enc_forward(self, input):
    #print('forward pass through encoder')
    encoder_hidden = self.enc.initHidden()
    input_length = input.size(0)
    encoder_outputs = torch.zeros(input_length, self.enc.hidden_size, device=device)
    
    for ei in range(input_length):
      encoder_output, encoder_hidden = self.enc(
          input[ei], encoder_hidden)
      encoder_outputs[ei] = encoder_output[0, 0]
    
    return encoder_hidden, encoder_outputs

  
  def dec_forward(self, target, encoder_hidden, encoder_outputs):

    target_length = target.size(0)
    decoder_input = torch.tensor([[SOS_token]], device=device) 
    decoder_hidden = encoder_hidden
    loss = 0
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = self.dec(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        loss += self.criterion(decoder_output, target[di])
        if decoder_input.item() == EOS_token:
            break
    return loss


  def new(self, vocab):
    new = Model2(vocab, vocab, self.criterion).to(device)
    new.load_state_dict(self.state_dict())
    return new


  def generate(self, input, tokenizer, vocab):
    print('generating')
    onehot_input = torch.zeros(input.size(0), vocab, device='cuda')
    index_tensor = input
    onehot_input.scatter_(1, index_tensor, 1.)
    input_train = onehot_input
    enc_hidden, enc_outputs = self.enc_forward(input_train)
    decoder_input = torch.tensor([[SOS_token]], device=device) 
    decoder_hidden = enc_hidden
    loss = 0
    outputs = []
    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden, decoder_attention = self.dec(
            decoder_input, decoder_hidden, enc_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # detach from history as input
        index = torch.argmax(decoder_output)
        outputs.append(int(index))
        if decoder_input.item() == EOS_token:
            break
    # outputs = torch.stack(outputs)
    #print(outputs)
    decoded_sentence = tokenizer.decode(outputs)
    print(decoded_sentence)
    return decoded_sentence

  
