#custom class for encoder embedding
class Enc_Embedding(nn.Module):
    def __init__(self, vocab, hidden_size=256):
      super(Enc_Embedding, self).__init__()
      self.embedding_matrix = torch.rand(vocab, hidden_size)

    def forward(self, onehot_input):
      emb_vector = torch.matmul(onehot_input, self.embedding_matrix) 
      return emb_vector

#encoder class
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = Enc_Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

      
    #create new model
    def new(self):
      enc_new = EncoderRNN(input_lang.n_words, self.hidden_size).to('cpu')
      enc_new.load_state_dict(self.state_dict())
      return enc_new



#decoder class
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


    #create new decoder
    def new(self):
      dec_new = DecoderRNN(self.hidden_size, output_lang.n_words).to('cpu')
      dec_new.load_state_dict(self.state_dict())
      return dec_new


    