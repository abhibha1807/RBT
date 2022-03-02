import utils
from torch.utils.data import TensorDataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from torch.autograd import Variable
from dataset import *
from arcitect import *
from Enc_Dec import *
from models import *

# TASK: French (source) -> English (target)


#fucnctions to load sentence pairs.
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return torch.stack((input_tensor, target_tensor))


def get_train_dataset(pairs):
  attn_idx = torch.arange(len(pairs))
  tensor_pairs = [tensorsFromPair(pairs[i]) for i in range(len(pairs))]
  a = torch.stack((tensor_pairs))
  train_data = TensorDataset(a, attn_idx)
  return train_data

def get_un_dataset(pairs):
  tensor_pairs = [tensorsFromPair(pairs[i]) for i in range(len(pairs))]
  a = torch.stack((tensor_pairs))
  un_data = TensorDataset(a)
  return un_data

#load data
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))


# to do: split data
# for now taking 10 instance for train, test and validation sets.
train_num_points, valid_num_points, un_num_points = 10, 10, 10
train_data = get_train_dataset(pairs[0:10])
un_data = get_un_dataset(pairs[10:20])


batch_size = 2

# create dataloaders 
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)

# valid_dataloader = DataLoader(valid_data, sampler=RandomSampler(valid_data), 
#                       batch_size=batch_size, pin_memory=True, num_workers=0)

un_dataloader = DataLoader(un_data, sampler=RandomSampler(un_data), 
                        batch_size=batch_size, pin_memory=True, num_workers=0)



#load model
hidden_size = 256
criterion = nn.NLLLoss()
model1_lr = 0.1 #dummy value
model2_lr = 0.1 
model1 = Model1(input_lang.n_words, output_lang.n_words) #teacher
model2 = Model2(input_lang.n_words, output_lang.n_words) #student
model1optim = optim.SGD(model1.parameters(), lr=model1_lr)
model2_optim = optim.SGD(model2.parameters(), lr=model2_lr)

A = attention_params(train_num_points) #initliase A


model1_mom, model1_wd, A_wd, A_lr, model2_wd, model2_mom = 0.1,0.1, 0.1,0.1, 0.1,0.1 # dummy values

architect = Architect(model1, model1_mom, model1_wd, A, A_lr, A_wd, device, model2, model2_wd, model2_mom)


def train(epoch, train_dataloader):
 

  for step, batch in enumerate(train_dataloader):
    model1.train()
    
    train_inputs = batch[0] #train inputs.
    idxs = batch[1] #A
   
    architect.step(train_inputs, un_inputs, model1_lr, A, idxs, criterion, model2_lr, model2_optim)

  
    
epoch = 0

train(epoch, train_dataloader)   