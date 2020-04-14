from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import os
from flask import Flask, request, render_template
import torch
import torch.nn as nn
#USE_CUDA = torch.cuda.is_available()
device = torch.device("cpu")
#device = torch.device("cuda" if USE_CUDA else "cpu")
from Encoder_Decoder_data import EncoderRNN, LuongAttnDecoderRNN, Voc
from Encoder_Decoder_data import contraction_mapping, SOS_token, EOS_token,RARE_WORD



model_name = 'cb_model'
#attn_model = 'dot'
attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.2

app = Flask(__name__)

#addr = "C:/Users/Administrator/ML/Heroku Deployements/Text_Summerization_Reviews/save/cb_model/2-2_500/"
addr = "save/cb_model/2-2_500/"
checkpoint_iter = 20000
#loadFilename = os.path.join(addr,'{}_checkpoint_1.tar'.format(checkpoint_iter))

loadFilename = os.path.join('{}_checkpoint.tar'.format(checkpoint_iter))

voc = Voc()

# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on #, map_location=device
    #checkpoint = torch.load(loadFilename, map_location='cpu')
    checkpoint = torch.load(loadFilename, map_location=lambda storage, loc: storage)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


embedding = nn.Embedding(voc.num_words, hidden_size)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)
embedding.load_state_dict(embedding_sd)
encoder = encoder.to(device)
decoder = decoder.to(device)
# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

stop_words = set(stopwords.words('english')) - set(['not','no'])
def clean(text, summary = True):
    newString = text.lower()
    newString = BeautifulSoup(newString, "lxml").text
    newString = re.sub(r'\([^)]*\)', '', newString)
    newString = re.sub('"','', newString)
    newString = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in newString.split(" ")])    
    newString = re.sub(r"'s\b","",newString)
    newString = re.sub("[^a-zA-Z]", " ", newString) 
    if summary:
      tokens = [w for w in newString.split()]
      long_words=[]
      for i in tokens:
          if len(i)>1:                  #removing short word < 2
              long_words.append(i)  
      print(len(long_words))
      if len(long_words)==0:
          return "empty_summary_field"
      else:
          return (" ".join(long_words)).strip()
    else:
      tokens = [w for w in newString.split() if not w in stop_words]
      long_words=[]
      for i in tokens:
          #print(len(i))
          if len(i)>=3:                  #removing short word length < 3 
              long_words.append(i)  
      return (" ".join(long_words)).strip()


def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] if word in voc.word2index.keys() else RARE_WORD for word in sentence.split(' ')] + [EOS_token]


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length=10):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


MAX_LENGTH=10
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(data, encoder, decoder, searcher, voc):
    input_sentence = ''
    try:
        # Normalize sentence
        input_sentence = clean(data)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        return ' '.join(output_words)

    except KeyError:
        return "Please try another sentence."

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    ''' For rendering results on HTML GUI'''
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder)
    if request.method == 'POST':
        message = request.form['message']
        output = evaluateInput(message, encoder, decoder, searcher, voc)
    return render_template('home.html', text='Text : {}'.format(message), prediction_text='Summary : {}'.format(output))


if __name__ == '__main__':
    app.run()