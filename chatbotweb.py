import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from bs4 import BeautifulSoup
import torch
import pickle
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
from PIL import Image
import plotly.graph_objects as go

# class reviewer(nn.Module):
#   def __init__(self, vocab, dim):
#     super().__init__()
#     self.embedding = nn.Embedding(vocab, dim)
#     self.lin = nn.Linear(dim, 16)
#     self.lin2 = nn.Linear(16, 1)
#     self.tan = nn.Tanh()
#   def forward(self, x):
#     embed = self.embedding(x)
#     mean = torch.mean(embed, axis = 1)
#     lin = self.lin(mean)
#     tan = self.tan(lin)
#     lin2 = self.lin2(tan)
#     tanner = self.tan(lin2)
#     return tanner
class reviewer(nn.Module): #this is the base model
  def __init__(self, vocab, dim):
    super().__init__()
    self.emdding = nn.Embedding(vocab, dim)
    self.lin = nn.Linear(dim, 16)
    self.lin2 = nn.Linear(16, 1)
    self.sig = nn.Sigmoid()
    self.re = nn.ReLU()
  def forward(self, x):
    embed = self.emdding(x)
    mean = torch.mean(embed, axis = 1)
    lin2 = self.lin(mean)
    relu = self.re(lin2)
    sigs = self.sig(relu)
    lin = self.lin2(sigs)
    sig = self.sig(lin)
    return sig


vocab = 367809

my_model = reviewer(vocab, 256)


checkpoint = torch.load('model1_200loops.pth.tar', map_location=torch.device('cpu'))


my_model.load_state_dict(checkpoint['state_dict'])


with open('tokenizer (1).pkl', 'rb') as file:
   tokenizer = pickle.load(file)


buffer = io.StringIO()

df = pd.read_csv('IMDB Dataset.csv')

mask = df['sentiment'].str.contains("positive")


df.loc[mask, 'sentiment'] = 1.0
df.loc[~mask, 'sentiment'] = 0.0

def clean_data_movies(data):
    cleaned_reviews = []

    for review in data:
      soup = BeautifulSoup(review, "html.parser")
      clean_text = soup.get_text(separator="")
      clean_text=clean_text.lower()
      cleaned_reviews.append(clean_text)

    return cleaned_reviews

def lex_order_new(hashing, dataset, padd_width):
    lex = []

    lister = np.zeros((padd_width), dtype=np.int64)
    index = 0
    review = dataset.lower()
    for word in review.split():
        if word in hashing:
            lister[index] = hashing[word]
            index += 1
            if index == padd_width:
                break
    lister = torch.tensor(lister, dtype=torch.long)
    lex.append(lister)

    lex = torch.nn.utils.rnn.pad_sequence(lex, batch_first=True).long()
    return lex

def plot_probability(prob: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        gauge={'axis': {'range': [0, 100]}},
        title={'text': "Probability of being positive (%)"}
    ))
    return fig




def testing_input(padd_width, model, hashing):
    # Get user input via a text box
    user_text = st.text_input("Enter your review:")

    # When the button is clicked, process the input
    if st.button("Analyze"):
        # Check if the input is empty
        if not user_text:
            st.write("Please try again. Empty input is not valid.")
        else:
            # Preprocess the input as needed
            examples = lex_order_new(hashing, user_text, padd_width)

            model.eval()
            prediction = model(examples).item()

            # Display the result based on the prediction value
            if prediction < 0.5:
                st.write("Sounds like a negative reviewer is afoot")
                st.write("here is a graph of your response")
                st.write(plot_probability(prediction))
            else:
                st.write("Somebody is brimming with positivity")
                st.write("here is a graph of your response")
                st.write(plot_probability(prediction))
            




def make_lists():
    #review = ["a b d c a", "a b e", "e g t h", "u h o p", "a b","a b d c", "a b e", "e g t h", "u h o p", "a b","a b d c", "a b e", "e g t h", "u h o p", "a b","a b d c", "a b e", "e g t h", "u h o p", "a b","a b d c", "a b e", "e g t h", "u h o p", "a b","a b d c", "a b e", "e g t h", "u h o p", "a b"]
    #score = [1.,1.,0.,1.,0.,1.,1.,0.,1.,0.,1.,1.,0.,1.,0.,1.,1.,0.,1.,0.,1.,1.,0.,1.,0.,1.,1.,0.,1.,0.]
    review = []
    score = []
    #stop = 1000
    check = 2

    df = pd.read_csv('IMDB Dataset.csv')

    mask = df['sentiment'].str.contains("positive")


    df.loc[mask, 'sentiment'] = 1.0
    df.loc[~mask, 'sentiment'] = 0

    # Convert the entire column to numeric (floats)
    df['sentiment'] = pd.to_numeric(df['sentiment'])

    review = df.review.tolist()
    score = df.sentiment.tolist()



    review = clean_data_movies(review)
    return review, df


reviews, df = make_lists()

my_series = pd.Series(reviews, name='reviews')
df['review'] = my_series

df['sentiment'].info(buf=buffer)
info_str = buffer.getvalue()
st.title('Data used')
st.write("In this model we used publicly available reviews from IMDB.")
st.write("Using a multitude of Python libraries to parse through, clean, and verify the data for training, such as pandas and beautiful soup.")
st.write('Below is the data representing the user sentiment after cleaning.')




st.write('An sample of the reviews and scores after cleaning.')

st.write(df.head(20))
st.text(info_str)

st.title('Training')
st.write('Training improvements: Below is the model from before countless hours of efficiency improvements were made.')
img = Image.open("1000loops.JPG")
st.image(img, caption="Before improvements")
img2 = Image.open('200loop.JPG')
st.image(img2, caption='After')
st.write('The graphs shown above are from the beginning, and after optimization, the model went from taking approximately an hour to a mere 15 minutes to train.')
st.title("Interactive sentiment analysis") # put something on the screen

st.write('The fully trained model in the back end is ready to read, analyze, and predict based on your input.')
testing_input(padd_width=2450, hashing=tokenizer, model=my_model)

