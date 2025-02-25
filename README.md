# sentiment-analysis
This sentiment analysis model is a supervised embedding neural network designed to classify text sentiment from Kaggle datasets, which include both Steam Reviews and movie reviews. The model uses embeddings to convert raw text into dense vector representations, allowing it to capture the subtle semantic relationships between words and phrases.

**Data Sources & Preprocessing:**
Steam Reviews: These reviews are rich with emojis, extraneous characters, and a distinct, often sarcastic tone typical of gaming communities. This noise makes them more challenging to clean and preprocess.
Movie Reviews: Compared to Steam reviews, movie reviews are generally cleaner and easier to process, leading to more accurate sentiment extraction.

**Model Training & Performance:**
Over the course of 401 training loops (as shown in our accompanying PNG file), our model achieved an accuracy just above 90%.
The validation curve did not plateau, which suggests that the model had not yet overfit the training data, indicating promising generalization capabilities.
