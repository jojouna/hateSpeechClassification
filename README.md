# SUMMARY 
This page contains the ipynb file for hate speech classification conducted as an assignment for NLP study. You can find a short summary of this study here. If you wish to see the whole code, please see the attached ipynb file.

Dataset has enough texts (31060). The number of hateful and non-hateful texts are balanced (Each 15530).
The dataset contains various hashtags, mentions, emojis, special characters etc such are common on social media. It also has corrupted texts (e.g. ð, â¦).
Majority of words from both single and 2-grams were all stopwords that needed to be extracted.

Text preprocessing was performed to convert emojis into meaningful texts, expand contractions (e.g. can't -> cannot), apply tokenizer using TreebankWordTokenizer, remove stopwords and punctutations, and Lemmatize.
Used TF-IDF to calculate the importance of words in each document relative to the entire training dataset.

MultinomialNB was used as a model since it is common for text classification with word counts or frequencies.

Further improved the model through hyperparameter tuning of the TF-IDF vectorizer. I introduced both unigrams and bigrams (1- and 2-word combinations), set min_df=2 to exclude words and n-grams appearing in fewer than two documents, and enabled sublinear_tf=True to apply logarithmic scaling to term frequency, reducing the influence of frequently repeated words.

After changing the hyperparameters, the score lifted precision (69% → 75%), recall (61% → 75%) and accuracy (75% → 76%). 
