# Tweet Sentiment Analyzer

A sentiment analysis system for tweets and short text, built with a fine-tuned DistilBERT model. Classifies text as **Positive**, **Neutral**, or **Negative** with confidence scores.


## Features

- 3-class sentiment classification (Positive / Neutral / Negative)
- Multilingual input with automatic translation to English
- Text preprocessing — emoji normalization, slang expansion, stopword removal
- Single text and batch CSV analysis modes
- Interactive confidence gauge and probability charts
- Built with DistilBERT fine-tuned on the [tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) dataset (~45k tweets)


## Model Details

| Setting | Value |
|---|---|
| Base model | `distilbert-base-uncased` |
| Dataset | `cardiffnlp/tweet_eval` (sentiment) |
| Training size | 45,615 tweets |
| Max sequence length | 64 tokens |
| Batch size | 16 |
| Epochs | 4 |
| Optimizer | AdamW (lr=2e-5) |
| Validation accuracy | ~71.85% |
