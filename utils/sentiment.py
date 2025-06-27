from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import pandas as pd
from tqdm import tqdm

pretrained_model=f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer= AutoTokenizer.from_pretrained(pretrained_model)
# tokenizer.model_max_length = 512
model= AutoModelForSequenceClassification.from_pretrained(pretrained_model)

# #IGNORE THIS
# def N(selected_user,df):
#     if selected_user!='Overall':
#         df=df[df['Contact']==selected_user]
#     return df.shape[0]

def polarity_score(df,col):
    neg_scores,neu_scores,pos_scores=[],[],[]

    with tqdm(total=df.shape[0], desc="Processing") as pbar:
        for m in df[col]:
            encoded_text=tokenizer(m, return_tensors='pt',truncation=True,max_length=512)
            output=model(**encoded_text)
            scores=output[0][0].detach().numpy()
            scores=softmax(scores)
            neg_scores.append(scores[0])
            neu_scores.append(scores[1])
            pos_scores.append(scores[2])
            pbar.update(1)
    # overall_neg_score=np.mean(neg_scores)
    # overall_neu_score=np.mean(neu_scores)
    # overall_pos_score=np.mean(pos_scores)
    data = {
    col: df[col],
    'Negative': neg_scores,
    'Neutral': neu_scores,
    'Positive': pos_scores
    }
    df_sentiment=pd.DataFrame(data)
    return df_sentiment
