import gensim
from nltk.tokenize import word_tokenize
def fixed_attention(sentence, question, w2v_model, embedding_converted = False):
    w_sent = word_tokenize(sentence)
    w_question = word_tokenize(question)
    attention = 1.0
    for w1 in w_sent:
        sim_list = [w2_model.similarity(w1,w2) for w2 in w_question]
        sim_list.sort(reverse=True)
    for sim in sim_list[:3]:
        attention = attention * sim
    return attention
