import gensim
from nltk.tokenize import word_tokenize
def fixed_attention(sentence, question, w2v_model, embedding_converted = False):
    ignore_word_list = ['.', ',', ':', '?']
    w_sent = word_tokenize(sentence)
    w_question = word_tokenize(question)
    attention = 0.0
    for w1 in w_sent:
        if w1 in ignore_word_list: continue
        sim_list = list()
        for w2 in w_question:
            if (w2 in ignore_word_list): continue
            sim_list.append(abs(w2v_model.similarity(w1,w2)))
        sim_list.sort(reverse=True)
    for sim in sim_list[:3]:
        attention = attention + sim
    return attention
