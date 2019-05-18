import nltk
from nltk.corpus import stopwords
import numpy as np
import json

# source = ['abdomen', 'bat', 'blanket', 'broadleaf plantain', 'cold', 'dog', 'dog harness', 'face', 'feather', 'female dog', 'fine powder snow', 'frog', 'icicle', 'light blue', 'male dog', 'man', 'penny', 'snow', 'snowflake', 'snow on branches or rooftops', 'tooth', 'top', 'upper part of stomach', 'wolf']
# grammar1 = nltk.CFG.fromstring("""
#   S -> NP VP
#   VP -> V NP | V NP PP
#   PP -> P NP
#   V -> "saw" | "ate" | "walked"
#   NP -> "John" | "Mary" | "Bob" | Det N | Det N PP | N N
#   Det -> "a" | "an" | "the" | "my"
#   N -> "man" | "dog" | "cat" | "telescope" | "park" | "bradleaf"
#   P -> "in" | "on" | "by" | "with"
#   """)
# rd_parser = nltk.RecursiveDescentParser(grammar1)
# for word_phrase in source:
#     if len(word_phrase.split()) > 1:
#         to_send = word_phrase.split()
#     else:
#         continue
#     print(to_send)
#     for tree in rd_parser.parse(to_send):
#         print(tree)

source_sep = ['abdomen', 'bat', 'blanket', 'broadleaf', 'plantain', 'cold', 'dog', 'dog', 'harness', 'face', 'feather', 'female', 'dog', 'fine', 'powder', 'snow', 'frog', 'icicle', 'light', 'blue', 'male', 'dog', 'man', 'penny', 'snow', 'snowflake', 'snow', 'on', 'branches', 'or', 'rooftops', 'tooth', 'top', 'upper', 'part', 'of', 'stomach', 'wolf', 'duck', 'rope']
stop_words = stopwords.words('english')
source_sep = [word for word in source_sep if word not in stop_words]
source_sep_set = set(source_sep)
'''
relationships:
equal: if the main part has large likelihood
support : if the complement part has large likelihood with another's main part or its complement part

'''
source_main_vec = {}
done_main = 0
with open('crawl-300d-2M.vec', encoding='utf8') as f:
    f.readline()
    while True:
        line = f.readline()
        if len(line.split()) >= 1:
            word = line.split()[0]
            if word in source_sep_set:
                source_main_vec[word] = [float(x) for x in line.split()[1:]]
                done_main += 1
                if done_main == len(source_sep):
                    print('done transferring word to vec')
                    print('termination upon complete set')
                    break
        else:
            print('done transferring word to vec')
            print('termination upon end of dict')
            break
print(source_main_vec)
with open('wordvec.json', 'w') as f:
    json.dump(source_main_vec, f)
# print(len(source_main_vec))


def cos_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    len_x = np.sqrt(sum(x ** 2))
    len_y = np.sqrt(sum(y ** 2))
    return np.dot(x, y) / (len_x * len_y)


connected_components = []
for n1, w1 in enumerate(source_sep):
    maxsimab = 0
    maxprofile = [0, '', '']
    for n2, w2 in enumerate(source_sep):
        if n1 != n2:
            simab = cos_similarity(source_main_vec[w1], source_main_vec[w2])
            if simab == max(maxsimab, simab):
                maxsimab = max(maxsimab, simab)
                maxprofile = [maxsimab, w1, w2]
            if simab > 0.4:
                print(w1, ' and ', w2, ' has similarity ', simab)
    print('max:', maxprofile[1], ' and ', maxprofile[2], ' has similarity ', maxprofile[0])

target = []


def char_decompose(list):
    max_char = max([len(i) for i in list])
    char_dict = {}
    for i in range(len(max_char)):
        for word in list[:-1 * i]:
            pass


# part1 : source
'''
step 1 : identify and guess compound words and non compound words using glove
snow -> *independent
dog harness -> dog + harness
'''
# for word_phrase in source:
#     word_phrase = word_phrase.split(' ')
#     tagged = nltk.pos_tag(word_phrase)
#     print(tagged)

'''
step 2 : calculate the correlation between main words using cos similarity
'''

'''
step 3 : calculate the correlation between descriptive words using cos similarity
'''

# part2 : target
'''
step1 : for each target[i] target[j] pair, compute min_changing_distance and remaining percent
ex : bat -> bate min_distance = 1, remaining percent = 100 %
ex : redt -> relt min_distance = 1, remaining percent = 75 %
'''

'''
step 2 : for each target[i], link i-j if remaining percent(target[j],target[i]) = max(remaining percent(target[k],target[i]))
form a graph of groups

'''
