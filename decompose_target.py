'''
decomposes unknown languages into element units
'''
import re
import json
import numpy as np
from nltk.corpus import stopwords
import itertools


with open('target.txt', encoding='utf8') as target:
    word_set = target.readlines()

# word_set[0] = word_set[0][1:]
# with open('target.txt', 'w', encoding='utf8') as target:
#     target.writelines(word_set)
word_set = [w[:-1] for w in word_set if len(w) > 1]
word_set.sort(key=len)
word_profile = {}
for key in word_set:
    word_profile[key] = {}

decompose_words = []

source_words = ['abdomen', 'bat', 'blanket', 'broadleaf plantain', 'cold', 'dog', 'dog harness', 'face', 'feather', 'female dog', 'fine powder snow', 'frog', 'icicle', 'light blue', 'male dog', 'man', 'penny', 'snow', 'snowflake', 'snow on branches or rooftops', 'tooth', 'top', 'upper part of stomach', 'wolf']
associates_source = {}


all_words = ['abdomen', 'bat', 'blanket', 'broadleaf', 'plantain', 'cold', 'dog', 'dog', 'harness', 'face', 'feather', 'female', 'dog', 'fine', 'powder', 'snow', 'frog', 'icicle', 'light', 'blue', 'male', 'dog', 'man', 'penny', 'snow', 'snowflake', 'snow', 'on', 'branches', 'or', 'rooftops', 'tooth', 'top', 'upper', 'part', 'of', 'stomach', 'wolf', 'duck', 'rope']
stop_words = stopwords.words('english')

all_words = [word for word in all_words if word not in stop_words]
with open('wordvec.json') as f:
    vecs = json.load(f)


def cos_similarity(x, y):
    x = np.array(x)
    y = np.array(y)
    len_x = np.sqrt(sum(x ** 2))
    len_y = np.sqrt(sum(y ** 2))
    return np.dot(x, y) / (len_x * len_y)


def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


print('^_^ constructing dictionary')
print('>', word_profile)
print('\n^_^ finding roots')
components = {}
seed = {"tl'ol": "rope", "dətay": "duck"}
for word in word_set:
    original_word = word
    elem = True

    if len(word) != 0:
        print('>processing:', word)
        while not word.isspace():

            exhaust = True
            keys = list(components.keys())
            keys.sort(key=len, reverse=True)
            for parts in keys:
                if parts in word:
                    elem = False

                    word_breakdown = word.replace(parts, ' ' + parts + ' ')
                    word = word.replace(parts, ' ')

                    print('-extracted:', parts)
                    if isinstance(components[parts], str):
                        try:
                            word_profile[original_word]['parts'].append(components[parts])
                        except:
                            word_profile[original_word]['parts'] = [components[parts]]
                        components[components[parts]] += 1
                    else:
                        try:
                            word_profile[original_word]['parts'].append(parts)
                        except:
                            word_profile[original_word]['parts'] = [parts]
                        components[parts] += 1
                    exhaust = False
                    break
            if elem:
                word_breakdown = word
                decompose_words.append(word_breakdown)
                print('!new word registered:', word)
                components[word] = 1
                word_profile[original_word]['parts'] = [word]
                word = ' '
                elem = False
                continue
            elif not exhaust:
                continue
            else:
                decompose_words.append(word_breakdown)
                word = word.split()
                for w in word:
                    if len(w) > 0:
                        mineditdistance = len(w)
                        for root in components:
                            editdistance = edit_distance(w, root)
                            if editdistance < mineditdistance:
                                mineditdistance = editdistance
                                toroot = [root, mineditdistance]

                        if w in seed.keys() or toroot[1] / max(len(w), len(root)) >= 0.4:
                            components[w] = 1
                            print('!new root found and registered:', w)
                            word_profile[original_word]['parts'].append(w)
                            continue

                        else:
                            components[toroot[0]] += 1
                            components[w] = toroot[0]
                            print('!' + w + ' unfound in dict. Auto mapped to ' + toroot[0] + 'instead')
                            # change le to lec
                            word_profile[original_word]['parts'].append(toroot[0])
                break
print('>', word_profile)
print('>', components)
print('\n^_^ finished extracting roots')
done_roots = {}
for root in components:
    done_roots[root] = 0

associates = {}
for root in components:
    for word in word_profile:
        if root in word_profile[word]['parts']:
            for r in word_profile[word]['parts']:
                if r != root:
                    try:
                        associates[root].append(r)
                    except:
                        associates[root] = [r]
source_associates = {}
for n1, w1 in enumerate(source_words):
    w1s = w1.split()
    maxsimab = 0
    maxprofile = [0, '', '']
    for n2, w2 in enumerate(source_words):
        if w1 == w2:
            continue
        maxsimab = 0
        w2s = w2.split()
        for i in w1s:
            for j in w2s:
                try:
                    simab = cos_similarity(vecs[i], vecs[j])
                except:
                    continue
                maxsimab = max(maxsimab, simab)
        if maxsimab > 0.4:
            try:
                source_associates[w1].append(w2)
            except:
                source_associates[w1] = [w2]
            try:
                source_associates[w2].append(w1)
            except:
                source_associates[w2] = [w1]
            print(w1, ' and ', w2, ' has similarity ', simab)


_target_non_compound_n = []
_target_compound_n = []
for word in word_profile.keys():
    if len(word_profile[word]['parts']) == 1:
        _target_non_compound_n.append(word)
    elif len(word_profile[word]['parts']) == 2:
        _target_compound_n.append(word)
print('> indv words:', _target_non_compound_n)
print('> compound words:', _target_compound_n)

# for root in components:
#     if root not in _singles:
#         print(root)

print('\n^_^ translating from given information')
translator = {}

clue = []
for root in components:
    try:
        translator[root] = seed[root]
        clue.append((root, seed[root]))
    except:
        translator[root] = ""
print("> mappings:", translator)
print("! clues:", clue)


importance = {}
for i in range(len(all_words)):
    maxsimij = 0
    for j in range(len(all_words)):
        if i != j:
            simij = cos_similarity(vecs[all_words[i]], vecs[all_words[j]])
            if simij > maxsimij:
                maxsimij = max(simij, maxsimij)

    importance[all_words[i]] = maxsimij
print('>importance:', importance)

source_profile = {}
extract_parts = []
for wd in source_words:
    source_profile[wd] = {}

    source_profile[wd]['parts'] = [word for word in wd.split() if word not in stop_words]
    if len(source_profile[wd]['parts']) > 2:
        for n, w in enumerate(source_profile[wd]['parts']):
            source_profile[wd]['parts'][n] = [importance[source_profile[wd]['parts'][n]], source_profile[wd]['parts'][n]]
        source_profile[wd]['parts'].sort(reverse=True)
        source_profile[wd]['parts'] = [source_profile[wd]['parts'][top2][1] for top2 in range(2)]
        extract_parts += source_profile[wd]['parts']
print(source_profile)


def make_guess(a, filter=True, gimme=False):
    maxprofile = []
    for word in all_words:
        if vecs[word][0] != a[0] or filter == False:
            simab = cos_similarity(a, vecs[word])
            maxprofile.append([word, simab])
    maxprofile.sort(key=lambda x: x[1], reverse=True)
    if gimme:
        return profile
    for profile in maxprofile:
        if not filter:
            return profile[0]
        elif profile[0] not in translator.values():
            return profile[0]



# target
clue_source = []
done_pairs = {}


def processor(clue):
    first_clue_source = clue[1]
    first_clue_target = clue[0]

    # target find similar

    peanut_target = []
    for word in word_profile.keys():
        if first_clue_target in word_profile[word]['parts']:
            peanut_target.append([word])
            for part in word_profile[word]['parts']:
                if part != first_clue_target:
                    peanut_target[-1].append(part)

    # find nearest concept to idenitfy the source word
    paraphrase = make_guess(vecs[first_clue_source], filter=True)
    print('\n>:closest word to ' + first_clue_source + ' is ' + paraphrase)

    source = []
    for word in source_words:
        if paraphrase in word:
            source.append(word)
    print(source)
    for s in source:
        ls = s.split()
        if len(ls) == 1:
            vec_source = vecs[paraphrase]
        else:
            vec_source = 0
            for w in ls:
                vec_source += np.array(vecs[w])
    print('\n>:', paraphrase + ' can be found in ', source)
    if len(source) == 1:
        print(peanut_target[0][0])
        translator[peanut_target[0][0]] = source[0]
        done_pairs[source[0]] = peanut_target[0][0]
        for root in word_profile[peanut_target[0][0]]['parts']:
            done_roots[root] += 1
        print('\n>:guess that ' + peanut_target[0][0] + ' is ' + source[0] + ' because it is closest to ' + first_clue_source)

    else:
        pass

    # extract the other root using formula ideal_root_vector = source_vector - clue_vector
    peanut_root_ideal = vec_source - np.array(vecs[first_clue_source])

    translator[peanut_target[0][1]] = make_guess(peanut_root_ideal, filter=False)
    done_pairs[translator[peanut_target[0][1]]] = peanut_target[0][1]
    for root in word_profile[peanut_target[0][1]]['parts']:
        done_roots[root] += 1
    print('\n!:guess that ' + peanut_target[0][1] + ' is ' + translator[peanut_target[0][1]] + ' because it is closest to ' + '+'.join(source) + ' - ' + first_clue_source)
    clue_source.insert(0, (peanut_target[0][1], translator[peanut_target[0][1]]))

    # combine and extract the whole meaning by using formula ideal_target_vector = clue_vector + real_root_vector
    if source[0] == translator[peanut_target[0][1]]:
        peanut_target_ideal = np.array(vecs[first_clue_source]) + np.array(vecs[translator[peanut_target[0][1]]])
        translator[peanut_target[0][0]] = make_guess(peanut_target_ideal, filter=True)
        done_pairs[translator[peanut_target[0][0]]] = peanut_target[0][0]
        print('\n>:reguess that ' + peanut_target[0][0] + ' is ' + translator[peanut_target[0][0]] + ' because it is closest to ' + translator[peanut_target[0][1]] + ' + ' + first_clue_source)
        # clue.insert(0, (peanut_target[0][0], translator[peanut_target[0][0]]))
    print(translator)


for c in clue:
    processor(c)

source_compound = []
source_non_compound = []
source_unsure = []
source_extra = []
# determine compound or single:
print(done_pairs)
print(word_profile)
print('----')
for word in source_words:
    if len(word.split()) > 1:
        source_profile[word]['compound'] = True
        source_compound.append(word)
    else:
        try:
            is_c = len(word_profile[done_pairs[word]]['parts'])
        except:
            source_unsure.append(word)
            continue
        if is_c == 1:
            source_profile[word]['compound'] = False
            source_non_compound.append(word)
        else:
            print(is_c)
            source_profile[word]['compound'] = True
            source_compound.append(word)
print(source_non_compound)
print(source_compound)
print(source_unsure)

classified_confidence = {}
for w in source_compound:
    classified_confidence[w] = 1
for w in source_non_compound:
    classified_confidence[w] = 1
while len(source_unsure) != 0:
    single_unk = []
    for unk in source_unsure:
        maxsimij = -1
        for compound in source_compound:

            if len(source_profile[compound]['parts']) == 1:
                simij = cos_similarity(vecs[unk], vecs[compound]) * classified_confidence[compound]
                if simij > maxsimij:
                    maxsimij = simij
                    tagger = 'non_compound'
                    closest = compound
            else:
                for parts in source_profile[compound]['parts']:
                    simij = cos_similarity(vecs[unk], vecs[parts])
                    if simij > maxsimij:
                        maxsimij = simij
                        tagger = 'non_compound'
                        closest = parts
        for non_c in source_non_compound:
            simij = cos_similarity(vecs[unk], vecs[non_c]) * classified_confidence[non_c]
            if simij >= maxsimij - 0.01:
                maxsimij = max(simij, maxsimij)
                tagger = 'compound'
                closest = non_c
        single_unk.append([maxsimij, unk, tagger, closest])
    single_unk.sort(reverse=True)
    if single_unk[0][2] == 'compound':
        if len(source_compound) < len(_target_compound_n):
            source_compound.append(single_unk[0][1])
        else:
            source_non_compound.append(single_unk[0][1])
        classified_confidence[single_unk[0][1]] = single_unk[0][0]
        print(single_unk[0][1] + ' classified to compound with confidence ', single_unk[0][0], single_unk[0][3])
    else:
        source_non_compound.append(single_unk[0][1])
        print(single_unk[0][1] + ' classified to non_compound with confidence ', single_unk[0][0], single_unk[0][3])
        classified_confidence[single_unk[0][1]] = single_unk[0][0]
    print('u', source_unsure)
    print('s', source_compound)
    print('n', source_non_compound)
    # print('e', source_extra)
    print(single_unk[0][1])
    source_unsure.remove(single_unk[0][1])

print(source_profile)
print(len(source_compound), len(_target_compound_n))
print(len(source_non_compound), len(_target_non_compound_n))

# source
confidence = {}
print(components)
print(done_roots)
while len(clue_source) != 0:
    for clue in clue_source:
        confidence[clue[1]] = []
        clue_contribute_n = components[clue[0]]
        for words in source_words:
            if words in translator.values():
                continue
            else:
                try:
                    maxsimab = max([cos_similarity(vecs[clue[1]], vecs[w]) for w in words.split()])
                except:
                    pass
            if words in source_compound:
                iscompound = True
            else:
                iscompound = False
            confidence[clue[1]].append([maxsimab, words, iscompound])
        confidence[clue[1]].sort(key=lambda x: x[0], reverse=True)
    print(confidence)
    maxconfidence = 0
    for clue in confidence:
        clue_confidence = confidence[clue][0][0]
        if clue_confidence > maxconfidence:
            maxconfidence = clue_confidence
            maxclue = clue
            selected_clue = confidence[clue]
            selected_clue_name = done_pairs[clue]
    target_candidates = [x for x in associates[done_pairs[clue]] if len(translator[x]) == 0]
    order = []
    source_candidates = []
    word_associates = []
    word_relates = []
    for relates in range(components[selected_clue_name] - done_roots[selected_clue_name]):

        print(selected_clue[relates])
        word_relates.append(selected_clue[relates][1])
        word_associates.append([])
        vec_new = 0
        for w in selected_clue[relates][1].split():
            vec_new += np.array(vecs[w])

            for word in source_non_compound:
                if word not in done_pairs:
                    vec_combine = np.array(vecs[word]) + np.array(vecs[clue])
                    simij = cos_similarity(vec_new, vec_combine)
                    word_associates[-1].append([simij, word])
        word_associates[-1].sort(reverse=True)
        order.append(word_associates[-1][0][0])
    # print(word_associates[0])
    # print(word_associates[1])
    # print(word_associates[2])
    order = np.array(order)
    sorted_word_relates = []
    for n in range(components[selected_clue_name] - done_roots[selected_clue_name]):
        selector = np.argmax(order)
        i = 0
        while True:
            to_select = word_associates[selector][i][1]
            if to_select not in source_candidates:
                source_candidates.append(to_select)
                order[selector] = 0
                sorted_word_relates.append(word_relates[selector])
                break
            else:
                i += 1
    print(source_candidates)
    # print(source_associates)
    source_connectivity = []
    target_connectivity = []
    for n, x in enumerate(target_candidates):
        try:
            target_connectivity.append([len(associates[x]), x])
        except:
            target_connectivity.append([0, x])
    for n, x in enumerate(source_candidates):
        try:
            source_connectivity.append([len(source_associates[x]), x, sorted_word_relates[n]])
        except:
            source_connectivity.append([0, x, sorted_word_relates[n]])
    source_connectivity.sort(reverse=True)
    target_connectivity.sort(reverse=True)
    print(source_connectivity)
    print(target_connectivity)
    new_add = {}
    for m, n in zip(source_connectivity, target_connectivity):
        new_add[n[1]] = m[1]
        for w in word_profile:
            if n[1] in word_profile[w]['parts'] and done_pairs[clue] in word_profile[w]['parts']:
                print('i')
                vec_new = 0
                new_add[w] = m[2]

    print(translator)
    print(word_profile)
    print(word_relates)
    for n, e in enumerate(clue_source):
        if done_pairs[clue] in e:
            clue_source.pop(n)
    for i, j in zip(new_add.keys(), new_add.values()):
        if clue_source in _target_non_compound_n:
            clue_source.append(i, j)
        translator[i] = j
    print(translator)
    break
    # break
    # if components[target_candidates] == 0:
    #     pass

    # print(order)
translator['nani'] = ''
print(word_profile)
rest_target = []
for word in word_profile:
    if len(word_profile[word]['parts']) == 1:
        if word not in translator.keys():
            rest_target.append(word)
        elif len(translator[word]) == 0:
            rest_target.append(word)
print(source_non_compound)
rest_source = []
print('t:', translator)
for word in source_non_compound:
    if word not in translator.values():
        rest_source.append(word)

print(rest_target)
print(rest_source)

ls = [i for i in range(len(rest_target))]
for n in list(itertools.permutations(ls)):
    for i in range(len(rest_target)):
        translator[rest_target[i]] = rest_source[ls[i]]


rest_target_c = []
for word in word_profile:
    if len(word_profile[word]['parts']) == 2:
        if word not in translator.keys():
            rest_target_c.append(word)
        elif len(translator[word]) == 0:
            rest_target_c.append(word)
rest_source_c = []
print('t:', translator)
for word in source_compound:
    if word not in translator.values():
        rest_source_c.append(word)
print(rest_source_c)
print(rest_target_c)
ls = [i for i in range(len(rest_target_c))]
for n in list(itertools.permutations(ls)):
    for i in range(len(rest_target)):
        translator[rest_target_c[i]] = rest_source_c[ls[i]]
print(translator)
