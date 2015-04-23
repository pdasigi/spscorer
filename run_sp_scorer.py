from sp_scorer import SPScorer
import sys
import random, operator
import cPickle

embed_size = 50
maxiter = 10
margin = 5
valid_prop = 0.1
oov_prop = 0.1 #proportion of pred and arg vocab that will be considered OOV

arg_labels = ['ARG0', 'ARG1', 'ARG2', 'AM-LOC', 'AM-TMP']
train_file = open(sys.argv[1])
numargs = len(arg_labels)


train_valid_text = []
pred_freqs = {}
arg_freqs = {}
for line in train_file:
    parts = line.strip().split('\t')
    pred = parts[0].lower()
    if pred in pred_freqs:
        pred_freqs[pred] += 1
    else:
        pred_freqs[pred] = 1
    arg_words = ['NULL'] * len(arg_labels)
    for i in range(1, len(parts), 2):
        if parts[i] in arg_labels:
            arg_label = parts[i]
            arg = parts[i+1].lower()
            arg_words[arg_labels.index(arg_label)] = arg
            if arg in arg_freqs:
                arg_freqs[arg] += 1
            else:
                arg_freqs[arg] = 1
    train_valid_text.append((pred, arg_words))
train_file.close()

# Determine pred and arg oovs
orig_pred_vocab_size = len(pred_freqs)
sorted_pred_freqs = sorted(pred_freqs.items(), key=operator.itemgetter(1))
pred_oov_num = int(oov_prop * orig_pred_vocab_size)
pred_oov = [p for (p, _) in sorted_pred_freqs[:pred_oov_num]]
orig_arg_vocab_size = len(arg_freqs)
sorted_arg_freqs = sorted(arg_freqs.items(), key=operator.itemgetter(1))
arg_oov_num = int(oov_prop * orig_arg_vocab_size)
arg_oov = [a for (a, _) in sorted_arg_freqs[:arg_oov_num]]

# Now create index non-oov preds and args
preds = {x: i for (i, x) in enumerate([p for (p, _) in sorted_pred_freqs[pred_oov_num:]], 1)}
preds['UNK'] = 0
args = {x: i for (i, x) in enumerate([a for (a, _) in sorted_arg_freqs[arg_oov_num:]], 2)}
args['UNK'] = 0
args['NULL'] = 1

pred_file = open("preds.txt", "w")
for pred in preds:
    print >>pred_file, pred, preds[pred]
pred_file.close()
arg_file = open("args.txt", "w")
for arg in args:
    print >>arg_file, arg, args[arg]
arg_file.close()

train_valid_data = []

for pred, arg_words in train_valid_text:
    if pred in preds:
        pred_ind = preds[pred]
    else:
        pred_ind = preds['UNK']
    arg_inds = []
    for arg in arg_words:
        if arg in args:
            arg_ind = args[arg]
        else:
            arg_ind = args['UNK']
        arg_inds.append(arg_ind)
    train_valid_data.append((pred_ind, arg_inds))

train_valid_size = len(train_valid_data)
print >>sys.stderr, "All data size: %d"%train_valid_size
pred_vocab_size = len(preds)
print >>sys.stderr, "Total number of predicate types: %d (%d OOV)"%(pred_vocab_size, pred_oov_num)
arg_vocab_size = len(args)
print >>sys.stderr, "Total number of argument types: %d (%d OOV)"%(arg_vocab_size, arg_oov_num)

valid_size = int(valid_prop * train_valid_size)
random.shuffle(train_valid_data)
valid_data = train_valid_data[:valid_size]
train_data = train_valid_data[valid_size:]

train_size = len(train_data)
print >>sys.stderr, "Training data size is %d and validation data size is %d"%(train_size, valid_size)
corr_valid_data = []
for pred_ind, arg_inds in valid_data:
    corr_pos = random.randint(0, len(arg_inds)-1)
    corr_ind = random.randint(0, arg_vocab_size - 1)
    corr_arg_inds = arg_inds[:corr_pos] + [corr_ind] + arg_inds[corr_pos+1:]
    corr_valid_data.append((pred_ind, corr_arg_inds))

sps = SPScorer(numargs, embed_size, pred_vocab_size, arg_vocab_size, margin=margin)
sps_train = sps.get_train_function()
sps_score = sps.get_score_function()

for itr in range(maxiter):
    train_loss = 0.0
    for pred_ind, arg_inds in train_data:
        tr_inputs = [pred_ind] + arg_inds
        tl = sps_train(*tr_inputs)
        train_loss += tl
    print >>sys.stderr, "Finished iter %d, average train loss is %f"%(itr+1, train_loss/train_size)
    if train_loss == 0.0:
        break

# Check validation accuracy at the end of training
num_orig_better = 0
for (pi, ais), (_, cais) in zip(valid_data, corr_valid_data):
    sc = sps_score(pi, *ais)
    csc = sps_score(pi, *cais)
    if sc > csc:
        num_orig_better += 1
print >>sys.stderr, "Validation accuracy is %f"%(float(num_orig_better)/valid_size)
paramfile = open("spscorer_param.pkl", "wb")
cPickle.dump((sps.pred_rep.get_value(), sps.arg_rep.get_value(), sps.scorer.get_value()), paramfile)
paramfile.close()
