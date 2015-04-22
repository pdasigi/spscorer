from sp_scorer import SPScorer
import sys
import random

embed_size = 50
maxiter = 10
margin = 5
valid_prop = 0.1

arg_labels = ['ARG0', 'ARG1', 'ARG2', 'AM-LOC', 'AM-TMP']
train_file = open(sys.argv[1])
preds = {} # pred: index
args = {'NULL':0} # arg: index
numargs = len(arg_labels)

train_valid_data = []

for line in train_file:
    parts = line.strip().split('\t')
    pred = parts[0]
    if pred in preds:
        pred_ind = preds[pred]
    else:
        pred_ind = len(preds)
        preds[pred] = pred_ind
    arg_inds = [0]*len(arg_labels)
    for i in range(1, len(parts), 2):
        if parts[i] in arg_labels:
            arg_label = parts[i]
            arg = parts[i+1]
            if arg in args:
                arg_ind = args[arg]
            else:
                arg_ind = len(args)
                args[arg] = arg_ind
            arg_inds[arg_labels.index(arg_label)] = arg_ind
    train_valid_data.append((pred_ind, arg_inds))
train_file.close()

train_valid_size = len(train_valid_data)
print >>sys.stderr, "All data size: %d"%train_valid_size
pred_vocab_size = len(preds)
print >>sys.stderr, "Total number of predicate types: %d"%pred_vocab_size
arg_vocab_size = len(args)
print >>sys.stderr, "Total number of argument types: %d"%arg_vocab_size

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
    num_orig_better = 0
    for (pi, ais), (_, cais) in zip(valid_data, corr_valid_data):
        sc = sps_score(pi, *ais)
        csc = sps_score(pi, *cais)
        if sc > csc:
            num_orig_better += 1
    print >>sys.stderr, "\tvalidation accuracy is %f"%(float(num_orig_better)/valid_size)
    if train_loss == 0.0:
        break

