from sp_scorer import SPScorer
import sys

embed_size = 2
maxiter = 10

arg_labels = ['rel1', 'rel2', 'rel3', 'rel4']
train_file = open(sys.argv[1])
preds = {} # pred: index
args = {'NULL':0} # arg: index
numargs = len(arg_labels)

train_data = []

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
    train_data.append((pred_ind, arg_inds))
train_file.close()
train_size = len(train_data)
print >>sys.stderr, "Training data size: %d"%train_size
pred_vocab_size = len(preds)
print >>sys.stderr, "Total number of predicate types: %d"%pred_vocab_size
arg_vocab_size = len(args)
print >>sys.stderr, "Total number of argument types: %d"%arg_vocab_size

sps = SPScorer(numargs, embed_size, pred_vocab_size, arg_vocab_size)
sps_train = sps.get_train_function()
sps_score = sps.get_score_function()

for itr in range(maxiter):
    train_loss = 0.0
    for pred_ind, arg_inds in train_data:
        tr_inputs = [pred_ind] + arg_inds
        tl = sps_train(*tr_inputs)
        train_loss += tl
    print >>sys.stderr, "Finished iter %d, average train loss is %f"%(itr+1, train_loss/train_size)

