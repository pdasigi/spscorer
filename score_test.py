from sp_scorer import SPScorer
import sys
import cPickle

arg_labels = ['ARG0', 'ARG1', 'ARG2', 'AM-LOC', 'AM-TMP']
numargs = len(arg_labels)
(pred_rep, arg_rep, scorer) = cPickle.load(open("spscorer_param.pkl", "rb"))
pred_vocab_size, pred_embed_size = pred_rep.shape
arg_vocab_size, arg_embed_size = arg_rep.shape
scorer_dim = scorer.shape[0]

assert pred_embed_size == arg_embed_size, "Pred and arg reps have different dimensionality."
embed_size = pred_embed_size
assert scorer_dim == embed_size * (numargs + 1), "Scorer's dimensionality is not compatible."

preds = {x.split()[0]:int(x.split()[1]) for x in open("preds.txt").readlines()}
args = {x.split()[0]:int(x.split()[1]) for x in open("args.txt").readlines()}

assert pred_vocab_size == len(preds), "Pred list not the same size inferred from pickled parameters."
assert arg_vocab_size == len(args), "Arg list not the same size inferred from pickled parameters."

sps = SPScorer(numargs, embed_size, pred_vocab_size, arg_vocab_size)
sps.pred_rep.set_value(pred_rep)
sps.arg_rep.set_value(arg_rep)
sps.scorer.set_value(scorer)

get_score = sps.get_score_function()

test_file = open(sys.argv[1])

for line in test_file:
    parts = line.strip().split('\t')
    pred = parts[0].lower()
    if pred in preds:
        pred_ind = preds[pred]
    else:
        pred_ind = preds['UNK']
    arg_inds = [args['NULL']] * numargs
    for i in range(1, len(parts), 2):
        if parts[i] in arg_labels:
            if parts[i+1] in args:
                arg_ind = args[parts[i+1]]
            else:
                arg_ind = args['UNK']
            arg_inds[arg_labels.index(parts[i])] = arg_ind
    score_input = [pred_ind] + arg_inds
    print get_score(*score_input)
test_file.close()
