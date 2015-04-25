import theano, numpy
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class SPScorer(object):
    def __init__(self, numargs, embed_size, pred_vocab_size, arg_vocab_size, initial_pred_rep=None, initial_arg_rep = None, margin = 5, lr=0.01, activation=T.nnet.sigmoid):
        numpy_rng = numpy.random.RandomState(12345)
        theano_rng = RandomStreams(54321)
        self.lr = lr
        #margin = 5
        # Initializing predicate representations
        if initial_pred_rep is not None:
            num_preds, pred_dim = initial_pred_rep.shape
            assert pred_vocab_size == num_arrays, "Initial predicate representation is not the same size as pred_vocab_size"
            assert embed_size == pred_dim, "Initial predicate representation does not have the same dimensionality as embed_size"
        else:
            initial_pred_rep_range = 4 * numpy.sqrt(6. / (pred_vocab_size + embed_size))
            initial_pred_rep = numpy.asarray(numpy_rng.uniform(low = -initial_pred_rep_range, high = initial_pred_rep_range, size = (pred_vocab_size, embed_size)))
            
        self.pred_rep = theano.shared(value=initial_pred_rep, name='P')
        
        # Initializing argument representations
        if initial_arg_rep is not None:
            arg_rep_len, arg_dim = initial_arg_rep.shape
            assert arg_vocab_size == arg_rep_len, "Initial argument representation is not the same size as arg_vocab_size"
            assert embed_size == arg_dim, "Initial argument representation does not have the same dimensionality as embed_size"
        else:
            initial_arg_rep_range = 4 * numpy.sqrt(6. / (arg_vocab_size + embed_size))
            initial_arg_rep = numpy.asarray(numpy_rng.uniform(low = -initial_arg_rep_range, high = initial_arg_rep_range, size = (arg_vocab_size, embed_size)))
            
        self.arg_rep = theano.shared(value=initial_arg_rep, name='A')
        
        # Initialize scorer
        scorer_dim = embed_size * (numargs + 1) # Predicate is +1
        initial_scorer_range = 4 * numpy.sqrt(6. / scorer_dim)
        initial_scorer = numpy.asarray(numpy_rng.uniform(low = -initial_scorer_range, high = initial_scorer_range, size = scorer_dim))
        self.scorer = theano.shared(value=initial_scorer, name='s')
        
        # Initialize indicator
        indicator_dim = embed_size * (numargs + 1) # Predicate is +1
        initial_indicator_range = 4 * numpy.sqrt(6. / (indicator_dim + numargs))
        initial_indicator = numpy.asarray(numpy_rng.uniform(low = -initial_indicator_range, high = initial_indicator_range, size = (indicator_dim, numargs)))
        self.indicator = theano.shared(value=initial_indicator, name='I')
        
        # Define symbolic pred-arg
        self.pred_ind = T.iscalar('p')
        self.arg_inds = T.iscalars(numargs)
        pred = self.pred_rep[self.pred_ind].reshape((1, embed_size))
        args = self.arg_rep[self.arg_inds].reshape((1, embed_size * numargs))
        pred_arg = activation(T.concatenate([pred, args], axis=1))
        
        # Define symbolic rand pred-arg for training scorer
        rand_pred_ind = theano_rng.random_integers(low=0, high=pred_vocab_size-1)
        rand_arg_inds = theano_rng.random_integers([1, numargs], low=0, high=arg_vocab_size-1)
        rand_pred = self.pred_rep[rand_pred_ind].reshape((1, embed_size))
        rand_args = self.arg_rep[rand_arg_inds].reshape((1, embed_size * numargs))
        rand_pred_arg = activation(T.concatenate([rand_pred, rand_args], axis=1))

        # Define symbolic pred_rand-arg for training indicator
        pred_rand_arg = activation(T.concatenate([pred, rand_args], axis=1))

        # Define scores and loss
        self.corr_score = T.sum(T.dot(pred_arg, self.scorer))
        rand_score = T.sum(T.dot(rand_pred_arg, self.scorer))
        self.margin_loss = T.maximum(0, margin - self.corr_score + rand_score)
        
        # Define indicator values and loss
        orig_ind_labels = T.constant(numpy.zeros(numargs))
        self.indicator_pred = T.nnet.sigmoid(T.dot(pred_arg, self.indicator))
        rand_ind_labels = T.constant(numpy.ones(numargs))
        rand_indicator_pred = T.nnet.sigmoid(T.dot(pred_rand_arg, self.indicator))
        self.indicator_loss = T.mean((self.indicator_pred - orig_ind_labels) ** 2) + T.mean((rand_indicator_pred - rand_ind_labels) ** 2)
        
        # Define params and inputs
        self.score_params = [self.pred_rep, self.arg_rep, self.scorer]
        self.indicator_params = [self.pred_rep, self.arg_rep, self.indicator]
        self.score_ind_inputs = [self.pred_ind] + list(self.arg_inds)

    def get_score_train_function(self):
        gparams = T.grad(self.margin_loss, self.score_params)
        updates = []
        for param, gparam in zip(self.score_params, gparams):
            updates.append((param, param - self.lr * gparam))
        return theano.function(self.score_ind_inputs, self.margin_loss, updates=updates)

    def get_indicator_train_function(self):
        gparams = T.grad(self.indicator_loss, self.indicator_params)
        updates = []
        for param, gparam in zip(self.indicator_params, gparams):
            updates.append((param, param - self.lr * gparam))
        return theano.function(self.score_ind_inputs, self.indicator_loss, updates=updates)

    def get_score_function(self):
        return theano.function(self.score_ind_inputs, self.corr_score)
    
    def get_indicator_function(self):
        return theano.function(self.score_ind_inputs, self.indicator_pred)
