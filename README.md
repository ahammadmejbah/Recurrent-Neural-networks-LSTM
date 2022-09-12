<h3><center>Recurrent Neural networks -Deep Neuron AI </center></h3>

A recurrent neural network (RNN) is a class of artificial neural network where connections between units form a directed cycle. This creates an internal state of the network which allows it to exhibit dynamic temporal behavior.

```python
keras.layers.recurrent.SimpleRNN(units, activation='tanh', use_bias=True, 
                                 kernel_initializer='glorot_uniform', 
                                 recurrent_initializer='orthogonal', 
                                 bias_initializer='zeros', 
                                 kernel_regularizer=None, 
                                 recurrent_regularizer=None, 
                                 bias_regularizer=None, 
                                 activity_regularizer=None, 
                                 kernel_constraint=None, recurrent_constraint=None, 
                                 bias_constraint=None, dropout=0.0, recurrent_dropout=0.0)
```

#### Arguments:

<ul>
<li><strong>units</strong>: Positive integer, dimensionality of the output space.</li>
<li><strong>activation</strong>: Activation function to use
    (see <a href="http://keras.io/activations/">activations</a>).
    If you pass None, no activation is applied
    (ie. "linear" activation: <code>a(x) = x</code>).</li>
<li><strong>use_bias</strong>: Boolean, whether the layer uses a bias vector.</li>
<li><strong>kernel_initializer</strong>: Initializer for the <code>kernel</code> weights matrix,
    used for the linear transformation of the inputs.
    (see <a href="https://keras.io/initializers/">initializers</a>).</li>
<li><strong>recurrent_initializer</strong>: Initializer for the <code>recurrent_kernel</code>
    weights matrix,
    used for the linear transformation of the recurrent state.
    (see <a href="https://keras.io/initializers/">initializers</a>).</li>
<li><strong>bias_initializer</strong>: Initializer for the bias vector
    (see <a href="https://keras.io/initializers/">initializers</a>).</li>
<li><strong>kernel_regularizer</strong>: Regularizer function applied to
    the <code>kernel</code> weights matrix
    (see <a href="https://keras.io/regularizers/">regularizer</a>).</li>
<li><strong>recurrent_regularizer</strong>: Regularizer function applied to
    the <code>recurrent_kernel</code> weights matrix
    (see <a href="https://keras.io/regularizers/">regularizer</a>).</li>
<li><strong>bias_regularizer</strong>: Regularizer function applied to the bias vector
    (see <a href="https://keras.io/regularizers/">regularizer</a>).</li>
<li><strong>activity_regularizer</strong>: Regularizer function applied to
    the output of the layer (its "activation").
    (see <a href="https://keras.io/regularizers/">regularizer</a>).</li>
<li><strong>kernel_constraint</strong>: Constraint function applied to
    the <code>kernel</code> weights matrix
    (see <a href="https://keras.io/constraints/">constraints</a>).</li>
<li><strong>recurrent_constraint</strong>: Constraint function applied to
    the <code>recurrent_kernel</code> weights matrix
    (see <a href="https://keras.io/constraints/">constraints</a>).</li>
<li><strong>bias_constraint</strong>: Constraint function applied to the bias vector
    (see <a href="https://keras.io/constraints/">constraints</a>).</li>
<li><strong>dropout</strong>: Float between 0 and 1.
    Fraction of the units to drop for
    the linear transformation of the inputs.</li>
<li><strong>recurrent_dropout</strong>: Float between 0 and 1.
    Fraction of the units to drop for
    the linear transformation of the recurrent state.</li>
</ul>


#### Backprop Through time  

Contrary to feed-forward neural networks, the RNN is characterized by the ability of encoding longer past information, thus very suitable for sequential models. The BPTT extends the ordinary BP algorithm to suit the recurrent neural
architecture.

<img width="536" alt="image" src="https://user-images.githubusercontent.com/56669333/189617695-a510fe1c-388a-482d-b8dd-302e61417450.png">

## IMDB sentiment classification task

This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. 

IMDB provided a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. 

There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. 

http://ai.stanford.edu/~amaas/data/sentiment/

## LSTM
A LSTM network is an artificial neural network that contains LSTM blocks instead of, or in addition to, regular network units. A LSTM block may be described as a "smart" network unit that can remember a value for an arbitrary length of time. 

Unlike traditional RNNs, an Long short-term memory network is well-suited to learn from experience to classify, process and predict time series when there are very long time lags of unknown size between important events.

<img width="583" alt="image" src="https://user-images.githubusercontent.com/56669333/189617890-b0fb1d77-37f0-4160-a92e-619a768fbf4b.png">


```python
keras.layers.recurrent.LSTM(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
                            kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                            bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
                            recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                            kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
                            dropout=0.0, recurrent_dropout=0.0)
```


#### Arguments

<ul>
<li><strong>units</strong>: Positive integer, dimensionality of the output space.</li>
<li><strong>activation</strong>: Activation function to use
    If you pass None, no activation is applied
    (ie. "linear" activation: <code>a(x) = x</code>).</li>
<li><strong>recurrent_activation</strong>: Activation function to use
    for the recurrent step.</li>
<li><strong>use_bias</strong>: Boolean, whether the layer uses a bias vector.</li>
<li><strong>kernel_initializer</strong>: Initializer for the <code>kernel</code> weights matrix,
    used for the linear transformation of the inputs.</li>
<li><strong>recurrent_initializer</strong>: Initializer for the <code>recurrent_kernel</code>
    weights matrix,
    used for the linear transformation of the recurrent state.</li>
<li><strong>bias_initializer</strong>: Initializer for the bias vector.</li>
<li><strong>unit_forget_bias</strong>: Boolean.
    If True, add 1 to the bias of the forget gate at initialization.
    Setting it to true will also force <code>bias_initializer="zeros"</code>.
    This is recommended in <a href="http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf">Jozefowicz et al.</a></li>
<li><strong>kernel_regularizer</strong>: Regularizer function applied to
    the <code>kernel</code> weights matrix.</li>
<li><strong>recurrent_regularizer</strong>: Regularizer function applied to
    the <code>recurrent_kernel</code> weights matrix.</li>
<li><strong>bias_regularizer</strong>: Regularizer function applied to the bias vector.</li>
<li><strong>activity_regularizer</strong>: Regularizer function applied to
    the output of the layer (its "activation").</li>
<li><strong>kernel_constraint</strong>: Constraint function applied to
    the <code>kernel</code> weights matrix.</li>
<li><strong>recurrent_constraint</strong>: Constraint function applied to
    the <code>recurrent_kernel</code> weights matrix.</li>
<li><strong>bias_constraint</strong>: Constraint function applied to the bias vector.</li>
<li><strong>dropout</strong>: Float between 0 and 1.
    Fraction of the units to drop for
    the linear transformation of the inputs.</li>
<li><strong>recurrent_dropout</strong>: Float between 0 and 1.
    Fraction of the units to drop for
    the linear transformation of the recurrent state.</li>
</ul>



Gated recurrent units are a gating mechanism in recurrent neural networks. 

Much similar to the LSTMs, they have fewer parameters than LSTM, as they lack an output gate.

<img width="614" alt="image" src="https://user-images.githubusercontent.com/56669333/189618069-8f5bc8b5-9823-4d0a-8732-5163a331e79d.png">

```python
keras.layers.recurrent.GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, 
                           kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
                           bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, 
                           bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, 
                           recurrent_constraint=None, bias_constraint=None, 
                           dropout=0.0, recurrent_dropout=0.0)
```


