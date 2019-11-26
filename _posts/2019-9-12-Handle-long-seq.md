---
layout:     post
title:      Techniques to Handle Very Long Sequences with LSTMs
subtitle:   cheer up
date:       2019-9-12
author:     Jason Brownlee PhD
header-img: 
catalog: true
tags:
    - DL

---
<p id = "build"></p>
---

Long Short-Term Memory or LSTM recurrent neural networks are capable of learning and remembering over long sequences of inputs.

LSTMs work very well if your problem has one output for every input, like time series forecasting or text translation. But LSTMs can be challenging to use when you have very long input sequences and only one or a handful of outputs.

This is often called sequence labeling, or sequence classification.

Some examples include:

Classification of sentiment in documents containing thousands of words (natural language processing).
Classification of an EEG trace of thousands of time steps (medicine).
Classification of coding or non-coding genes for sequences of thousands of DNA base pairs (bioinformatics).
These so-called sequence classification tasks require special handling when using recurrent neural networks, like LSTMs.

In this post, you will discover 6 ways to handle very long sequences for sequence classification problems.

Discover how to develop LSTMs such as stacked, bidirectional, CNN-LSTM, Encoder-Decoder seq2seq and more in my new book, with 14 step-by-step tutorials and full code.

Let’s get started.


## 1. Use Sequences As-Is 

The starting point is to use the long sequence data as-is without change.

This may result in the problem of very long training times.

More troubling, attempting to back-propagate across very long input sequences may result in vanishing gradients, and in turn, an unlearnable model.

A reasonable limit of 250-500 time steps is often used in practice with large LSTM models.

## 2. Truncate Sequences

A common technique for handling very long sequences is to simply truncate them.

This can be done by selectively removing time steps from the beginning or the end of input sequences.

This will allow you to force the sequences to a manageable length at the cost of losing data.

The risk of truncating input sequences is that data that is valuable to the model in order to make accurate predictions is being lost.

## 3. Summarize Sequences

In some problem domains, it may be possible to summarize the input sequences.

For example, in the case where input sequences are words, it may be possible to remove all words from input sequences that are above a specified word frequency (e.g. “and”, “the”, etc.).

This could be framed as only keep the observations where their ranked frequency in the entire training dataset is above some fixed value.

Summarization may result in both focusing the problem on the most salient parts of the input sequences and sufficiently reducing the length of input sequences.

## 4. Random Sampling

A less systematic approach may be to summarize a sequence using random sampling.

Random time steps may be selected and removed from the sequence in order to reduce them to a specific length.

Alternately, random contiguous subsequences may be selected to construct a new sampled sequence over the desired length, care to handle overlap or non-overlap as required by the domain.

This approach may be suitable in cases where there is no obvious way to systematically reduce the sequence length.

This approach may also be used as a type of data augmentation scheme in order to create many possible different input sequences from each input sequence. Such methods can improve the robustness of models when available training data is limited.

## 5. Use Truncated Backpropagation Through Time

Rather than updating the model based on the entire sequence, the gradient can be estimated from a subset of the last time steps.

This is called Truncated Backpropagation Through Time, or TBPTT for short. It can dramatically speed up the learning process of recurrent neural networks like LSTMs on long sequences.

This would allow all sequences to be provided as input and execute the forward pass, but only the last tens or hundreds of time steps would be used to estimate the gradients and used in weight updates.

Some modern implementations of LSTMs permit you to specify the number of time steps to use for updates, separate for the time steps used as input sequences. For example:

The “truncate_gradient” argument in Theano.

## 6. Use an Encoder-Decoder Architecture

You can use an autoencoder to learn a new representation length for long sequences, then a decoder network to interpret the encoded representation into the desired output.

This may involve an unsupervised autoencoder as a pre-processing pass on sequences, or the more recent encoder-decoder LSTM style networks used for natural language translation.

Again, there may still be difficulties in learning from very long sequences, but the more sophisticated architecture may offer additional leverage or skill, especially if combined with one or more of the techniques above.

##  Honorable Mentions and Crazy Ideas
This section lists some additional ideas that are not fully thought through.

Explore splitting the input sequence into multiple fixed-length subsequences and train a model with each subsequence as a separate feature (e.g. parallel input sequences).
Explore a Bidirectional LSTM where each LSTM in the pair is fit on half of the input sequence and the outcomes of each layer are merged. Scale from 2 to more to suitably reduce the length of the subsequences.
Explore using sequence-aware encoding schemes, projection methods, and even hashing in order to reduce the number of time steps in less domain-specific ways.
Do you have any crazy ideas of your own?
Let me know in the comments.

## Further Reading
This section lists some resources for further reading on sequence classification problems:

Sequence labeling on Wikipedia.
A Brief Survey on Sequence Classification, 2010
Summary
In this post, you discovered how you can handle very long sequences when training recurrent neural networks like LSTMs.

## Specifically, you learned:

How to reduce sequence length using truncation, summarization, and random sampling.
How to adjust learning to use Truncated Backpropagation Through Time.
How to adjust the network architecture to use an encoder-decoder structure.

## 转载
[Techniques to Handle Very Long Sequences with LSTMs](https://machinelearningmastery.com/handle-long-sequences-long-short-term-memory-recurrent-neural-networks/)