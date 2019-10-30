---
layout:     post
title:      Question Answer
subtitle:   Notes about Book speech and language processing
date:       2019-10-29
author:     RJ
header-img: 
catalog: true
tags:
    - NLP

---
<p id = "build"></p>
---

## IR-based factoid QA:
### 25.1.1 Question Processing:<br>
问题处理阶段的主要目标是提取查询，即将关键字传递给IR系统以匹配潜在的文档。有些系统还会提取进一步的信息，如:
- answer type: the entity type (person, location, time, etc.) of the answer.
- focus: the string of words in the question that is likely to be replaced by the
answer in any answer string found.
- question type: is this a definition question, a math question, a list question?

举个栗子：

for the question Which US state capital has the largest population?<br>
- the query processing might produce:<br>
- query: “US state capital has the largest population”<br>
- answer type: city<br>
- focus: state capital<br>

### 25.1.2 Query Formulation
**Query formulation** is the task of creating a query—a list of tokens— to send to an information retrieval system to retrieve documents that might contain answer strings.<br>
Q-F将问题转化为一个tokens列表，传给信息检索系统处理以取得可能含有问题答案的文档。

- wh-word did A verb B → . . . A verb+ed B
- Where is A → A is located in

### 25.1.3 Answer Types
Some systems make use of question classification, the task of finding the answer answer type, the named-entity categorizing the answer. 

- A question like “Who founded Virgin Airlines?” expects an answer of type PERSON. 
- A question like “What Canadian city has the largest population?” expects an answer of type CITY. 

If we know that the answer type for a question is a person, we can avoid examining every sentence in the document collection, instead focusing on sentences mentioning people.

答案类型的确定可以帮助减少搜寻文档的次数，缩小搜索的范围。答案类型可以只用NER来简单确定，也可以用更细粒度的分类层次确定——答案类型分类学，如WordNet,或者根据本体论来构建更系统的分类。

While answer types might just be the named entities like PERSON, LOCATION, and ORGANIZATION described in Chapter 18, we can also use a larger hierarchical set of answer types called an answer type taxonomy. Such taxonomies can be builtautomatically, from resources like WordNet (Harabagiu et al. 2000, Pasca 2003), or they can be designed by hand. Figure 25.4 shows one such hand-built ontology, the
Li and Roth (2005) tagset; a subset is also shown in Fig. 25.3. In this hierarchical tagset, each question can be labeled with a coarse-grained tag like HUMAN or a finegrained tag like 

- HUMAN:DESCRIPTION, HUMAN:GROUP, HUMAN:IND, and so on.
- The HUMAN:DESCRIPTION type is often called a BIOGRAPHY question 

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191029225710QA_answer_type.png)

because the answer is required to give a brief biography of the person rather than just a name. Question classifiers can be built by hand-writing rules like the following rule from (Hovy et al., 2002) for detecting the answer type BIOGRAPHY:

(25.6) who {is | was | are | were} PERSON

Most question classifiers, however, are based on supervised learning, trained on databases of questions that have been hand-labeled with an answer type (Li and Roth, 2002). Either feature-based or neural methods can be used. Feature based methods rely on words in the questions and their embeddings, the part-of-speech of each word, and named entities in the questions. Often, a single word in the question
gives extra information about the answer type, and its identity is used as a feature.
This word is sometimes called the **answer type word** or question headword, and may be defined as the headword of the first NP after the question’s wh-word; headwords are indicated in boldface in the following examples:
- (25.7) Which city in China has the largest number of foreign financial companies?
- (25.8) What is the state flower of California?

In general, question classification accuracies are relatively high on easy question types like PERSON, LOCATION, and TIME questions; 
![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191029225834_typology.png)



**detecting REASON and DESCRIPTION questions can be much harder**.

### 25.1.4 Document and Passage Retrieval
The IR query produced from the question processing stage is sent to an IR engine, resulting in a set of documents ranked by their relevance to the query. Because most answer-extraction methods are designed to apply to smaller regions such as paragraphs, QA systems next divide the top n documents into smaller passages such as sections, paragraphs, or sentences. These might be already segmented in the source document or we might need to run a paragraph segmentation algorithm.

从问题处理阶段产生的IR查询被发送到IR引擎，根据它们与查询的相关性产生一组文档排序。由于大多数的抽取答案的方法都是针对更小的区域设计的，比如段落，QA系统接下来会把前n个文档分成更小的节段，比如段落，句子。这些可能已经在源文档中分段，或者我们可能需要运行一个段落分段算法。

The simplest form of passage retrieval is then to simply pass along every passage to the answer extraction stage. A more sophisticated variant is to filter the passages by running a named entity or answer type classification on the retrieved passages, discarding passages that don’t contain the answer type of the question. It’s also possible to use supervised learning to fully rank the remaining passages, using features like:

• The number of named entities of the right type in the passage

• The number of question keywords in the passage

• The longest exact sequence of question keywords that occurs in the passage

• The rank of the document from which the passage was extracted

• The proximity of the keywords from the original query to each other (Pasca 2003, Monz 2004).

• The number of n-grams that overlap between the passage and the question (Brill et al., 2002).

文档ranked by query and then was divided into smaller passages.<br>
段落 might be filtered by NER or answer type classification, discarding passages which don't contain the answer type.

### 25.1.5 Answer Extraction
The final stage of question answering is to extract a specific answer from the passage, for example responding 29,029 feet to a question like “How tall is Mt. Everest?”. This task is commonly modeled by span labeling: given a passage, identifying the
span of text which constitutes an answer. 
A simple baseline algorithm for answer extraction is to run a named entity tagger on the candidate passage and return whatever span in the passage is the correct answer type. Thus, in the following examples, the underlined named entities would be extracted from the passages as the answer to the HUMAN and DISTANCE-QUANTITY questions:

- “Who is the prime minister of India?”
- Manmohan Singh, Prime Minister of India, had told left leaders that the deal would not be renegotiated.
- “How tall is Mt. Everest?”
- The official height of Mount Everest is 29029 feet

Unfortunately, the answers to many questions, such as DEFINITION questions, don’t tend to be of a particular named entity type. For this reason modern work on answer extraction uses more sophisticated algorithms, generally based on supervised learning. The next section introduces a simple feature-based classifier, after which we turn to modern neural algorithms.

### 25.1.6 Feature-based Answer Extraction
- Answer type match: True if the candidate answer contains a phrase with the correct answer type.
- Pattern match: The identity of a pattern that matches the candidate answer.
- Number of matched question keywords: How many question keywords are contained in the candidate answer.
- Keyword distance: The distance between the candidate answer and query keywords.
- Novelty factor: True if at least one word in the candidate answer is novel, that is, not in the query.
- Apposition features: True if the candidate answer is an appositive to a phrase containing many question terms. Can be approximated by the number of question
- terms separated from the candidate answer through at most three words and one comma (Pasca, 2003).
- Punctuation location: True if the candidate answer is immediately followed by acomma, period, quotation marks, semicolon, or exclamation mark.
- Sequences of question terms: The length of the longest sequence of question terms that occurs in the candidate answer.

### 25.1.7 N-gram tiling answer extraction
An alternative approach to answer extraction, used solely in Web search, is on n-gram tiling, an approach that relies on the redundancy of the web (Brill et al. 2002, Lin 2007). 

This simplified method begins with the snippets returned from the Web search engine, produced by a reformulated query. In the first step, n-gram mining, every unigram, bigram, and trigram occurring in the snippet is extracted and weighted. The weight is a function of the number of snippets in which the n-gram occurred, and the weight of the query reformulation pattern that returned it. In the n-gram filtering step, n-grams are scored by how well they match the predicted answer type. These scores are computed by handwritten filters built for each answer type. Finally, an n-gram tiling algorithm concatenates overlapping ngram fragments into longer answers. A standard greedy method is to start with the highest-scoring candidate and try to tile each other candidate with this candidate. The best-scoring concatenation is added to the set of candidates, the lower-scoring candidate is removed, and the process continues until a single answer is built.


### 25.1.8 Neural Answer Extraction
Neural network approaches to answer extraction draw on the intuition that a question and its answer are semantically similar in some appropriate way. As we’ll see, this intuition can be fleshed out by computing an embedding for the question and an embedding for each token of the passage, and then selecting passage spans whose embeddings are closest to the question embedding.

神经网络方法提取答案的直觉是一个问题和它的答案在语义上以某种适当的方式相似。我们将看到，这种思想可以通过计算问题的嵌入和每个文章段落的嵌入来实现，然后选择嵌入最接近问题嵌入的文章段落跨度。


### 25.1.9 A bi-LSTM-based Reading Comprehension Algorithm
Neural algorithms for reading comprehension are given a question q of l tokens q1,...,ql and a passage p of m tokens p1,..., pm. Their goal is to compute, for each token pi the probability p_start(i) that pi is the start of the answer span, and the probability p_end(i) that pi is the end of the answer span.

Fig. 25.7 shows the architecture of the Document Reader component of the DrQA system of Chen et al. (2017). Like most such systems, DrQA builds an embedding for the question, builds an embedding for each token in the passage, computes a similarity function between the question and each passage word in context, and then uses the question-passage similarity scores to decide where the answer span starts and ends.

![](https://raw.githubusercontent.com/rejae/rejae.github.io/master/img/20191030_BiLSTM_QA.png)


### 25.1.10 BERT-based Question Answering










## Knowledge-based QA:
