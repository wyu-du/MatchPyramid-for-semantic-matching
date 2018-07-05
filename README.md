# MatchPyramid-for-semantic-matching
A simple Keras implementation of MatchPyramid model for semantic matching.<br>
Please refer paper：<a href="https://arxiv.org/abs/1602.06359">Text Matching as Image Recognition</a>

## Quick Glance
1. Input Data Format
* Train/Valid set:
```
label	|q1	|q2
1	|Q2119	|D18821
0	|Q2119	|D18822
```

* Test set:
```
q1	|q2
Q2241	|D19682
Q2241	|D19684
```

* Preprocessed Corpus:
```
qid	|words
D9980	|47 0 1 2 3 4 5 6 7 8 9 10
D5796	|21 40 41 42 43 44 14 45
```

* Word Embedding:
```
word	|embedding (50-dimension)
28137	|-0.54645991 2.28509140 ... -0.34052843 -2.01874685
8417	|-9.01635551 -3.80108356 ... 1.86873138 2.14706421
```

* Word Dictionary:
```
word	|wid
preparing	|0
to	|1
rebuild	|2
```

2. Train the model
```
$ python match_pyramid.py
```

3. Loss and Accuracy<br>
<table>
  <tr>
	<th>  </th>
    <th>Best Valid loss</th>
    <th>Best Valid accuracy</th>
  </tr>
  <tr>
    <td>WikiQA</td>
    <td>0.3973</td>
    <td>0.8786</td>
  </tr>
  <tr>
    <td>QouraQP</td>
    <td>0.4525</td>
    <td>0.7797</td>
  </tr>
</table>

## Requirements
* Python 3.5
* TensorFlow 1.8.0
* Keras 2.1.6