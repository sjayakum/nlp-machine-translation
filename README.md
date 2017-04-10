# English to Hindi Machine Translation




### Team Members

<table>
    <tr>
        <th>Name</th>
        <th>Email-id </th>
    </tr>
    <tr>
          <td>Rakesh Ramesh</td>
          <td>rakeshr@usc.edu</td>
    </tr>
    <tr>
          <td>Rishit Jain</td>
          <td>rishitja@usc.edu</td>
    </tr>
    <tr>
          <td>Rahul Kapoor</td>
          <td>rahulkap@usc.edu</td>
    </tr>
    <tr>
          <td>Suraj Jayakumar</td>
          <td>sjayakum@usc.edu</td>
    </tr>
</table>


### Introduction


### Architectures


##### Baseline - Dictionary based unigram text translation

##### Experiment - 1 Character based vanilla RNN using transliteration (one-hot-encoded) for text translation

##### Experiment - 2  Encoder-Decoder LSTM using Word Embeddings (word2vec)

##### Experiment - 3 Encoder-Decoder Attention based GRU (one-hot-encoded)

##### Experiment - 4 Encoder-Decoder Attention based GRU (with word embeddings)

# Machine Translation using Sequence to Sequence with Attention

The code contains Sequence to Sequence attention models implemented using Tensorflow for English -> Hindi and Hindi -> English translation.
This code can be generalized for any language to language conversion.

Dataset used - HindEnCorp - About 280,000 Sentences.
Out of this, only a subset of 140,000 Sentences were used for training and testing the model, since the max sequence length had to be restricted to 20 words for good translation performance

## Files

data_utils.ipynb -> Used for data processing and creating vocabulary and corpus. This is a prereq for other scripts. 

Model Train.ipynb -> Used for training the model. Training time averages around 1.2 sec per 1000 iterations on GTX1060. Model trained for 30,000 Iterations.
					 Refer script for parameters used. Fine tuning may lead to better performance. 

Model Test.ipynb -> Used for testing the model on the test set of around 10,000 sentences. 

Model Test User Sentence.ipynb -> Used for testing the model for user inputs. Also has a baseline for comparision. 

BLEUScore.ipynb -> Evaluation metric. For smoothing function as method2, it gives a BLEU Score of 0.393

This model can be made to perform better by fine tuning parameters, running it on more data or clean data and also by running it for more iterations till losses have been reduced.

## Trained Models Link 

English to Hindi ->  https://drive.google.com/drive/folders/0B9H4uv4_og55cktGRVRUQldpWXc?usp=sharing

Hindi to English ->  https://drive.google.com/drive/folders/0B9H4uv4_og55ZnRJNGhoUGNSQm8?usp=sharing

### References

