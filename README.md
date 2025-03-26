# CS-541-Homework-5

# 2 Comparing Vanilla RNN with Variants in Sequence Modeling

You will implement and train three different neural networks for sequence modeling: a Vanilla RNN (a simple
RNN with shared weights), and two variants of a NN with a similar architecture to the Vanilla RNN but
which do not share weights. You will compare their performance on a sequence prediction task and analyze
the differences between them.

**Prediction task:** this is a many-to-one regression task, i.e., a sequence of inputs is used for predicting a
single output. In particular, the i-th input sequence Xi = (xi,1, . . . , xi,ℓi ) has length ℓi, where xi,j ∈ R10;
the i-th output is a scalar yi ∈ R.

**Dataset:** You will use a synthetic dataset containing sequences of variable lengths stored in the zip file
homework5_question2_data.zip. Each sequence consists of input features and corresponding target values.
The sequences are generated such that they represent a time-dependent process. Note that ℓi may be different
than ℓj for i ̸ = j. So the (pickled) numpy object X is actually a list of sequences.

**Tasks:**
1. Implement a Vanilla RNN: Implement a Vanilla RNN architecture (needless to say, weights are shared
across time steps). A pytorch starter code is provided in homework5_starter.py. Important: You
are not allowed to use an RNN layer implementation from any library.
2. Implement a NN with Sequences Truncated to the Same Length: Implement a NN where sequences
are truncated to have the same length before training. In other words, if the shortest sequence in the
dataset has length L, all sequences should be truncated to length L before training.
3. Implement a NN with Sequences Padded to the Same Length: Implement another variant of NN where
sequences are padded to have the same length before training. Use appropriate padding techniques to
ensure that all sequences have the same length, and implement a mechanism to ignore the padding
when computing loss and predictions.
4. Train and Compare the Models:
- Train all three models (Vanilla RNN, Truncated NN, Padded NN) on the provided dataset.
- Use a suitable loss function for sequence prediction tasks, such as mean squared error (MSE) or cross-entropy.
- Train each model for a fixed number of epochs or until convergence.
- Monitor and record performance metrics, such as training loss, on a validation set during training.
5. Evaluate and Compare the Models:
- Evaluate the trained models on a separate test dataset.
- Compare the performance of the three models in terms of MSE, convergence speed, and overfitting tendencies.
- Analyze the results and discuss the advantages and disadvantages of each approach in terms of modeling sequences with varying lengths.

**Additional Information:**
You can choose the specific hyperparameters for your models, such as the number of hidden units,
learning rate, batch size, and sequence length. Feel free to use any deep learning framework or library
you are comfortable with, and provide clear code documentation. Note: Be sure to clearly explain your
implementation, provide code comments, and present your results in a well-organized manner in the report.
