# HW 2: Text Classification using LSTMs
**Due: February 27, 9:30 AM**

For this assignment, please complete the _problem set_ found in `hw2-pset.ipynb`. The problem set includes coding
problems as well as written problems.

For the coding problems, you will implement functions defined in the Python files `tokenizer.py`, `model.py`, and
`train_test.py`, replacing the existing code (which raises a `NotImplementedError`) with your own code. **Please
write all code within the relevant function definitions**; failure to do this may break the rest of your code.

For the written problems, please submit your answers in PDF format, using the filename `hw2-written.pdf`. Make sure
to clearly mark each problem in order to minimize the chances of grading errors.

You do not need to submit anything for Problems 1a, 2a, 2c, 3a, or 4a.

You are free to use any resources to help you with this assignment, including books or websites. You are also free
to collaborate with any other student or students in this class. However, you must write and submit your own answers to
the problems, and merely copying another student's answer will be considered cheating on the part of both students. If
you choose to collaborate, please list your collaborators at the top of your `hw2-written.pdf` file.

## Setup

You will need to complete your code problems in Python 3, preferably Python 3.8 or later. Apart from the standard
Python libraries, you will need the following dependencies:
* [NumPy](https://numpy.org)
* [NLTK](https://nltk.org)
* [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
* [tqdm](https://tqdm.github.io/)
* [PyTorch](https://pytorch.org/)
* [🤗 Datasets](https://huggingface.co/docs/datasets/index)

Please refer to their websites for installation instructions.

## Submission

For your submission, please upload the following files to [Gradescope](https://www.gradescope.com):
* `tokenizer.py`
* `model.py`
* `train_test.py`
* `hw2-written.pdf`

Do not change the names of these files, and do not upload any other files to Gradescope. Failure to follow the
submission instructions may result in a penalty of up to 5 points.

## Grading

The point values for each problem are given below. Problems 1c, 2b, and 4d are worth 5 extra credit points, but the 
maximum
possible grade on this assignment is 100 points. If you earn a total of 100 points or more _including the extra
credit points_, then your grade will be 100.

| Problem | Problem Type | Points |
|---|---|---|
| 1b: Benchmarks in NLP | Written | 5 |
| 1c: Extra Credit | Written | 5 EC |
| 1d: Understanding the Dataset | Written | 5 |
| 2b: Extra Credit | Written | 5 EC |
| 2d: Implement the Tokenization Pipeline | Code | 10 |
| 2e: Prepare Model Input | Code | 10 |
| 3b: Define Architecture Components | Code | 10 |
| 3c: Load Pre-Trained Embeddings | Code | 5 |
| 3d: Define Forward Pass | Code | 15 |
| 4b: Model Evaluation | Code | 10 |
| 4c: Model Training | Code | 20 |
| 4d: Experiment | Written | 10 + 5 EC |
| **Total** | | **100** |

### Rubric for Code Problems
Code questions will be graded using a series of [Python unit tests](https://realpython.com/python-testing/). Each
function you implement will be tested on a number of randomly generated inputs, which will match the specifications
described in the function docstrings. **The unit tests will run immediately upon submitting your solution to
[Gradescope](https://www.gradescope.com), and you will be able to see the results as soon as the tests have finished running.** Therefore, you are
encouraged to debug and resubmit your code if one or more unit tests fail.

For code questions, you will receive:
* full points if your code runs and passes all test cases
* at least 5 points if your code runs but fails at least one test case
* 0 points if your code does not run.

Partial credit may be awarded at the TAs' discretion, depending on the correctness of your logic and the severity of
bugs or other mistakes in your code. All code problems will be graded **as if all other code problems had been
answered correctly**. Therefore, an incorrect implementation of one function should (in theory) not affect your
grade on other problems that depend on that function.

### Rubric for Written Problems
For written problems, you will receive:
* full points if results are reported accurately and are accompanied by at least 1 to 2 sentences of thoughtful
  analysis
* at least 2 points if a good-faith effort (according to the TAs' judgement) has been made to answer the question
* 0 points if your answer is blank.

Partial credit may be awarded at the TAs' discretion.

## Late Submissions and Resubmissions

Grading will commence on March 6, 2023, and solutions will be released on that day. Therefore, no late
submissions will be accepted after 9:30 AM on March 6. You may resubmit your solutions as many times as you
like; only the final submission will be graded. If the final submission occurs after the deadline on Febuary 27, then
your submission will be considered late even if you have previously submitted your solution before the deadline.