# Introduction

All large language models hallucinate from time to time. They can generate convincingly looking, but false 
outputs even when the question is intentionally misleading or unanswerable. A reliable and safe AI system 
must be able to detect such cases and deal with them with dignity, for example, refrain from answering 
altogether or escalate to a human supervisor. A reliable method of detecting hallucinations would help to 
make this capability a reality.

In this project I turned to the recently published paper that suggests a simple and elegant method of 
detecting hallucinations in LLMs based on semantic entropy. While the authors of the paper focused on 
single-modal language models, I found the described method general enough to be applied to any kind of 
large model providing sequential output. For my project, I chose to use a vision-language model LLaVa-7b 
in the visual question answering (VQA) setting. These choices are motivated by the following reasons:
* As far as I know at the time of the writing (September 2024), there’s no published research about applicability of this method to multimodal models, so it would be interesting to try out.
* Hallucinations in LLMs are harder to trigger reliably, while the VQA setting gives lots of opportunities to trigger unsubstantiated answers. I saw it as an opportunity to build an engaging and illustrative demo of the approach.

# Method
I largely based this work on the paper by [Farquhar et. al.](https://www.nature.com/articles/s41586-024-07421-0) and reused parts of the accompanying code for semantic entropy calculation. The method is based on estimating the model’s uncertainty measured by semantic entropy and thresholding the result to get the binary output (hallucinating/not hallucinating).

# Dataset
To test the method described above I chose to use the dataset provided by the [Unsolvable Problem Detection](https://huggingface.co/datasets/MM-UPD/MM-UPD) 
challenge , that aims to test visual-language models ability to detect unanswerable questions in the VQA 
setting and refrain from answering in such situations. For this project I decided to focus on the 
Incompatible Visual Question Detection problem, a setting where the image and question are unrelated. 
This is achieved by complementing the original dataset with examples where images are accompanied by 
mismatching questions. For every question, the model is provided with a set of answer options and is asked
to answer with a single letter.

# Experiments and Results
I calculated the semantic entropy score for a small subset of 700 examples split 70:20 into training and 
testing sets. Then I trained a simple logistic regression model using semantic entropy as a feature and 
“answerability” of the question as the label, which gave me the accuracy of 64.4% on the train set and 62.5% on the test set. 
The threshold for semantic entropy calculated by this method was around 0.58.  As a fun illustration I also included an 
interactive module that demonstrates how a model's semantic entropy score changes for related and unrelated
questions for a set of images. All the code and the interactive demo is available as a [colab notebook](https://colab.research.google.com/drive/16sKYCiWGOK2vEGtWDbD9M2GsKbRl3DK5#scrollTo=fxl40hYwdTW5).