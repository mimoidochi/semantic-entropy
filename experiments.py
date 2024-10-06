import run_llava
import semantic_entropy
import pandas as pd
import numpy as np
from datasets import load_dataset
from PIL import Image
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generate_model_outputs(prompt, image_file, num_generations = 10, temp = 1):
  outputs = []
  for i in range(num_generations):
    outputs.append(query_model(model, processor, prompt, image_file, temp))
  return outputs

def calculate_uncertainty(model_outputs, question, entailment_model):
  responses = [" ".join([question, output[0]]) for output in model_outputs]
  semantic_ids = get_semantic_ids(responses, model=entailment_model, strict_entailment=True, example=None)

  transition_scores = [output[2] for output in model_outputs]
  log_liks_agg = [np.mean(np.array(log_lik.cpu())) for log_lik in transition_scores]
  log_likelihood_per_semantic_id = logsumexp_by_id(semantic_ids, log_liks_agg, agg='sum_normalized')
  perplexity = np.exp(-np.array(log_liks_agg))
  return semantic_ids, predictive_entropy_rao(log_likelihood_per_semantic_id), perplexity, responses

def calculate_ps(context, question, image_file, entailment_prefix=None):
  if entailment_prefix == None:
    entailment_prefix = question
  prompt = " ".join([context, question])
  model_outputs = generate_model_outputs(prompt, image_file, num_generations = 10, temp = 1)
  model_answer = query_model(model, processor, prompt, image_file, temperature = 0)[0]

  semantic_ids, semantic_entropy, perplexity, responses = calculate_uncertainty(model_outputs, entailment_prefix, entailment_model)
  return semantic_ids, semantic_entropy, perplexity, responses, model_answer

def build_base_prompt(example):
  q = example['question']
  options = '\n'.join([f'{ch}:\n{example[ch]}' for ch in ['A', 'B', 'C', 'D', 'E'] if example[ch] != 'nan'])
  prompt = f'{q}\n{options}'
  return prompt


context = "Answer the following question with a single letter."

def annotate(example):
  question = build_base_prompt(example)
  image = example['decoded_image']

  semantic_ids, semantic_entropy, perplexity, responses, model_answer = calculate_ps(context, question, image, entailment_prefix="The answer is ")
  # print(semantic_ids, semantic_entropy, perplexity, responses, model_answer)
  example['semantic_ids'] = semantic_ids
  example['semantic_entropy'] = semantic_entropy
  example['sampled_responses'] = responses
  example['model_answer'] = model_answer
  return example

# Function to decode the image from bytes
def decode_image(bytes_data):
    # Convert the byte string back into binary bytes
    image_bytes = base64.b64decode(bytes_data)

    # Create a BytesIO stream from the binary bytes and load the image
    image = Image.open(io.BytesIO(image_bytes))
    return image

# Apply the decoding function to the dataset
def decode_images(example):
    example['decoded_image'] = decode_image(example['image'])
    return example

columns = ['index', 'hint', 'question', 'A', 'B', 'C', 'D', 'E', 'answer', 'masked_answer', 'category', 'source', 'l2-category', 'comment', 'split', 'type', 'semantic_ids', 'semantic_entropy', 'sampled_responses', 'model_answer']

if __name__ == '__main__':
    dataset = load_dataset("MM-UPD/MM-UPD", "mmivqd_base", trust_remote_code=True)

    # Assuming 'image_bytes' is the column with image data
    dataset = dataset.map(decode_images)
    dataset = dataset.shuffle(seed=42)

    train_set = dataset['test'].select(range(500))
    test_set = dataset['test'].select(range(500, 700))

    train_set = train_set.map(annotate)
    # Uncomment to store train_set in csv format:
    # train_set.to_csv("upd_ivqd_train_500.csv", columns=columns)

    test_set = test_set.map(annotate)
    # Uncomment to store test_set in csv format:
    # test_set.to_csv("upd_ivqd_test_200.csv", columns=columns)

    train_df = pd.DataFrame(train_set)
    test_df = pd.DataFrame(test_set)

    # Prepare the training data
    X_train = train_df['semantic_entropy'].values.reshape(-1, 1)
    y_train = train_df['type']

    # Prepare the testing data
    X_test = test_df['semantic_entropy'].values.reshape(-1, 1)
    y_test = test_df['type']

    # Train a logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    print(f'Model\'s cofficients: {lr_model.coef_}')
    print(f'Model\'s intercept: {lr_model.intercept_}')

    # Make predictions on the train set
    y_pred = lr_model.predict(X_train)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_train, y_pred)
    print(f"Train set accuracy: {accuracy}")

    # Make predictions on the test set
    y_pred = lr_model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test set accuracy: {accuracy}")

    se_th = -lr_model.intercept_[0] / lr_model.coef_[0][0]
    print(f'Semantic entropy threshold: {se_th:.2f}')