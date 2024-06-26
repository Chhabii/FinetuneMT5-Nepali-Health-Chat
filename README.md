---
license: apache-2.0
datasets:
- NepaliAI/Nepali-HealthChat
- NepaliAI/Nepali-Health-Fact
language:
- en
- ne
metrics:
- bleu
pipeline_tag: text2text-generation
tags:
- health
- medical
- nlp
---
# Open-Source Nepali Health-QA Language Model with FineTuned Transformers 
MT5-small is finetuned with large corups of Nepali Health Question-Answering Dataset.

## Table of Contents
- [Introduction](#introduction)
- [Training Procedure](#training-procedure)
- [Use Case](#use-case)
- [Evaluation](#evaluation)
- [FineTune](#finetune)
  
## Introduction
In the ever-evolving landscape of Natural Language Processing (NLP), our project, titled ”OPEN-SOURCE NEPALI HEALTH-QA LANGUAGE MODEL WITH FINE-TUNED TRANSFORMERS AND EXTERNAL KNOWLEDGE BASES,” represents a dedicated effort to address the pressing need for accessible and accurate health-related information in the Nepali language. This project is driven by the recognition of the critical role that natural language understanding plays in fostering meaningful interactions, particularly in the domain of health-related inquiries. Our question-answering model doesn’t just answer queries; it provides a valuable second opinion, offering additional perspectives and comprehensive insights on healthcare matters.

## Training Procedure
The model was trained for more than 125 epochs with the following training parameters:

Learning Rate: 2e-4

Batch Size: 2

Gradient Accumulation Steps: 8

FP16 (mixed-precision training): Disabled

Optimizer: Adafactor

The training loss consistently decreased, indicating successful learning.

## Use Case

```python

  !pip install transformers sentencepiece

  from transformers import MT5ForConditionalGeneration, AutoTokenizer 
  # Load the trained model
  model = MT5ForConditionalGeneration.from_pretrained("Chhabi/mt5-small-finetuned-Nepali-Health-50k-2")
  
  # Load the tokenizer for generating new output
  tokenizer = AutoTokenizer.from_pretrained("Chhabi/mt5-small-finetuned-Nepali-Health-50k-2",use_fast=True)


    
  query = "म धेरै थकित महसुस गर्छु र मेरो नाक बगिरहेको छ। साथै, मलाई घाँटी दुखेको छ र अलि टाउको दुखेको छ। मलाई के भइरहेको छ?"
  input_text = f"answer: {query}"
  inputs = tokenizer(input_text,return_tensors='pt',max_length=256,truncation=True).to("cuda")
  print(inputs)
  generated_text = model.generate(**inputs,max_length=512,min_length=256,length_penalty=3.0,num_beams=10,top_p=0.95,top_k=100,do_sample=True,temperature=0.7,num_return_sequences=3,no_repeat_ngram_size=4)
  print(generated_text)
  # generated_text
  generated_response = tokenizer.batch_decode(generated_text,skip_special_tokens=True)[0]
  tokens = generated_response.split(" ")
  filtered_tokens = [token for token in tokens if not token.startswith("<extra_id_")]
  print(' '.join(filtered_tokens))

```
## Evaluation
### BLEU score:
![Screenshot from 2024-03-04 22-47-37](https://github.com/Chhabii/FinetuneMT5-Nepali-Health-Chat/assets/60286478/56f97b14-77ad-45d0-a016-e6ad08be5a32)

![image](https://github.com/Chhabii/FinetuneMT5-Nepali-Health-Chat/assets/60286478/dfa3f3e2-7297-44c7-80dc-0c28a5816324)


### Inference from finetuned model:

![Screenshot from 2024-03-05 08-23-24](https://github.com/Chhabii/FinetuneMT5-Nepali-Health-Chat/assets/60286478/50f1ee5a-8139-4cd2-885a-fa44f3e88388)

![image](https://github.com/Chhabii/FinetuneMT5-Nepali-Health-Chat/assets/60286478/a91d5ace-1f41-4185-a8e0-ab5cf8af6b7c)

![image](https://github.com/Chhabii/FinetuneMT5-Nepali-Health-Chat/assets/60286478/88adc0d0-a9cf-43d7-a8af-dd5a5e103461)

## FineTune 
### How to finetune your model for your custom datasets? 

[Dataset: NepaliHealthQA-20k-Good-Quality Dataset](https://huggingface.co/datasets/NepaliAI/Nepali-Health-Fact)

[Dataset: NepaliHealthQA-55k-low Quality Dataset](https://huggingface.co/datasets/NepaliAI/Nepali-HealthChat)

[Code: QuestionAnsweringFinetuning](https://github.com/Chhabii/FinetuneMT5-Nepali-Health-Chat/blob/master/mt-5-finetuned.ipynb)
