from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import ast
import argparse
import json
import re
import pandas as pd

def format_dolly(sample):
    sample['instruction']=sample['instruction'].replace("'","")
    format_mess = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": f"What is the aspect based sentiment of the following customer content, answer in format [aspect term, aspect category, opinion term, sentiment polarity]? {sample['instruction']}" }
    ]
    return format_mess

def get_res(model,tokenizer,messages):
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True)

def get_llama3_res(model,tokenizer,dataset):
    for i in tqdm(range(len(dataset))):
        input_data = dataset[i]
        res = get_res(model,tokenizer,format_dolly(input_data))
        dataset[i]['pred_result'] = res

def llama3_predict(model,tokenizer,dataset):
    get_llama3_res(model,tokenizer,dataset)
    return dataset

def get_llama(x1):
    x1 = x1.split('### Answer\n')[-1]
    list_obj = ast.literal_eval(x1.replace("'s "," ").replace("'ll "," ").replace("'t "," ")) 
    list_res = [(i[1],i[3]) for i in list_obj]
    return list_res

def get_label_prediction(x):
    x1 = x['response']
    res_label = [(i[1],i[3])for i in x1]
    res_pred = get_llama(x['pred_result'])
    return res_label,res_pred

def compute_f1_scores(pred_pt, gold_pt):
    """
    Function to compute F1 scores with pred and gold pairs/triplets
    The input needs to be already processed
    """
    # number of true postive, gold standard, predicted aspect terms
    res = {}

    n_tp, n_gold, n_pred = 0, 0, 0

    for i in range(len(pred_pt)):
        gold_pt[i] = list(set(gold_pt[i]))
        pred_pt[i] = list(set(pred_pt[i]))

        n_gold += len(gold_pt[i])
        n_pred += len(pred_pt[i])

        for t in pred_pt[i]:
            if t in gold_pt[i]:
                n_tp += 1

    precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
    recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    scores = {'total precision': precision, 'total recall': recall, 'total f1': f1}

    return scores


def get_subset(original_list, element):
    return [sublist for sublist in original_list if element in sublist[0]]


def calc_tag(element, all_predictions, all_labels):
    pred_pt_sub = [get_subset(original_list, element) for original_list in all_predictions]
    gold_pt_sub = [get_subset(original_list, element) for original_list in all_labels]
    score = compute_f1_scores(pred_pt_sub, gold_pt_sub)
    return score

def get_tags(x):
    y = [i[0] for i in x]
    return y 

def get_tag_list(all_labels):
    merged = [get_tags(x) for x in all_labels]
    flat_list = [item for sublist in merged for item in sublist]
    unique_elements = list(set(flat_list))
    return unique_elements

def get_eval_res(tags,all_predictions, all_labels):
    precisons = []
    recall = []
    f1 = []
    for i in tags:
        score = calc_tag(i, all_predictions, all_labels)
        precisons.append(score['total precision'])
        recall.append(score['total recall'])
        f1.append(score['total f1'])

    res = pd.DataFrame({'tag':tags, 'precision':precisons, 'recall':recall,'f1':f1})
    return res

def eval_function(dataset):
    all_labels = []
    all_predictions = []
    for i in tqdm(range(len(dataset))):
        x = dataset[i]
        res_label, res_pred = get_label_prediction(x) 
        all_labels.append(res_label)
        all_predictions.append(res_pred)
        
    print("\nResults of raw output, only tag category & sentiment")
    raw_scores_2 = compute_f1_scores(all_predictions, all_labels)
    print (raw_scores_2)
    
    tags = get_tag_list(all_labels)
    res = get_eval_res(tags,all_predictions, all_labels)
    res.sort_values(by=['recall'],ascending=False,inplace=True)
    return res

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_path_or_name",
        type=str,
        help="Model id to use for inference.",
    )
    parser.add_argument(
        "--dataset_path", type=str, default="lm_dataset", help="Path to inference dataset."
    )
    args, _ = parser.parse_known_args()
    return args

def main():
    args = parse_arge()
    
    with open(args.dataset_path,'r') as file:
        dataset=json.load(file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path_or_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_path_or_name, torch_dtype=torch.float16,device_map="auto")
    
    dataset=llama3_predict(model,tokenizer,dataset)
    metric=eval_function(dataset)
    print(metric)
    metric.to_csv('eval_metric.csv',index=False)


if __name__ == "__main__":
    main()
