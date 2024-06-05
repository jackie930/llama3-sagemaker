import pandas as pd
import os
import json
import argparse
from tqdm import tqdm
import json
import random

def rename_keywords(keywordls):
    key_dict = {'1': 'Primary Keywords', '2':'Attribute Keywords', '3':'Other Keywords', '4': 'Shopee-Specific Infringement Keywords','5': 'Category Keywords','6': 'Tag Keywords','7':'Compatibility Keywords', '8':'Synonyms', '9':'Scene Keywords','10':'Quantity','11':'Model/Type', '12':'Applicable Scope','13':'Product Attributes','14':'Color/Size'}
    res = list(eval(keywordls).values())[0]
    keys = res.keys()
    res = json.dumps(dict(zip([key_dict[i] for i in keys], res.values())))
    return res

def get_keyword_train(input):
    #根据用户输出的标题、描述、属性名列表，输出对应的属性值
    instruction = f"You are a senior cross-border e-commerce product consultant, your task is write keyword based on the title, description provided by the user, answer in format a dictionary of[keyword_type: keyword]. the input title is {input['title']}, input product description is {input['desc']}, what's the keywords? "
    response = f"output result is {rename_keywords(input['输出结果2：关键词'])}"
    prompt={}
    prompt['instruction']=instruction
    prompt['context']=""
    prompt['response']=response
    prompt['history']=[]

    return prompt

def get_att_train(input):
    #根据用户输出的标题、描述、属性名列表，输出对应的属性值
    instruction = f"You are a senior cross-border e-commerce product consultant, your task is extract attribute information based on the title, description, and list of attribute names provided by the user, answer in format a dictionary of[attribute_name: attribute_value]. the input title is {input['title']}, input product description is {input['desc']}, input attrNameList is {input['attrNameList']}, what's the rephrased output? "
    response = f"output result is {input['输出结果1:属性和属性值对应关系']}"
    prompt={}
    prompt['instruction']=instruction
    prompt['context']=""
    prompt['response']=response
    prompt['history']=[]
    return prompt

def main(input_file,output_folder,output_filename):
    # get data
    dataset = pd.read_excel(input_file)

    res = []
    print ("<<< process attributes")
    for i in tqdm(range(len(dataset))):
        input = dataset.iloc[i, :]
        data = get_att_train(input)
        res.append(data)

    print("<<< process keywords")
    for i in tqdm(range(len(dataset))):
        try:
            input = dataset.iloc[i, :]
            data = get_keyword_train(input)
            res.append(data)
        except:
            continue

    print ("<<<output json file!")
    # output json, train/test split
    # shuffle data
    random.shuffle(res)
    total_len = len(res)
    split_idx = int(total_len * 0.9)
    train_json = res[:split_idx]
    test_json = res[split_idx:]

    # Open a file for writing (text mode with UTF-8 encoding)
    with open(os.path.join(output_folder, output_filename), 'w') as file:
        json.dump(train_json, file)

    with open(os.path.join(output_folder, 'test.json'), 'w', encoding='utf-8') as f:
        # Use json.dump() to write the list to the file
        json.dump(test_json, f)  # Optional parameter for indentation
    print('Data written to json')

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--output_filename', default='train.json',type=str)
    args = parser.parse_args()
    main(args.input_file, args.output_folder, args.output_filename)
