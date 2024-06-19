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
    x = rename_keywords(input['输出结果2：关键词'])
    if len(json.loads(x))>=3:
        instruction = f"You are a senior cross-border e-commerce product consultant, your task is write keyword based on the title, description provided by the user, answer in format a dictionary of[keyword_type: keyword]. the input title is {input['title']}, input product description is {input['desc']}, what's the keywords? Present the answer in a dictionary format, where the keys are the names inside  {list(json.loads(x).keys())} and the values are lists containing the keywords."
        response = f"output result is {x}"
        prompt={}
        prompt['instruction']=instruction
        prompt['context']=""
        prompt['response']=response
        prompt['history']=[]

        return prompt
    else:
        return {}

def get_att_train(input):
    #根据用户输出的标题、描述、属性名列表，输出对应的属性值
    instruction = f"You are a senior cross-border e-commerce product consultant, your task is extract attribute information based on the title, description, and list of attribute names provided by the user, answer in format a dictionary of[attribute_name: attribute_value]. the input title is {input['title']}, input product description is {input['desc']}, input attrNameList is {input['attrNameList']}, what's the rephrased output? Present the answer in a dictionary format, where the keys are the names of each attributes and the values are lists containing the attributes content."
    response = f"output result is {input['输出结果1:属性和属性值对应关系']}"
    prompt={}
    prompt['instruction']=instruction
    prompt['context']=""
    prompt['response']=response
    prompt['history']=[]
    return prompt


def get_att_train_multi(input):
    prompts = []
    # 根据用户输出的标题、描述、属性名列表，输出对应的属性值
    x = input['输出结果1:属性和属性值对应关系']

    for i in list(json.loads(x).keys()):
        # 根据用户输出的标题、描述、属性名列表，输出对应的属性值
        instruction = f"You are a senior cross-border e-commerce product consultant, your task is extract attribute information based on the title, description, and list of attribute names provided by the user, the input title is {input['title']}, input product description is {input['desc']}, what's the {i}?"
        response = f"{i} is {json.loads(x)[i]}"
        prompt = {}
        prompt['instruction'] = instruction
        prompt['context'] = ""
        prompt['response'] = response
        prompt['history'] = []
        prompts.append(prompt)

    return prompts

def get_keyword_train_multi(input):
    prompts = []
    # 根据用户输出的标题、描述、属性名列表，输出对应的属性值
    x = rename_keywords(input['输出结果2：关键词'])
    for i in list(json.loads(x).keys()):
        instruction = f"You are a senior cross-border e-commerce product consultant, your task is write keyword based on the title, description provided by the user, the input title is {input['title']}, input product description is {input['desc']}, what's the {i}?"
        response = f"{i} is {json.loads(x)[i]}"
        prompt = {}
        prompt['instruction'] = instruction
        prompt['context'] = ""
        prompt['response'] = response
        prompt['history'] = []

        prompts.append(prompt)

    return prompts

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
            if data!={}:
                res.append(data)
            else:
                continue
        except:
            continue

    res2 = []
    ### format keywords to extend training data
    print("<<< process multi keywords")
    for i in tqdm(range(len(dataset))):
        try:
            input = dataset.iloc[i, :]
            data = get_keyword_train_multi(input)
            if data != []:
                res2 = res2+data
            else:
                continue
        except:
            continue

    ### format attributes to extend training data
    print("<<< process multi attributes")
    for i in tqdm(range(len(dataset))):
        try:
            input = dataset.iloc[i, :]
            data = get_att_train_multi(input)
            if data != []:
                res2 = res2 + data
            else:
                continue
        except:
            continue

    ###sample res2 to 30000 size
    res2_sample = random.sample(res2, 30000)
    res = res + res2_sample

    print ("<<<output json file!")
    # output json, train/test split
    # shuffle data
    random.shuffle(res)
    total_len = len(res)
    split_idx = int(total_len * 0.9)
    train_json = res[:split_idx]
    test_json = res[split_idx:]
    print ("<<<< train data size: ", split_idx)

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
