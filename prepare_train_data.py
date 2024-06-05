import pandas as pd
import os
import shutil
import json

def write_to_dict(df,file_path):
    '''
    write back to txt
    '''
    res = []
    # output txt file
    df = df.reset_index()
   
    for i in range(len(df)):
        data = {'instruction': df.loc[i, 'review_content'].strip(),
                'context': '',
                'response': df.loc[i, 'label'],
                 'category': ''}
        data=format_dolly(data)
        res.append(data)
    with open(file_path,'w') as file:
        json.dump(res,file)
    return res


def mkdir_rm(folder):
    '''
    make directory if not exists
    '''
    os.makedirs(folder,exist_ok=True)
    print("<< path valid!")

def get_label(row):
    #try:
     #   if pd.isna(row['Remark']):
      #      return (str(row['aspect_term']),str(row['aspect_category']),str(row['opinion_term']),str(row['sentiment_polarity']))
       # elif row['Remark']=='改':
        #    return (str(row['sentiment_polarity_1']),str(row['aspect_category_1']),str(row['opinion_term_1']),str(row['sentiment_polarity_1']))
        #elif row['Remark']=='修改':
         #   return (str(row['sentiment_polarity_1']),str(row['aspect_category_1']),str(row['opinion_term_1']),str(row['sentiment_polarity_1']))
    #except:
    return (str(row['aspect_term']),str(row['aspect_category']),str(row['opinion_term']),str(row['sentiment_polarity']))

def get_data(type,excelname,sheetname=None):
    if type == 'direct':
        jsonObj = pd.read_excel(excelname)
    elif type == 'subsheet':
        jsonObj = pd.read_excel(excelname, sheet_name=sheetname)
    #revise content
    jsonObj['review_content'] = jsonObj['review_content'].fillna(method='ffill')
    jsonObj = jsonObj[-jsonObj['review_content'].isnull()]
    jsonObj['review_content'] = jsonObj['review_content'].apply(lambda x: x.replace('\n', ' '))

    # capitalize
    jsonObj['aspect_category'] = jsonObj['aspect_category'].map(lambda x: str(x).capitalize())
    
    #map sentiment
    if 'sentiment_polarity_1' in jsonObj.columns:
        jsonObj['sentiment_polarity_1'] = jsonObj['sentiment_polarity_1'].map(lambda x: 'Positive' if str(x).capitalize()=='Neutral' else x)
    jsonObj['sentiment_polarity'] = jsonObj['sentiment_polarity'].map(lambda x: 'Positive' if str(x).capitalize()=='Neutral' else x)

    # generate label
    jsonObj['label'] = jsonObj.apply(lambda row: get_label(row), axis=1)
    
    #print ('jsonObj: ', jsonObj.head())

    jsonObj = jsonObj[['review_content', 'label']]
    jsonObj = jsonObj[-jsonObj['label'].isnull()]

    # agg label
    jsonObj_grouped = jsonObj.groupby('review_content')['label'].apply(list).reset_index()
    return jsonObj_grouped


def preprocess_data(df,output_path, save_path, over_sample=True):
    # remove & remake the output folder
    mkdir_rm(output_path)

    
    save_path=os.path.join(output_path, save_path)
    res = write_to_dict(df,save_path)
    print("<<<finish data preparing!")
    return res

def format_dolly(sample):
    instruction = f"What is the aspect based sentiment of the following customer content, answer in format [aspect term , aspect category , opinion term , sentiment polarity]? {sample['instruction']}"
    context = f"{sample['context']}" if len(sample["context"]) > 0 else None
    response = f"{[list(i) for i in sample['response']]}"
    prompt={}
    prompt['instruction']=instruction
    prompt['context']=""
    prompt['response']=response
    prompt['history']=[]
    
    # join all the parts together
    # prompt = "\n\n".join([i for i in [instruction, context, response] if i is not None])
    return prompt

def main():
    # get data
    
    dataset = get_data('subsheet','../../2600条评论汇总-20240103.xlsx','训练集')
    output_path = 'data_huabao'
    save_path='data_huabao.json'

    dataset = preprocess_data(dataset,output_path,save_path, over_sample=False)


if __name__ == "__main__":
    main()
