import difflib
import json
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import re
from dask import delayed, compute
import asyncio
import textwrap

from flask import Flask, request,jsonify
app = Flask(__name__)

# app = FastAPI()

# class Item(BaseModel):
#     input_txt: list

class TextCorrectionClass:
    def __init__(self,  model_path='/app/models'):
    # def __init__(self, model_path=r'/home/compuapps/py/Grammer_API/T5'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.t5_tokenizer = T5Tokenizer.from_pretrained(model_path,legacy=False)
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_path).to(self.device)
        self.default_width = 125
    def correct_grammar(self, input_text, num_return_sequences):
        batch = self.t5_tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(self.device)
        translated = self.t5_model.generate(**batch, max_length=64, num_beams=4, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.t5_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def flatten(self, l):
        flat_list = []
        for sublist in l:
            for item in sublist:
                flat_list.append(item)
        return flat_list
    def multi_flatten(self,text_blocks):
        flattened_list = []
        for block in text_blocks:
            flattened_text = '' 
            for inner_block in block:
                for item in inner_block:
                    flattened_text += item + ' '
            flattened_list.append(flattened_text.strip())
            return flattened_list
        
    def correct_text_async(self, input_text):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        inference_results = loop.run_until_complete(self.main(input_text))
        final_correct = self.flatten(inference_results)
        final_correct_single_list = self.flatten(final_correct)
        loop.close()
        return final_correct_single_list
    
    def process_text_async(self,para):
        corrected =  self.correct_text_async(para)
        incorrect =  self.process_incorrect_sentence(para)
        final_correct = self.paragraph_handling_method(corrected, incorrect)
        return final_correct
    def process_all_paragraphs(self,paras):
        processed_paras_delayed = [delayed(self.process_text_async)(para) for para in paras]
        processed_paras = compute(*processed_paras_delayed,scheduler="threads")
        return processed_paras
    def is_single_word(self,sentence):
        words = sentence.split()
        return len(words) == 1
    def divide_into_small_sentences(self, big_sentence, width=135):
        small_sentences = textwrap.wrap(big_sentence, width)
        small_sentences_with_full_stops = [sentence + '.' if not sentence.endswith('.') else sentence for sentence in small_sentences]
        return small_sentences_with_full_stops
    

    def adjust_lists(self, list1, list2):
        result = []
        modified_sentences = [sentence.rstrip('\r') for sentence in list1]
        
        for sent1, sent2 in zip(modified_sentences, list2):
            # print('--------------------adjust_lists incorrect------------',sent1)
            # print('--------------------adjust_lists correct--------------',sent2)

           
            if sent1.endswith(".") and not sent2.endswith("."):
                sent2 += "."
            elif not sent1.endswith(".") and sent2.endswith("."):
                sent2 = sent2.rstrip(".")
            result.append(sent2)
        return result
     
    def process_incorrect_sentence(self,para):
        split_list = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', para)
        
        corrected_sentences = []  # Initialize an empty list to store corrected sentences
        
        for input_sentence in split_list:
            if len(input_sentence) > 225:
                incorrect_sentences = textwrap.wrap(input_sentence, 135)
                corrected_sentences.extend(incorrect_sentences)  # Add wrapped sentences to the list
            else:
                corrected_sentences.append(input_sentence)  # Add the original sentence to the list
        
        return corrected_sentences  # Return the list of corrected sentences

    def process_correct_sentence(self,input_sentence):
        split_list =  re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', input_sentence)
        to_correct = []
        for input_sentence in split_list:
            if len(input_sentence) > 225:
                small_sentences = self.divide_into_small_sentences(input_sentence)
                for i in small_sentences:
                    to_correct.append(i)
            else:
                to_correct.append(input_sentence)
        return to_correct

    async def async_inference(self, input_text):
        decoded_output =self.correct_grammar(input_text,1) 
        return decoded_output

    async def main(self, sentence_list):
        results=[]
        correct=[]
        
        if self.is_single_word(sentence_list):
            correct.append(sentence_list)
        else:
            correct.append(self.process_correct_sentence(sentence_list))
        final_correct = self.flatten(correct)
        tasks = [self.async_inference(input_text) for input_text in final_correct]
        completed_tasks=await asyncio.gather(*tasks)
        results.append(completed_tasks)

        return results
    def paraphrase_sentence(self,input_sentence):
        # Paraphrase the sentence (You can add your paraphrasing logic here)
        paraphrased_text = input_sentence  # Replace this with your paraphrasing logic
        paraphrased_text = paraphrased_text.strip('[]').strip("'")
        return paraphrased_text
    def paragraph_handling_method(self, final_correct, final_incorrect):
   
        # print('*********************************',final_correct)
        adjusted_sentence2 = self.adjust_lists(final_incorrect, final_correct)
        # print('***************adjusted_sentence2(((((((((())))))))))',adjusted_sentence2)
        correct = [self.score_based_correct(i, j) for i, j in zip(final_incorrect, adjusted_sentence2)]
        # print('***************correct(((((((((())))))))))',adjusted_sentence2)
        highlighted_text = [self.show_highlights(i, j) for i, j in zip(final_incorrect, correct)]
        # print('***************highlighted_text(((((((((())))))))))',highlighted_text)
        corrected_final_text=self.space_equal_list(final_incorrect,highlighted_text)
        # print('***************corrected_final_text(((((((((())))))))))',corrected_final_text)
        corrected_final_symbol_list=[self.check_symbols(final_incorrect,corrected_final_text) for final_incorrect,corrected_final_text in zip(final_incorrect,corrected_final_text) ]
        corrected_final_stop_word=[self.stop_words_checking(final_incorrect,corrected_final_text) for final_incorrect,corrected_final_text in zip(final_incorrect,corrected_final_symbol_list) ]

        # print('***************symbol   list(((((((((())))))))))',corrected_final_symbol_list)
        paraphrased_sentences =  [self.paraphrase_sentence(sentence) for sentence in corrected_final_stop_word]
        # print('***************paras(((((((((())))))))))',paraphrased_sentences)
        result = ' '.join(paraphrased_sentences)
        print('***************result(((((((((())))))))))',result)
        return result
    
    def combine_sentences(self, sentence1, sentence2):
        tokens1 = sentence1.split()
        tokens2 = sentence2.split()

        # Find the index where sentence 2 starts matching sentence 1
        matching_index = 0
        for i in range(min(len(tokens1), len(tokens2))):
            if tokens1[i] == tokens2[i]:
                matching_index = i + 1
            else:
                break

        # Combine sentences while avoiding duplication
        combined_tokens = tokens1 + tokens2[matching_index:]
        combined_sentence = ' '.join(combined_tokens)

        return combined_sentence        

    def make_sentences_almost_same(self, sentence1, sentence2):
        words1 = sentence1.split()
        words2 = sentence2.split()
        
        if len(words1) == 0 or len(words2) == 0:
            return sentence2  # Return sentence2 if either sentence is empty
        
        if words1[0] != words2[0]:
            result = self.combine_sentences(sentence1, sentence2)
        else:
            return sentence2
        
        return result   
    def word_differences(self, string1, string2):
        words1 = string1.split()
        words2 = string2.split()
        differ = difflib.ndiff(words1, words2)
        differences = [diff[2:] for diff in differ if diff.startswith('+ ')]
        return differences
    def has_proper_words(self, element):
        words = element.split()
        proper_words = [word for word in words if word.isalpha() and len(word) > 1 or (word == 'a')]
        return len(proper_words) > 0

    def score_based_correct(self, i, j):
        correct_list=''
        final_correct=self.make_sentences_almost_same(i,j)

        differences = self.word_differences(i, final_correct)

        if not differences:
            correct_list=i
        correct_count=0
        incorrect_count=0
        for element in differences:
            if self.has_proper_words(element):
                correct_count+=1 
            else:
                incorrect_count+=1

        if correct_count<=2 and incorrect_count==0:

            correct_list=final_correct
        else:
            correct_list=i 
        return correct_list 
    def get_last_word(self,input_string):
        words = input_string.split()
        if words:
            return words[-1]
        else:
            return None
    def char_comparison(self, a, b):
        # Implementation of charComparison function
        m = difflib.SequenceMatcher(a=a, b=b)
        resultString = ''
        #Ex. a = were, b = was
        for tag, i1, i2, j1, j2 in m.get_opcodes():
            if tag == 'replace':
                #print 1 4 1 3
                #<del> + index a[1:4] + </del>= > return <del> 'ere' </del>
                #*DEL
                resultString += f'<span style="color:rgb(255,99,71);text-decoration: line-through;background-color:  rgb(255,203,164);max-height:25px;padding:0px 3px;-webkit-box-align:center;margin:0px;">{a[i1:i2]}</span>'
                #<ins> + index b[1:3] => return <ins> <ins>'as'</ins>
                resultString += f'<span style="background-color:rgba(73, 149, 87, 0.1);border:1px solid rgb(73, 149, 87);border-radius:2px;max-height:25px;padding:0px 3px;-webkit-box-align:center;margin:0px;color:rgb(73, 149, 87)">{b[j1:j2]}</span>'
            if tag == 'delete':
                resultString += f'<span style="color:rgb(255,99,71);text-decoration: line-through;background-color:  rgb(255,203,164);max-height:25px;padding:0px 3px;-webkit-box-align:center;margin:0px;">{a[i1:i2]}</span>'
            if tag == 'insert':
                resultString += f'<span style="background-color:rgba(73, 149, 87, 0.1);border:1px solid rgb(73, 149, 87);border-radius:2px;max-height:25px;padding:0px 3px;-webkit-box-align:center;margin:0px;color:rgb(73, 149, 87)">{b[j1:j2]}</span>'
            if tag == 'equal':
                #print 0 1 0 1 
                #The first index with end index a[0:1] = return 'w'
                resultString += f'<span>{a[i1:i2]}</span>'
        return resultString   

    def space_equal_list(self,sentences1, sentences2):
        adjusted_sentences2 = []
        for sent1, sent2 in zip(sentences1, sentences2):
            # Use regex to find all spaces in sent1, including double spaces
            spaces_sent1 = re.findall(r'\s+', sent1)

            # Initialize an iterator for the spaces in sent1
            spaces_iterator = iter(spaces_sent1)

            # Replace spaces in sent2 with spaces from sent1, including double spaces
            sent2_adjusted = ''
            for char in sent2:
                if char.isspace():
                    sent2_adjusted += next(spaces_iterator, ' ')
                else:
                    sent2_adjusted += char
            
            adjusted_sentences2.append(sent2_adjusted)
        
        return adjusted_sentences2    
    

    def ensure_symbols_in_sent2(self,sent1, sent2):
    # Extract symbols and their positions from sent1
        symbols_and_positions = [(match.group(), match.start()) for match in re.finditer(r'[^\w\s]+', sent1)]
        
        # Add missing symbols to sent2 at the specified positions
        for symbol, position in symbols_and_positions:
            if symbol not in sent2:
                sent2 = sent2[:position] + symbol + ' ' + sent2[position:]

        return sent2
    def check_symbols(self,sent1,sent2):
    # Remove spaces and letters from both sentences
        clean_sent1 = ''.join(filter(lambda x: not x.isalpha() and not x.isspace(), sent1))
        clean_sent2 = ''.join(filter(lambda x: not x.isalpha() and not x.isspace(), sent2))

        if clean_sent1 == clean_sent2:
            print("They have the same symbols. It should come.")
            return sent2
        else:
            print("They do not have the same symbols. It should go.")
            return self.ensure_symbols_in_sent2(sent1, sent2)

    def stop_words_checking(self,str1, str2):
        # Tokenize the strings into words (split by spaces and remove punctuation)
        common_english_words  = ['down', 'does', 'can', 'now', 'am', 'have', 'those', 'so', 'she', 'himself', 'until', 'no', 'for', 'there', 'through', 'doing', "shouldn't", 'just', 'which', 'few', 'shan', 'each', 'aren', 'but', 'most', "didn't", 'they', 'myself', 'themselves', "she's", "don't", "hadn't", 'the', "isn't", "should've", 'on', 'you', 'being', 'some', 'it', "hasn't", 'are', 'why', 'all', 're', "mightn't", 'while', "it's", 'having', 'had', 'itself', 'mustn', 'yours', 'between', 'were', 'his', 'this', 'nor', 'out', 'an', 'of', 'further', 'its', 'once', 'me', 'don', 'such', 'ma', 'if', 'before', 'm', 'up', "wouldn't", "mustn't", 'ours', 'shouldn', 'we', 'hasn', 'was', 'who', 'over', 'how', 's', 'not', 'very', 'off', 'yourself', 'them', 'only', "couldn't", 'ain', 'mightn', 'needn', 'our', 'too', 'that', 't', "you'd", 'what', 'been', 'y', 'be', 'again', 'their', "doesn't", 'weren', 'him', 'he', "weren't", 'ourselves', 'your', 'or', 'whom', 'with', 've', 'is', 'did', 'o', 'hadn', 'below', 'yourselves', "wasn't", 'above', 'i', 'hers', 'any', 'here', 'at', 'about', 'and', 'against', "won't", 'as', 'couldn', 'into', 'during', 'should', 'than', 'own', "haven't", 'her', 'these', 'other', 'to', 'd', 'will', 'both', 'my', 'doesn', 'more', 'isn', "you'll", 'by', 'then', 'herself', 'a', 'where', "aren't", 'has', 'theirs', 'from', 'under', 'do', "you've", 'after', 'same', "shan't", "that'll", 'won', 'because', "you're", 'll', 'when', 'didn', 'haven', 'wouldn', 'wasn', "needn't", 'in']
        
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        # Find the unique set of words between the two strings
        unique_words = words1.symmetric_difference(words2)
        
        unique_words1 = words1 - words2
        unique_words2 = words2 - words1
        print("Unique words in strings:", unique_words)
        grammar_word_count = 0
        non_grammar_word_count = 0
    
        for word in unique_words:
            if word in common_english_words:
        #         grammar_words2.append(word)
                grammar_word_count += 1
            else:
                non_grammar_word_count += 1
        #         non_grammar_words2.append(word)
        
    #     if word_in_unique_words1_with_s:
    #         result = string2
    #     elif word_in_unique_words2_with_s:
    #         result = string2
        print("Grammar words in string:", grammar_word_count)
        print("Non-grammar words in string:", non_grammar_word_count)
        word_in_unique_words1_with_s = next((word2 for word2 in unique_words2 if any(word1 == word2 + 's' for word1 in unique_words1)), None)
        print('*******************************',word_in_unique_words1_with_s)
        word_in_unique_words2_with_s = next((word1 for word1 in unique_words1 if any(word2 == word1 + 's' for word2 in unique_words2)), None)
        print('*******************************',word_in_unique_words2_with_s)
        if word_in_unique_words1_with_s:
            result = str2
        elif word_in_unique_words2_with_s:
            result = str2
        elif non_grammar_word_count>0:
            print('------')
            result = str1
        elif len(unique_words)==0:
            print('&&&&&&&&&&&&&&&&&&&&&&&')
            result= str1
        else:
            print('--***********----')
            result = str2
        return result


    def show_highlights(self, a, b):
        print('********************incorrect************',a)
        print('********************correct**************',b)
        # Implementation of show_highlights function
        correct_list=''
        new_str2=''
        if self.get_last_word(a)==self.get_last_word(b):
            correct_list=b
        else:
            new_str2 = self.remove_extra_word_and_match_endings(a, b)
            correct_list =new_str2   
        return correct_list
    def remove_extra_word_and_match_endings(self, str1, str2):
        words1 = str1.split()
        words2 = str2.split()
        if len(words2) > len(words1):
            words2 = words2[:len(words1)]
        return ' '.join(words2)  
    
    def charComparison(self,a,b):
        m = difflib.SequenceMatcher(a=a, b=b)
        retrun_dict = {'insert':[],"delete":[]}
        # for tag, i1, i2, j1, j2 in m.get_opcodes():
        for tag, i1, i2, j1, j2 in m.get_opcodes():
            if tag == 'replace':
                retrun_dict['insert'].append([j1,j2,b[j1:j2],a[i1:i2]]) #* start, end, insert_txt,old_txt
            if tag == 'delete':
                retrun_dict['delete'].append([i1,i2,"",a[i1:i2]])
            if tag == 'insert':
                retrun_dict['insert'].append([j1,j2,b[j1:j2],a[i1:i2]])

        return retrun_dict
    
    def get_compare(self,original,edited):
        d = difflib.Differ()
        diff = d.compare(original.split(), edited.split())
        index = 0
        return_dict = {"insert":[],"delete":[]}
        diff = [value for value in diff if not value.startswith('?')]
        skip_index_ls = []
        for idx,i in enumerate(diff):            
            if index == 0:
                length = i[2:]
            else:
                length = i[1:]

            if not i.startswith('-') and not i.startswith('+') and not i.startswith('?'):        
                index += len(length)

            elif i.startswith('-'):
                nxt_idx = idx+1
                txt_ = i[1:]
                if nxt_idx < len(diff) and diff[nxt_idx].startswith('+') and nxt_idx not in skip_index_ls:
                    insert_txt = diff[nxt_idx]
                    chk_txt = insert_txt[2:]

                    insert_txt = insert_txt[1:]
                    start = index
                    end = index+len(txt_)
                    old_txt = txt_
                    if i[2:].__contains__(chk_txt) or chk_txt.__contains__(i[2:]):
                        res = self.charComparison(txt_,insert_txt)
                        if len(res['insert']) != 0:
                            data = res['insert'][0]
                            start = start + data[0]
                            end = index +len(txt_)
                            insert_txt = data[2]
                            old_txt = data[3]
                        elif len(res['delete']) != 0:
                            data = res['delete'][0]
                            start = start + data[0]
                            end = start + len(data[3])
                            insert_txt = data[2]
                            old_txt = data[3]

                    return_dict['insert'].append([start,end,insert_txt,old_txt])
                    skip_index_ls.append(nxt_idx)
                elif i.startswith('-'):
                    start = index
                    end = index + len(txt_)
                    return_dict['insert'].append([start,end,"",txt_])
                
                index += len(i[1:])
                print(f"Deletion at index {index}: {i[1:]}")
            elif i.startswith('+') and idx not in skip_index_ls:
                start = index
                insert_txt = i[1:]
                txt_ = original[index:index]
                end = index
                return_dict['insert'].append([start,end,insert_txt,txt_])
                # index += 1

        return_dict['insert'] = return_dict['insert'][::-1]
        return_dict['delete'] = return_dict['delete'][::-1]
        return return_dict
    

    def get_idx(self,org,corr):   
        # sleep(1) 
        a = org.lower()
        b = corr.lower()
        s = difflib.SequenceMatcher(a=a, b=b)
        return_dict = {"insert":[],"delete":[]}
        for tag, i1, i2, j1, j2 in reversed(s.get_opcodes()):
            del_txt = a[i1:i2]
            ins_txt = b[j1:j2]
            
            if tag == 'delete':
                if del_txt != ascii(del_txt).strip("'"):
                    a_ = 0
                else:
                    return_dict['delete'].append([i1,i2,a[i1:i2]])
                
            elif tag == 'insert' or (tag == "replace" and len(del_txt.split(' ')) == 1):
                check_del_txt = b[j1:j2]
                if ins_txt != ascii(ins_txt).strip("'"):
                    b_ = 0
                else: 
                    return_dict['insert'].append([i1,i2,b[j1:j2],a[i1:i2]]) #* start_Range, end_Range, Insert_Text, old_Text       
            # elif tag == 'replace':
            #     a1 = a[i1:i2]
            #     b1 = b[j1:j2]
            #     re = self.charComparison(a1,b1)
            #     for idx,ins in enumerate(re['insert']):
            #         st = re['insert'][idx][0]
            #         ed = re['insert'][idx][1]
            #         txt = re['insert'][idx][2]
            #         diff = ed - st
            #         if st == 0:
            #             return_dict['insert'].append([i1,i1+diff,txt,a[i1:i1+diff]])
            #         else:
            #             start = i1+st
            #             end = start + diff
            #             return_dict['insert'].append([start,end,txt,a[start:end]])

            #     for idx,del_ in enumerate(re['delete']):
            #         st = re['delete'][idx][0]
            #         ed = re['delete'][idx][1]
            #         txt = re['delete'][idx][2]
            #         diff = ed - st
            #         if st == 0:
            #             return_dict['delete'].append([i1,i1+diff,txt,a[i1:i1+diff]])
            #         else:
            #             start = i1+st
            #             end = start + diff
            #             return_dict['delete'].append([start,end,txt,a[start:end]])

        return_dict['insert'] = return_dict['insert'][::-1]
        return_dict['delete'] = return_dict['delete'][::-1]
                                
        return return_dict


@app.route("/test",methods=['GET'])
def read_root():
    return {"Hello": "World"}

@app.route("/process_txt",methods=['GET','POST'])
# async def process_grammer(item:Item):
def process_grammer():
    return_ls = []
    try:  
        start_time = time.time()
        print("coming inside this")
        req_data = json.loads(request.data) 
        input_para = req_data['input_txt']
        text_correction_obj = TextCorrectionClass()  
        processed_paras = text_correction_obj.process_all_paragraphs(input_para) 
        for incrrt_txt,crrt_txt in zip(input_para,processed_paras): 
            response = {"status":"","org_txt":"","corr_txt":"","Err":False,"insert":"","delete":""}           
            response['corr_txt'] = crrt_txt
            response['org_txt'] = incrrt_txt
            # res = text_correction_obj.get_idx(incrrt_txt,crrt_txt)
            res = text_correction_obj.get_compare(incrrt_txt,crrt_txt)
            response['insert'] = res['insert']
            response['delete'] = res['delete']
            return_ls.append(response)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
    
    except Exception as err:
        response = {"status":"","org_txt":"","corr_txt":"","Err":False,"insert":"","delete":""}
        response['status'] = str(err)
        response['Err'] = True
        response['insert'] = []
        response['delete'] = []
        return_ls.append(response)
        # print(err)
    return return_ls

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=5011)
