## 大部分的大模型(LLM)采用左填充的原因

在微调大模型LLM 时,发现目前很多的大模型的tokenizer方式采用的都是left-padding 并是不像bert一样采用right-padding来处理token,据此研究了一下原因.如有不足,或者错误请多多指正.

### 之前 encoder-only 模型的填充一般是right-padding

在使用bert等encoder-only的模型时, 我们往往根据习惯使用右填充(right-padding) 来输入模型 ,例如

```python
## 构造两个不等长的输入句子
input_text = [
    "I want to go to space",
    "I'm going to Greece for my holiday to see the beauty",
]
## 之前 encoder-only 模型一般都是右填充
bert_tokenizer= BertTokenizer.from_pretrained("./link_model/bert-base-uncased/")
bert = BertModel.from_pretrained("./link_model/bert-base-uncased/")
print(bert_tokenizer.padding_side)
## output : right 

## bert tokenzier 之后
tokens = bert_tokenizer(input_text,padding="longest",return_tensors="pt")
print(tokens.input_ids[0])
print(tokens.input_ids[1])
 '''
output: tensor([ 101, 1045, 2215, 2000, 2175, 2000, 2686,  102,    0,    0,    0,    0,
           0,    0,    0])
        tensor([ 101, 1045, 1005, 1049, 2183, 2000, 5483, 2005, 2026, 6209, 2000, 2156,
            1996, 5053,  102])
'''

## 最终输入模型获得输出的embedding

bert(input_ids= tokens["input_ids"], attention_mask=tokens["attention_mask"],return_dict=False)
```

> 我们使用encoder-only模型的主要目的是获取每个词的嵌入表示,并且,cls可能会有额外的作用, 因此选用right-padding 是非常合理的

### only-decoder 的LLM采用left padding的原因

首先我们看一个例子

```python
## 加载模型
llama_tokenzier = LlamaTokenizer.from_pretrained("./link_model/llama2-7b-hf/")
llama_model = LlamaForCausalLM.from_pretrained("./link_model/llama2-7b-hf/",trust_remote_code= True)
llama_tokenzier.pad_token = llama_tokenzier.eos_token
print(llama_tokenzier.padding_side)
## output : right

## 使用llama tokenzier进行token 化
tokens = llama_tokenzier(input_text,padding="longest",return_tensors="pt")
print(tokens.input_ids[0])
print(tokens.input_ids[1])
'''
output:
        tensor([   1,  306,  864,  304,  748,  304, 2913,    2,    2,    2,    2,    2,
           2,    2,    2])
        tensor([    1,   306, 29915, 29885,  2675,   304, 25549,   363,   590,  8753,
             22394,   304,  1074,   278, 15409])
'''
## 使用模型进行输入
output = llama_model.generate(pad_token_id=llama_tokenzier.pad_token_id, **tokens)
res_text = llama_tokenzier.batch_decode(output)
res_text
'''
output:
    ['<s> I want to go to space</s></s></s></s></s></s></s></s><s>nobody wants to go',
     "<s>I'm going to Greece for my holiday to see the beauty of the country and to"]
'''
## 并且我们得到的一个警告
'''
warnings.warn(
A decoder-only architecture is being used, but right-padding was detected! For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
'''
```

我们观察模型输入的内容, 如果不对模型输入的特殊token进行忽略,即采用这种方式来进行输入内容:

```python
res_text = llama_tokenzier.batch_decode(output,skip_special_tokens=True)
```

我们获得的内容中会掺杂很多padding的token在一个完整的句子中如 :

 '<s> I want to go to space</s></s></s></s></s></s></s></s><s>nobody wants to go',

直观上来看,它打断了句子连续的语义.

---

如果采用left-padding看看会发生什么

```python
llama_tokenzier = LlamaTokenizer.from_pretrained("./link_model/llama2-7b-hf/",padding_side = "left")
llama_tokenzier.pad_token = llama_tokenzier.eos_token
print(llama_tokenzier.padding_side)
## output :left
tokens = llama_tokenzier(input_text,padding="longest",return_tensors="pt")
print(tokens.input_ids[0])
print(tokens.input_ids[1])
'''
output:

tensor([   2,    2,    2,    2,    2,    2,    2,    2,    1,  306,  864,  304,
         748,  304, 2913])
tensor([    1,   306, 29915, 29885,  2675,   304, 25549,   363,   590,  8753,
        22394,   304,  1074,   278, 15409])
'''

output = llama_model.generate(pad_token_id=llama_tokenzier.pad_token_id, **tokens)
res_text = llama_tokenzier.batch_decode(output)
res_text
'''
output:

['</s></s></s></s></s></s></s></s><s>I want to go to space. I want to go',
 "<s>I'm going to Greece for my holiday to see the beauty of the country and to"]
'''
```

可以看到 left-padding的输出内容是完整且连续的,中间没有特殊的字符来打断语义,并且 padding的token也分布在模型的两侧,处理特殊的token也会更加方便.
并且,模型在调用generate()时,也不会再有警告了.

## 总结:

1. encoder-only模型主要采用right-padding的原因是,填充右侧的方式很直观,并且我们有时需要获得每个句子的首个token(cls),左侧不对齐不好操作
2. decoder-only模型采用 left-padding的原因是, 模型的输入是对模型输入的延续(模型的输出中会带着输入,并在输入后边补充输出),如果采用right-padding,会导致大量的[pad]token夹在模型的输入和输入之间,不利于处理结果.并且模型的输出句子的语义也被pad打乱了,输入并不直观.此外,decoder-only的模型并不需要cls等开头的token来做额外的处理,right-padding在decoder-only的模型中没有任何优势.
3. decoder-only的模型也可以使用right-padding方式,如果设置 skip_special_tokens=True 也应该会获得正确的输入,但是不建议使用.

## ref:

[Mismatch between logits from generate and forward with an attention mask for most GPT models · Issue #18388 · huggingface/transformers · GitHub](https://github.com/huggingface/transformers/issues/18388#issuecomment-1204369688)
