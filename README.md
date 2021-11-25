## 情感分类 (六分类)
### 1. 模型介绍
使用的模型包括BiLSTM, BiLSTM-Att, BERT和GPT, 同时使用支持向量机(SVM)作为基线模型。

### 2. 数据集介绍
训练集：16000  
测试集：2000  
验证集：2000  
情感标签：['anger', 'surprise', 'joy', 'sadness', 'love', 'fear']  

### 3. 性能评估
在测试集上，使用P、R、F1值来评估模型的性能, 结果如下：  

<table>
    <tr>
        <th align='center'>模型</th> 
        <th align='center'>P</th> 
        <th align='center'>R</th> 
        <th align='center'>F1</th> 
    </tr>
  <tr>
        <td align='center'>baseline (SVM)</td>
        <td align='center'>78.404</td> 
        <td align='center'>79.617</td> 
        <td align='center'>78.944</td>    
    </tr>
    <tr>
        <td align='center'>BiLSTM</td>
        <td align='center'>87.202</td> 
        <td align='center'>85.488</td> 
        <td align='center'>86.167</td>    
    </tr>
     <tr>
          <td align='center'>BiLSTM-Att</td>
          <td align='center'>87.962</td> 
          <td align='center'>86.524</td>    
          <td align='center'>87.078</td>
      </tr>
    <tr>
          <td align='center'>BERT</td>
          <td align='center'>90.189</td> 
          <td align='center'>90.786</td>    
          <td align='center'>90.424</td>
      </tr>
        <tr>
          <td align='center'>GPT</td>
          <td align='center'>90.735</td> 
          <td align='center'>91.192</td>    
          <td align='center'>90.884</td>
      </tr>
</table>
结论：与其他模型相比，预训练语言模型的分类性能最佳，尤其是GPT模型。在模型中，加入注意力机制可以提升模型的性能。