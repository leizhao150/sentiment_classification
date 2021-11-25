## 情感分类 (六分类)
### 1. 模型介绍
使用的模型包括BiLSTM, BiLSTM-Att, BERT和GPT, 同时使用支持向量机(SVM)作为基线模型。

### 2. 数据集介绍
实验的数据来源于Kaggle的一个文本情感分类数据集，实验的目标预测一段给定文本的情感倾向。 

训练集：16000  
测试集：2000  
验证集：2000  
情感标签：['anger', 'surprise', 'joy', 'sadness', 'love', 'fear'] 

link: <a href="https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset">https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset</a>

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
          <td align='center'>86.019</td> 
          <td align='center'>86.458</td>    
          <td align='center'>86.083</td>
      </tr>
        <tr>
          <td align='center'>GPT</td>
          <td align='center'>87.111</td> 
          <td align='center'>87.132</td>    
          <td align='center'>87.077</td>
      </tr>
</table>
