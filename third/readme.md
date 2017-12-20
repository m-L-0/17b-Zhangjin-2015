[请访问链接](http://note.youdao.com/noteshare?id=f4b5310520413433f6c53b8af8cd033f&sub=6F1F49888A2043AF85652C658E9304E3)
## **神经网络过程**
```
graph TD
    A[labels.CSV] -->B[Imgs路径]
    D[images] -->B
    A[labels.CSV] -->C[labels]
    B[Imgs路径] -->E[灰度化]
    B[Imgs路径] -->F[二值化]
    B[Imgs路径] -->G[resize 宽50*高40]
    B[Imgs路径] -->H[归一化]
    E -->I[图片打乱顺序]
    C -->I
    F -->I
    G -->I
    H -->I
    I -->J[切片分割4000一组]
    J -->K[转为TFrecord文件]
    K -->L[可视化数据-以长度为组]
    K -->M[训练模型]
    N[模型搭建]-->M
    M -->N
    N --连接-->P[网站衔接]
    Q[用户上传图片] -->P
    P --图片-->N
    N --结果-->P
    P -->O[网页展示]
    
```

- [x] 1. 根据.csv文件读取标签和图片名称
- [ ] 2. 将图片resize->宽50*高40（Image.open().resize((50,40),Image.ANTIALIAS)）
- [ ] 3. 灰度化、二值化、滤波、字符分割、归一化
>>3种灰度化的方法： 
>> - 取红绿蓝中的最大值
>> - 取红绿蓝的平均值
>> - 0.3×红 + 0.59×绿 + 0.11×蓝

>>二值化：
>> - 将灰度图上置为0或255

- [ ] 4. 批次读取数据并打乱数据
- [ ] 5. Img -> tobytes() 用 [a:a+4000] 分割 存储为Tfrecords文件
- [ ] 6. 读取数据并可视化
- [ ] 7. 构建模型
- [ ] 8. 训练模型
- [ ] 9. 网站衔接
- [ ] 10. test
