##这里存放模型以及他们的序列流程图，可以以下访问链接，详细查看


1. read_all 文件是简单的cnn，即用三层卷积池化和crelu，四个输出层[可视化](http://note.youdao.com/noteshare?id=88534de3bab985d1a60bb64cd654f458&sub=CB20C232BD974919860F3B05D35A1128)

2. ctc这个模型先用了三层卷积池化，然后是dense层，gru层，merge层，gru层，dense层，最后输出
