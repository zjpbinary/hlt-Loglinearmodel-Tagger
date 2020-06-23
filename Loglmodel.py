import math
class LModel:
    def __init__(self):
        self.sentset = []
        self.tagset = []
        self.tag = set()
        self.featureset = set()
        self.feature2index = dict()
        self.tag2index = dict()
        self.weight = []
    def preprocess(self, filename):
        with open(filename, 'r', encoding='utf-8') as f1:
            lines = [line.strip().split('\t') for line in f1.readlines()]
        sent = []
        taglist = []
        for line in lines:
            if line==['']:
                self.sentset.append(sent)
                self.tagset.append(taglist)
                sent = []
                taglist = []
            else:
                sent.append(line[1])
                taglist.append(line[3])
                self.tag.add(line[3])
        i = 0
        for t in self.tag:
            self.tag2index[t] = i
            i += 1
    #根据特征模板提取特征
    def featureexa(self, sent, pos):
        f = []
        word = sent[pos]
        if pos == len(sent)-1:
            next = '$$'
        else:
            next = sent[pos+1]
        if pos == 0:
            pre = '**'
        else:
            pre = sent[pos-1]
        f.append('02:'+word)
        f.append('03:'+pre)
        f.append('04:'+next)
        f.append('05:'+word+pre[-1])
        f.append('06:'+word+next[0])
        f.append('07:'+word[0])
        f.append('08:'+word[-1])
        for i in range(1, len(word)-1):
            f.append('09:'+word[i])
            f.append('10:'+word[0]+word[i])
            f.append('11:'+word[-1]+word[i])
        if len(word)==1:
            f.append('12:'+word+pre[-1]+next[0])
        for i in range(0, len(word)-2):
            if word[i]==word[i+1]:
                f.append('13:'+word[i]+'consecutive')
        if len(word)>=4:
            for k in range(4):
                f.append('14:'+word[:k+1])
                f.append('15:'+word[-k-1::])
        return f
    #featuremodel用于建立部分特征空间
    def featuremodel(self):
        for sent in self.sentset:
            for pos in range(len(sent)):
                partialfeature = self.featureexa(sent, pos)
                for f in partialfeature:
                    self.featureset.add(f)
        index = 0
        for f in self.featureset:
            self.feature2index[f] = index
            index += 1
        self.weight = [0. for _ in range(len(self.tag)*len(self.featureset))]
    def mergefeature(self, f, t):
        s = len(self.featureset)
        offset = self.tag2index[t]*s

        f = [offset+self.feature2index[e] for e in f if e in self.featureset]
        return f
    def dot(self, f):
        score = 0
        for i in f:
            score += self.weight[i]
        return score

    #getmaxtag函数用于预测标签
    def getmaxtag(self, sent, pos):
        maxtag = 'NULL'
        maxnum = -1e10
        tempnum = 0
        f = self.featureexa(sent, pos)
        for t in self.tag:
            curf = self.mergefeature(f, t)
            tempnum = self.dot(curf)
            if tempnum > (maxnum + 1e-10):
                maxnum = tempnum
                maxtag = t
        return maxtag

    def SGDtraining(self, batchsize, num):
        g = [0. for _ in range(len(self.weight))]
        b = 0
        k = 0
        for iterator in range(0, num):
            for i in range(len(self.sentset)):
                for p in range(len(self.sentset[i])):
                    f = self.featureexa(self.sentset[i], p)
                    correcttag = self.tagset[i][p]
                    fcorrect = self.mergefeature(f, correcttag)
                    for elem in fcorrect:
                        g[elem] += 1
                    z = 0
                    for t in self.tag:
                        fother = self.mergefeature(f, t)
                        tempnum = self.dot(fother)
                        z += math.e**tempnum
                    for t in self.tag:
                        fother = self.mergefeature(f, t)
                        tempnum = math.e**self.dot(fother)
                        for elem in fother:
                            g[elem] -= (tempnum/z)
                    b += 1
                    if b==batchsize:
                        k+=1
                        print(k)
                        for j in range(len(g)):
                            self.weight[j] += g[j]
                        b = 0
                        g = [0. for _ in range(len(self.weight))]
            precision = self.evaluate()
            print(precision)

    def evaluate(self):
        count = 0
        right = 0
        for i in range(len(self.sentset)):
            for p in range(len(self.sentset[i])):
                count+=1
                maxtag = self.getmaxtag(self.sentset[i], p)
                if maxtag == self.tagset[i][p]:
                    right+=1
        return right/count
    def predict(self, filename):
        with open(filename, 'r', encoding='utf-8') as f1:
            lines = [line.strip().split('\t') for line in f1.readlines()]
        sent = []
        taglist = []
        preset = []
        pretag = []
        for line in lines:
            if line == ['']:
                preset.append(sent)
                pretag.append(taglist)
                sent = []
                taglist = []
            else:
                sent.append(line[1])
                taglist.append(line[3])
        count = 0
        right = 0
        for i in range(len(preset)):
            for p in range(len(preset[i])):
                count += 1
                maxtag = self.getmaxtag(preset[i], p)
                if maxtag == pretag[i][p]:
                    right += 1
        print('测试集上的精度为：', right/count)


if __name__ == "__main__":
    model = LModel()
    model.preprocess('data/train.conll')
    model.featuremodel()
    model.SGDtraining(50, 3)
    model.predict('data/dev.conll')

