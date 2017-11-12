from segment import segment
import os
import sys
sys.path.append('src/ir/')

class data:
    def __init__(self, qFile, qSegFile, aFile, aSegFile, qaFile, indexFile):
        self.qFile = qFile
        self.qSegFile = qSegFile
        self.aFile = aFile
        self.aSegFile = aSegFile
        self.qaFile = qaFile
        self.indexFile = indexFile
        self.seg = segment()
        self.all_questions, self.maxQID = self.readFile(qFile)
        self.all_answers, self.maxAID = self.readFile(aFile)
        self.all_qas = self.readQAFile(qaFile)

    def readQAFile(self, f):
        d = {}
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                if fields[0] not in d:
                    d[fields[0]] = []
                d[fields[0]].append(fields[1])
        return d
        
    def readFile(self, f):
        d = {}
        maxID = -1
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                d[fields[0]] = fields[1]
                i = int(fields[0])
                if i > maxID:
                    maxID = i
        return d, maxID

    def writeFile(self, f, i, text):
        with open(f, 'a') as data:
            data.write('3ikids' + str(i) + '\t' + text + '\n')
    
    def inputQA(self, q, a):
        if q not in self.all_questions:
            self.maxQID += 1
            with open(self.qFile, 'a') as data:
                data.write(str(self.maxQID) + '\t' + q + '\n')
            with open(self.qSegFile, 'a') as data:
                data.write(str(self.maxQID) + '\t' + ' '.join(self.seg.seg(q)) + '\n')
            self.all_questions[q] = self.maxQID
        qid = self.all_questions[q]
        if a not in self.all_answers:
            self.maxAID += 1
            with open(self.aFile, 'a') as data:
                data.write(str(self.maxAID) + '\t' + a + '\n')
            with open(self.aSegFile, 'a') as data:
                data.write(str(self.maxAID) + '\t' + ' '.join(self.seg.seg(a)) + '\n')
            self.all_answers[a] = self.maxAID
        aid = self.all_answers[a]
        if qid not in self.all_qas or aid not in self.all_qas[qid]:
            if qid not in self.all_qas:
                self.all_qas[qid] = []
            self.all_qas[qid].append(aid)
            with open(self.qaFile, 'a') as data:
                data.write(str(qid) + '\t' + str(aid) + '\n')

        from createIndex_tfidf import CreateIndex
        c=CreateIndex()
        c.createIndex(self.qSegFile, self.indexFile)

if __name__=="__main__":
    test = data('data/small/question.txt', 'data/small/question-seg.txt', 'data/small/answer.txt', 'data/small/answer-seg.txt',
                'data/small/question_answer.txt', 'data/small/question-index.txt')
    import sys
    test.inputQA(sys.argv[1], sys.argv[2])