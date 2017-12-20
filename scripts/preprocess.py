import os
import sys
import codecs
sys.path.append('src/ir/')

class preprocess:
    def readFile(self, f):
        d = {}
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                d[fields[0]] = fields[1]
        return d
    
    def genUniqID(self, f, outf):
        records = self.readFile(f)
        text2new = {}
        new2old = {}
        for d in records:
            if records[d] in text2new:
                new2old[records[d]].append(d)
            else:
                text2new[records[d]] = len(text2new)
                new2old[records[d]] = [d]
        with open(outf, 'w') as out:
            for d in text2new:
                out.write(str(text2new[d]) + '\t' + d + '\t' + ' '.join(new2old[d]) + '\n')
                
    def readUniqIDFile(self, f):
        d = {}
        old2new = {}
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                d[fields[0]] = fields[1]
                for oldid in fields[2].split(' '):
                    old2new[oldid] = fields[0]
        return d, old2new
    
    def genQAuniq(self, f, outf):
        que, qOld2New = self.readUniqIDFile('data/small/question.txt')
        ans, aOld2New = self.readUniqIDFile('data/small/answer.txt')
        uniqQA = {}
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                if qOld2New[fields[0]] not in uniqQA:
                    uniqQA[qOld2New[fields[0]]] = set()
                uniqQA[qOld2New[fields[0]]].add(aOld2New[fields[1]])
        with open(outf, 'w') as data:
            for qid in uniqQA:
                for aid in uniqQA[qid]:
                    data.write(str(qid) + '\t' + str(aid) + '\n')
                    
if __name__=="__main__":
    
    pp = preprocess()
    # pp.genPosNeg(sys.argv[1], sys.argv[2], sys.argv[3], True)
    pp.genUniqID('data/small/raw/question.txt', 'data/small/question.txt')
    pp.genUniqID('data/small/raw/answer.txt', 'data/small/answer.txt')
    pp.genQAuniq('data/small/raw/question_answer.txt', 'data/small/question_answer.txt')
