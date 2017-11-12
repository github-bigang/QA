import os
import sys
import codecs
sys.path.append('src/ir/')

class preprocess:
    def genPosNeg(self, qfile, afile, outfile, test=False):
        import random
        qtext = self.readFile(qfile)
        qtext_set = {}
        for qid in qtext:
            qtext_set[qid] = set(qtext[qid].split(' '))
        atext = self.readFile(afile)
        atext_set = {}
        for aid in atext:
            atext_set[aid] = set(atext[aid].split(' '))
        from queryIndex_tfidf import QueryIndex
        q=QueryIndex('data/small/question-index.txt')
        with open(outfile, 'w') as out:
            for qid in qtext:
#                 if not test or random.uniform(0,1) < 0.5:
#                     for aid in atext:
#                         if qid == aid:
#                             out.write('1')
#                             out.write('\t' + qid + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + aid + '\n')
#                         elif len(qtext_set[aid].intersection(qtext_set[qid])) > 2 and len(atext_set[aid].intersection(qtext_set[qid])) > 2: # and random.uniform(0,1) < 0.01:
#                             out.write('2')
#                             out.write('\t' + qid + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + aid + '\n')
                out.write('1\t' + str(qid) + '\t' + qtext[qid] + '\t' + atext[qid] + '\t' + aid + '\n')        
                docs = q.queryIndex(qtext[qid], 100)
                if docs:
                    for d in docs:
                        if q.sigmoid(d[0]) < 0.5: break
                        if qtext[qid] != qtext[d[1]] and (not test or random.uniform(0,1) < 0.2):
                            out.write('2\t' + qid + '\t' + qtext[qid] + '\t' + atext[d[1]] + '\t' + d[1] + '\n') 

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
        que, qOld2New = self.readUniqIDFile('data/small/question-uniq.txt')
        ans, aOld2New = self.readUniqIDFile('data/small/answer-uniq.txt')
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
                    

pp = preprocess()
# pp.genPosNeg(sys.argv[1], sys.argv[2], sys.argv[3], True)
# pp.genUniqID(sys.argv[1], sys.argv[2])
pp.genQAuniq(sys.argv[1], sys.argv[2])
