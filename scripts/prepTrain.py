import sys
sys.path.append('src/ir/')

class prepTrain:
    def genPosNeg(self, qfile, afile, qafile, iFile, outfile1, outfile2):
        import random
        qtext = self.readFile(qfile)
        qtext_set = {}
        for qid in qtext:
            qtext_set[qid] = set(qtext[qid].split(' '))
        atext = self.readFile(afile)
        atext_set = {}
        for aid in atext:
            atext_set[aid] = set(atext[aid].split(' '))
        q2a ={}
        with open(qafile, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                if fields[0] not in q2a:
                    q2a[fields[0]] = []
                q2a[fields[0]].append(fields[1])
        from queryIndex_tfidf import QueryIndex
        q=QueryIndex(iFile)
        with open(outfile1, 'w') as out1, open(outfile2, 'w') as out2:
            for qid in qtext:
#                 if not test or random.uniform(0,1) < 0.5:
#                     for aid in atext:
#                         if qid == aid:
#                             out.write('1')
#                             out.write('\t' + qid + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + aid + '\n')
#                         elif len(qtext_set[aid].intersection(qtext_set[qid])) > 2 and len(atext_set[aid].intersection(qtext_set[qid])) > 2: # and random.uniform(0,1) < 0.01:
#                             out.write('2')
#                             out.write('\t' + qid + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + aid + '\n')
                for aid in q2a[qid]:
                    out1.write('1\t' + str(qid) + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + aid + '\n')   
                    out2.write('1\t' + str(qid) + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + aid + '\n')   
                docs = q.queryIndex(qtext[qid], 100)
                if docs:
                    for d in docs:
                        if q.sigmoid(d[0]) < 0.5: break
                        if qtext[qid] == qtext[d[1]]: continue
                        for aid in q2a[d[1]]:
                            out1.write('2\t' + qid + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + d[1] + '\n') 
                        if random.uniform(0,1) < 0.2:
                            for aid in q2a[d[1]]:
                                out2.write('2\t' + qid + '\t' + qtext[qid] + '\t' + atext[aid] + '\t' + d[1] + '\n') 

    def readFile(self, f):
        d = {}
        with open(f, 'r') as data:
            for line in data:
                fields = line.strip().split('\t')
                d[fields[0]] = fields[1]
        return d
    
if __name__=="__main__":
    
    pp = prepTrain()
    pp.genPosNeg('data/small/question-seg.txt', 'data/small/answer-seg.txt',  'data/small/question_answer.txt', 'data/small/question-index.txt', 'data/small/train.txt', 'data/small/test.txt')

