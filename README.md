# QA

1. Copy 3 data files to data/small/raw/: question.txt, answer.txt, question_answer.txt
2. cd root folder of this repository
3. Run 'python scripts/preprocess.py', you will get 3 files with same name at data/small/.
4. Create resources folder, and download segmentation model from and unzip it at resources folder.
5. Install required package, 'pip install pyltp'.
6. Run 'python src/segment.py', you will get 2 files at data/small/question-seg.txt and data/small/answer-seg.txt.
7. Run 'python src/ir/createIndex_tfidf.py', you will get an index file at data/small/question-index.txt'
8. Verify the index works, you can try 'python src/ir/queryIndex_tfidf.py "感冒"', you will see results output
9. Download a pre-trained word embedding file from, put it at resources/w2v/
10. Train deep learning model and run './scripts/runs/train.sh'. It will take several hours
